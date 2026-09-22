"""Worker that runs ON the Colab VM.

Lifecycle: started detached by the local orchestrator. Sets up Applio, then
walks the model manifest: dataset upload request -> preprocess -> extract ->
train (resuming from the newest checkpoint synced to Drive) -> export.

Persistent state lives on Google Drive under <drive_root>:
    logs/<model>/       checkpoints + trained markers only
    exported/<model>/   final weights + index
    cache/              pretrained/embedder/mute assets (reused across sessions)
    status.json         last-known worker status (post-mortem aid)
Datasets are uploaded per model by the orchestrator into UPLOAD_DIR and live
only on the VM's local disk.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

BASE = Path("/content")
APPLIO_DIR = BASE / "Applio"
WORK_DIR = BASE / "rvc_work"          # local working dir: datasets, status
UPLOAD_DIR = WORK_DIR / "upload"      # orchestrator drops <Model>.zip here
DATASET_DIR = WORK_DIR / "datasets"   # unzipped per-model ogg dirs
STATUS_LOCAL = WORK_DIR / "status.json"
WORKER_LOG = WORK_DIR / "worker.log"

DRIVE = Path("/content/drive/MyDrive")

CONFIG = json.loads((BASE / "rvc_config.json").read_text())
MANIFEST = json.loads((BASE / "models.json").read_text())["models"]

DRIVE_ROOT = DRIVE / CONFIG["drive_root"]
DRIVE_LOGS = DRIVE_ROOT / "logs"
DRIVE_EXPORT = DRIVE_ROOT / "exported"
DRIVE_CACHE = DRIVE_ROOT / "cache"
DRIVE_STATUS = DRIVE_ROOT / "status.json"

SYNC_PATTERNS = re.compile(r"^(G|D)_\d+\.pth$|^\d+e_\d+s\.pth$|\.index$|config\.json$|\.trained$")
STOP = threading.Event()
STATE = {"phase": "boot", "model": None, "epoch": 0, "waiting_dataset": None,
         "done": [], "failed": {}, "updated": time.time()}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with WORKER_LOG.open("a") as f:
        f.write(line + "\n")


def set_status(**kw):
    STATE.update(kw)
    STATE["updated"] = time.time()
    STATUS_LOCAL.write_text(json.dumps(STATE, indent=1))
    try:
        DRIVE_STATUS.write_text(json.dumps(STATE, indent=1))
    except OSError:
        pass


def run(cmd, cwd=None):
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.run([str(c) for c in cmd], cwd=cwd).returncode


# ---------- Drive <-> local sync ----------

def sync_up(model):
    """Copy new checkpoint artifacts local -> Drive."""
    src = APPLIO_DIR / "logs" / model
    dst = DRIVE_LOGS / model
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file() or not SYNC_PATTERNS.match(f.name):
            continue
        d = dst / f.name
        if not d.exists() or d.stat().st_size != f.stat().st_size:
            try:
                shutil.copy2(f, d)
            except OSError as e:
                log(f"sync_up {f.name}: {e}")


def sync_down(model):
    """Restore Drive checkpoints -> local before resuming."""
    src = DRIVE_LOGS / model
    dst = APPLIO_DIR / "logs" / model
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            d = dst / f.name
            if not d.exists() or d.stat().st_size != f.stat().st_size:
                shutil.copy2(f, d)


def sync_loop():
    while not STOP.is_set():
        m = STATE.get("model")
        if m:
            sync_up(m)
        STOP.wait(60)


def drive_epoch(model):
    d = DRIVE_LOGS / model
    best = 0
    for f in d.glob("*e_*s.pth") if d.is_dir() else []:
        m = re.search(r"_(\d+)e_\d+s\.pth$", f.name)
        if m:
            best = max(best, int(m.group(1)))
    if (d / ".trained").exists() if d.is_dir() else False:
        best = max(best, 10**9)
    return best


# ---------- phases ----------

def ensure_dataset(model):
    """Block until the orchestrator has uploaded the dataset zip."""
    local = DATASET_DIR / model
    if local.is_dir() and any(local.iterdir()):
        return local
    set_status(waiting_dataset=model)
    log(f"waiting for dataset {model}.zip")
    while True:
        for z in UPLOAD_DIR.glob(f"{model}*.zip"):
            try:
                local.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(local)
                z.unlink()
                log(f"dataset {model} ready ({sum(1 for _ in local.iterdir())} files)")
                set_status(waiting_dataset=None)
                return local
            except (zipfile.BadZipFile, OSError) as e:
                log(f"unzip {z.name}: {e} (retry)")
                time.sleep(5)
        time.sleep(10)


def applio_cmd(model, *args):
    return run([sys.executable, "core.py", *args], cwd=APPLIO_DIR)


def max_weight_epoch(model):
    """Newest epoch from weight files in the local logs dir."""
    d = APPLIO_DIR / "logs" / model
    return max(
        (int(m.group(1)) for f in d.glob("*e_*s.pth") if (m := re.search(r"_(\d+)e_", f.name))),
        default=0,
    ) if d.is_dir() else 0


def prune_corrupt_checkpoints(model):
    """Drop the newest G_/D_ pair if it fails to load (died mid-write)."""
    d = APPLIO_DIR / "logs" / model
    if not d.is_dir():
        return
    import torch
    for _ in range(4):
        pairs = []
        for f in d.glob("G_*.pth"):
            s = f.stem.split("_", 1)[1]
            if (d / f"D_{s}.pth").exists():
                pairs.append((int(s), f, d / f"D_{s}.pth"))
        if not pairs:
            return
        step, g, dd = max(pairs)
        try:
            torch.load(g, map_location="cpu")
            torch.load(dd, map_location="cpu")
            return
        except Exception as e:
            log(f"corrupt checkpoint step {step}: {e}; removing pair")
            g.unlink(missing_ok=True)
            dd.unlink(missing_ok=True)


def export(model):
    d = APPLIO_DIR / "logs" / model
    out = DRIVE_EXPORT / model
    out.mkdir(parents=True, exist_ok=True)
    weights = sorted(d.glob("*e_*s.pth"), key=lambda f: f.stat().st_mtime)
    if weights:
        shutil.copy2(weights[-1], out / weights[-1].name)
    for idx in d.glob("*.index"):
        shutil.copy2(idx, out / idx.name)
    log(f"exported {model}: {[p.name for p in out.iterdir()]}")


def process_model(m):
    name = m["name"]
    target = m["epochs"]
    set_status(model=name, phase="dataset", epoch=max_weight_epoch(name))
    ensure_dataset(name)

    sync_down(name)
    prune_corrupt_checkpoints(name)

    set_status(phase="preprocess")
    if applio_cmd(name, "preprocess", "--model-name", name, "--dataset-path",
                  DATASET_DIR / name, "--sample-rate", m["sample_rate"],
                  "--cpu-cores", os.cpu_count() or 4) != 0:
        raise RuntimeError("preprocess failed")

    set_status(phase="extract")
    if applio_cmd(name, "extract", "--model-name", name,
                  "--f0-method", m["f0_method"], "--sample-rate", m["sample_rate"],
                  "--embedder-model", m["embedder_model"], "--gpu", m.get("gpu", "0")) != 0:
        raise RuntimeError("extract failed")

    set_status(phase="train", epoch=max_weight_epoch(name))
    rc = applio_cmd(name, "train", "--model-name", name,
                    "--save-every-epoch", m["save_every_epoch"],
                    "--total-epoch", target,
                    "--sample-rate", m["sample_rate"],
                    "--batch-size", m["batch_size"], "--gpu", m.get("gpu", "0"))
    sync_up(name)
    if rc != 0:
        raise RuntimeError("train failed")

    (DRIVE_LOGS / name / ".trained").write_text(str(target))
    set_status(phase="export")
    export(name)
    STATE["done"].append(name)
    log(f"model {name} finished at target {target} epochs")


# ---------- setup ----------

def setup_applio():
    if not (APPLIO_DIR / "core.py").exists():
        run(["git", "clone", CONFIG["applio_repo"], APPLIO_DIR])
        run(["git", "checkout", CONFIG["applio_commit"]], cwd=APPLIO_DIR)
    set_status(phase="deps")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=APPLIO_DIR)

    # restore cached assets (pretraineds, embedders, mute files)
    for sub in ("rvc/models", "assets", "logs/mute"):
        cache = DRIVE_CACHE / sub
        dest = APPLIO_DIR / sub
        if cache.is_dir() and not dest.exists():
            shutil.copytree(cache, dest)
            log(f"restored cache {sub}")

    set_status(phase="prerequisites")
    applio_cmd("_setup", "prerequisites", "--pretraineds-hifigan", "--models")

    for sub in ("rvc/models", "assets", "logs/mute"):
        src = APPLIO_DIR / sub
        cache = DRIVE_CACHE / sub
        if src.is_dir() and not cache.exists():
            try:
                shutil.copytree(src, cache)
                log(f"cached {sub}")
            except OSError as e:
                log(f"cache {sub}: {e}")


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_LOGS.mkdir(parents=True, exist_ok=True)
    DRIVE_EXPORT.mkdir(parents=True, exist_ok=True)

    log("worker start")
    setup_applio()

    t = threading.Thread(target=sync_loop, daemon=True)
    t.start()

    set_status(phase="queue")
    for m in MANIFEST:
        name = m["name"]
        if drive_epoch(name) >= m["epochs"]:
            log(f"skip {name} (already >= {m['epochs']} epochs)")
            STATE["done"].append(name)
            continue
        try:
            process_model(m)
        except Exception as e:
            log(f"FAILED {name}: {e}")
            STATE["failed"][name] = str(e)
            set_status(model=None)

    set_status(phase="all_done", model=None, waiting_dataset=None)
    log("worker finished")


if __name__ == "__main__":
    main()
