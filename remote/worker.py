"""Single-model worker running ON the Colab VM.

Reads /content/model.json (one model config) and /content/rvc_config.json.
Pipeline: wait for dataset zip -> setup Applio -> preprocess -> extract ->
train -> done. All state is transient; the orchestrator mirrors checkpoints
to local disk and re-uploads them when a session is recreated.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

BASE = Path("/content")
APPLIO_DIR = BASE / "Applio"
WORK_DIR = BASE / "rvc_work"
UPLOAD_DIR = WORK_DIR / "upload"
DATASET_DIR = WORK_DIR / "datasets"
STATUS = WORK_DIR / "status.json"
LOG = WORK_DIR / "worker.log"

CONFIG = json.loads((BASE / "rvc_config.json").read_text())
MODEL = json.loads((BASE / "model.json").read_text())
NAME = MODEL["name"]
LOGS = APPLIO_DIR / "logs" / NAME


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


STATE = {"phase": "boot", "model": NAME, "epoch": 0}


def set_status(**kw):
    STATE.update(kw)
    STATE["updated"] = time.time()
    STATE["run"] = MODEL.get("run")
    STATUS.write_text(json.dumps(STATE))


def run(cmd, cwd=None):
    log("$ " + " ".join(map(str, cmd)))
    return subprocess.run([str(c) for c in cmd], cwd=cwd).returncode


def max_epoch():
    best = 0
    for f in LOGS.glob("*e_*s.pth") if LOGS.is_dir() else []:
        m = re.search(r"_(\d+)e_\d+s\.pth$", f.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def ensure_dataset():
    local = DATASET_DIR / NAME
    if local.is_dir() and any(local.iterdir()):
        return local
    set_status(phase="waiting_dataset")
    log(f"waiting for {NAME}.zip")
    while True:
        for z in UPLOAD_DIR.glob(f"{NAME}.zip*"):
            try:
                local.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(local)
                z.unlink()
                set_status(phase="dataset_ready")
                return local
            except (zipfile.BadZipFile, OSError) as e:
                log(f"unzip {z.name}: {e}")
                time.sleep(5)
        time.sleep(10)


def prune_corrupt():
    if not LOGS.is_dir():
        return
    import torch
    for _ in range(4):
        pairs = []
        for f in LOGS.glob("G_*.pth"):
            s = f.stem.split("_", 1)[1]
            if (LOGS / f"D_{s}.pth").exists():
                pairs.append((int(s), f, LOGS / f"D_{s}.pth"))
        if not pairs:
            return
        step, g, d = max(pairs)
        try:
            torch.load(g, map_location="cpu")
            torch.load(d, map_location="cpu")
            return
        except Exception as e:
            log(f"corrupt ckpt step {step}: {e}; removing")
            g.unlink(missing_ok=True)
            d.unlink(missing_ok=True)


def applio(*args):
    return run([sys.executable, "core.py", *args], cwd=APPLIO_DIR)


def setup_applio():
    if not (APPLIO_DIR / "core.py").exists():
        run(["git", "clone", CONFIG["applio_repo"], str(APPLIO_DIR)])
        run(["git", "checkout", CONFIG["applio_commit"]], cwd=APPLIO_DIR)
    if not (WORK_DIR / ".deps_ok").exists():
        set_status(phase="deps")
        run([sys.executable, "-m", "pip", "install", "-q",
             "-r", "requirements.txt"], cwd=APPLIO_DIR)
        (WORK_DIR / ".deps_ok").write_text("1")
    # prerequisites skips already-downloaded files internally, run always
    if not (WORK_DIR / ".prereq_ok").exists():
        set_status(phase="prerequisites")
        applio("prerequisites", "--pretraineds-hifigan", "--models")
        (WORK_DIR / ".prereq_ok").write_text("1")


def train_and_watch():
    cmd = [sys.executable, "core.py", "train", "--model-name", NAME,
           "--save-every-epoch", str(MODEL["save_every_epoch"]),
           "--total-epoch", str(MODEL["epochs"]),
           "--sample-rate", str(MODEL["sample_rate"]),
           "--batch-size", str(MODEL["batch_size"]),
           "--gpu", str(MODEL.get("gpu", "0")),
           "--save-only-latest"]
    log("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=APPLIO_DIR)
    while p.poll() is None:
        set_status(phase="train", epoch=max_epoch())
        time.sleep(30)
    return p.returncode


def main():
    for d in (WORK_DIR, UPLOAD_DIR, DATASET_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log(f"worker start: {NAME} target={MODEL['epochs']}")
    setup_applio()

    # restore checkpoint mirror pushed by the orchestrator
    restore = WORK_DIR / "restore" / NAME
    if restore.is_dir():
        LOGS.mkdir(parents=True, exist_ok=True)
        for f in restore.iterdir():
            shutil.copy2(f, LOGS / f.name)
        shutil.rmtree(restore, ignore_errors=True)
        log(f"restored {len(list(LOGS.iterdir()))} files")

    ensure_dataset()
    prune_corrupt()

    set_status(phase="preprocess")
    if applio("preprocess", "--model-name", NAME,
              "--dataset-path", DATASET_DIR / NAME,
              "--sample-rate", MODEL["sample_rate"],
              "--cpu-cores", os.cpu_count() or 4) != 0:
        set_status(phase="failed", error="preprocess")
        return
    shutil.rmtree(DATASET_DIR / NAME, ignore_errors=True)

    set_status(phase="extract")
    if applio("extract", "--model-name", NAME, "--f0-method", MODEL["f0_method"],
              "--sample-rate", MODEL["sample_rate"],
              "--embedder-model", MODEL["embedder_model"],
              "--gpu", MODEL.get("gpu", "0")) != 0:
        set_status(phase="failed", error="extract")
        return

    rc = train_and_watch()
    # train.py runs epochs in a child process that can die silently (rc 0)
    # e.g. "not enough data" -> no weight files despite "success"
    produced = list(LOGS.glob("*e_*s.pth")) if LOGS.is_dir() else []
    if rc != 0 or not produced:
        set_status(phase="failed",
                   error=f"train rc={rc}" if rc != 0 else "no_weights",
                   epoch=max_epoch())
        return
    set_status(phase="trained", epoch=max_epoch())
    log(f"done: {NAME} epoch={max_epoch()}")


if __name__ == "__main__":
    main()
