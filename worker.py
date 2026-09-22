"""RVC training queue worker. Runs inside a Colab session.

Reads RVC-Train/models.yaml from Google Drive, trains each pending model with
Applio, syncs checkpoints to Drive every SYNC_SEC seconds, and exports finished
weights to RVC-Train/exported/. Safe to re-run: state lives on Drive, so a new
session picks up where the last one died.

Usage (from the notebook cell):
    python worker.py
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

import yaml

DRIVE_ROOT = os.environ.get("RVC_DRIVE_ROOT", "/content/drive/MyDrive/RVC-Train")
WORK = "/content/rvc_work"
APPLIO_DIR = "/content/Applio"
APPLIO_REPO = "https://github.com/IAHispano/Applio.git"
APPLIO_REF = "55fe0b976a6990bb75261c32ccecf6bfca3198f1"
SYNC_SEC = 60

DATASETS = os.path.join(DRIVE_ROOT, "datasets")
LOGS_DRIVE = os.path.join(DRIVE_ROOT, "logs")
EXPORTED = os.path.join(DRIVE_ROOT, "exported")
MANIFEST = os.path.join(DRIVE_ROOT, "models.yaml")
PRETRAINED = os.path.join(APPLIO_DIR, "rvc/models/pretraineds/hifi-gan")

_stop = threading.Event()


def sh(args, **kw):
    kw.setdefault("cwd", APPLIO_DIR)
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, **kw)


def have_weights():
    return os.path.isdir(PRETRAINED) and any(f.endswith(".pth") for f in os.listdir(PRETRAINED))


def ensure_applio():
    if not os.path.isdir(os.path.join(APPLIO_DIR, "rvc")):
        subprocess.run(["git", "clone", "--depth", "1", APPLIO_REPO, APPLIO_DIR], check=True)
        subprocess.run(["git", "fetch", "--depth", "1", "origin", APPLIO_REF], cwd=APPLIO_DIR, check=True)
        subprocess.run(["git", "checkout", APPLIO_REF], cwd=APPLIO_DIR, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
                   cwd=APPLIO_DIR, check=True)
    if not have_weights():
        subprocess.run([sys.executable, "core.py", "prerequisites",
                        "--pretraineds-hifigan", "--models"], cwd=APPLIO_DIR, check=True)


def max_epoch(logdir):
    m = 0
    for d in (logdir,):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            mm = re.match(r"[GD]_.*?_(\d+)e_", f)
            if mm:
                m = max(m, int(mm.group(1)))
    return m


def sync_up(model):
    """Copy local logs/<model> to Drive logs/<model> (new or changed files only)."""
    src = os.path.join(APPLIO_DIR, "logs", model)
    dst = os.path.join(LOGS_DRIVE, model)
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if not os.path.isfile(s):
            continue
        if not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(s):
            try:
                shutil.copyfile(s, d)
            except OSError:
                pass  # file still being written; next tick picks it up


def sync_down(model):
    """Restore Drive logs/<model> into local logs/<model>."""
    src = os.path.join(LOGS_DRIVE, model)
    dst = os.path.join(APPLIO_DIR, "logs", model)
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not os.path.exists(os.path.join(dst, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    return max_epoch(dst)


def sync_loop(model):
    while not _stop.wait(SYNC_SEC):
        sync_up(model)


def ensure_dataset(name):
    """Unzip Drive datasets/<name>.zip into WORK/datasets/<name>/."""
    dst = os.path.join(WORK, "datasets", name)
    if os.path.isdir(dst):
        return dst
    zpath = os.path.join(DATASETS, name + ".zip")
    if not os.path.exists(zpath):
        return None
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dst)
    return dst


def train_one(m):
    name = m["name"]
    target = m.get("total_epoch") or m["epochs"]
    sr = m.get("sample_rate", 48000)
    bs = m.get("batch_size", 8)
    f0 = m.get("f0_method", "rmvpe")
    emb = m.get("embedder_model", "japanese-hubert-base")
    done_flag = os.path.join(LOGS_DRIVE, name, ".trained")

    if os.path.exists(done_flag) or max_epoch(os.path.join(LOGS_DRIVE, name)) >= target:
        print(f"[{name}] already done, skip", flush=True)
        return True

    ds = ensure_dataset(name)
    if not ds:
        print(f"[{name}] dataset zip missing on Drive, skip", flush=True)
        return False

    cur = sync_down(name)
    print(f"[{name}] resume at epoch {cur}/{target}", flush=True)

    t = threading.Thread(target=sync_loop, args=(name,), daemon=True)
    t.start()
    try:
        steps = [
            [sys.executable, "core.py", "preprocess", "--model-name", name,
             "--dataset-path", ds, "--sample-rate", str(sr)],
            [sys.executable, "core.py", "extract", "--model-name", name,
             "--f0-method", f0, "--sample-rate", str(sr),
             "--embedder-model", emb, "--gpu", "0"],
            [sys.executable, "core.py", "train", "--model-name", name,
             "--save-every-epoch", str(m.get("save_every_epoch", 10)),
             "--save-only-latest", "True", "--total-epoch", str(target),
             "--sample-rate", str(sr), "--batch-size", str(bs),
             "--gpu", "0", "--pretrained", "True", "--save-every-weights", "True"],
        ]
        for cmd in steps:
            if sh(cmd).returncode != 0:
                print(f"[{name}] step failed: {cmd[2]}", flush=True)
                return False
    finally:
        _stop.set()
        t.join()
        _stop.clear()
        sync_up(name)

    # export weights + index
    wdir = os.path.join(APPLIO_DIR, "logs", name)
    outdir = os.path.join(EXPORTED, name)
    os.makedirs(outdir, exist_ok=True)
    ok = False
    for f in os.listdir(wdir):
        if f.endswith(".pth") and f.startswith(name) or f.startswith("added_"):
            shutil.copyfile(os.path.join(wdir, f), os.path.join(outdir, f))
            ok = True
    if ok:
        open(done_flag, "w").write(str(target))
    print(f"[{name}] {'done' if ok else 'trained but no weights found'}", flush=True)

    shutil.rmtree(os.path.join(WORK, "datasets", name), ignore_errors=True)
    shutil.rmtree(wdir, ignore_errors=True)  # Drive copy is canonical
    return ok


def main():
    if not os.path.isdir(DRIVE_ROOT):
        sys.exit(f"Drive folder not found: {DRIVE_ROOT} — mount Drive and create it first")
    os.makedirs(LOGS_DRIVE, exist_ok=True)
    os.makedirs(EXPORTED, exist_ok=True)

    ensure_applio()

    with open(MANIFEST, encoding="utf-8") as f:
        models = yaml.safe_load(f)["models"]
    pending = [m for m in models
               if not os.path.exists(os.path.join(LOGS_DRIVE, m["name"], ".trained"))]
    print(f"queue: {len(pending)}/{len(models)} models pending", flush=True)

    failed = []
    for m in pending:
        try:
            if not train_one(m):
                failed.append(m["name"])
        except Exception as e:
            print(f"[{m['name']}] error: {e}", flush=True)
            failed.append(m["name"])

    print(f"ALL DONE. failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
