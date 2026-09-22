"""Local orchestrator: drives a Colab session through the training queue.

Loop: ensure session -> mount Drive -> upload worker+manifest -> spawn worker
detached -> poll status -> upload dataset zips on demand -> on session death,
recreate and repeat until every model reaches its target epoch.

Usage: python run_cloud.py [--gpu T4] [--once]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from common import REPO_DIR, load_config, load_manifest


def colab(*args, timeout=120, capture=True):
    cmd = ["colab", *args]
    r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    return r


def session_alive(name):
    r = colab("status", "-s", name)
    return r.returncode == 0 and "not found" not in (r.stdout + r.stderr).lower()


def ensure_session(name, gpu):
    if session_alive(name):
        print(f"session {name} alive")
        return True
    print(f"creating session {name} gpu={gpu}")
    r = colab("new", "-s", name, "--gpu", gpu, timeout=300)
    if r.returncode != 0:
        print("new failed:", r.stdout, r.stderr)
        return False
    r = colab("drivemount", "-s", name, timeout=180)
    if r.returncode != 0:
        print("drivemount failed:", r.stdout, r.stderr)
        return False
    return True


def push_files(name):
    files = [
        (REPO_DIR / "remote" / "worker.py", "/content/worker.py"),
        (REPO_DIR / "remote" / "bootstrap.py", "/content/bootstrap.py"),
        (REPO_DIR / "remote" / "poll.py", "/content/poll.py"),
    ]
    cfg = load_config()
    (REPO_DIR / ".state").mkdir(exist_ok=True)
    cfg_json = REPO_DIR / ".state" / "rvc_config.json"
    cfg_json.write_text(json.dumps({
        "applio_repo": cfg["applio_repo"],
        "applio_commit": cfg["applio_commit"],
        "drive_root": cfg["drive_root"],
    }))
    man_json = REPO_DIR / ".state" / "models.json"
    man_json.write_text(json.dumps({"models": load_manifest()}))
    files += [(cfg_json, "/content/rvc_config.json"), (man_json, "/content/models.json")]
    for local, remote in files:
        r = colab("upload", "-s", name, str(local), remote, timeout=300)
        if r.returncode != 0:
            print(f"upload {local.name} failed:", r.stdout, r.stderr)
            return False
    return True


def bootstrap(name):
    r = colab("exec", "-s", name, "-f", str(REPO_DIR / "remote" / "bootstrap.py"), timeout=60)
    out = r.stdout + r.stderr
    print("bootstrap:", out.strip().splitlines()[-1] if out.strip() else "?")
    return "BOOTSTRAP::" in out


def poll_status(name):
    r = colab("exec", "-s", name, "-f", str(REPO_DIR / "remote" / "poll.py"), timeout=60)
    for line in r.stdout.splitlines():
        if line.startswith("STATUS::"):
            try:
                return json.loads(line[len("STATUS::"):])
            except json.JSONDecodeError:
                return None
    return None


def upload_dataset(name, model, datasets_dir):
    z = datasets_dir / f"{model}.zip"
    if not z.exists():
        print(f"  !! missing local zip {z}; run tools/prepare_datasets.py")
        return False
    print(f"  uploading {z.name} ({z.stat().st_size/1e6:.0f} MB)")
    r = colab("upload", "-s", name, str(z), f"/content/rvc_work/upload/{z.name}", timeout=3600)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--once", action="store_true", help="single attempt, no relaunch loop")
    args = ap.parse_args()

    cfg = load_config()
    name = cfg["colab"]["session_name"]
    gpu = args.gpu or cfg["colab"]["gpu"]
    poll = cfg["colab"]["poll_seconds"]
    datasets_dir = Path(cfg["datasets_dir"])

    while True:
        if not ensure_session(name, gpu):
            print("session unavailable; retry in 300s")
            if args.once:
                return 2
            time.sleep(300)
            continue
        if not push_files(name) or not bootstrap(name):
            print("bootstrap failed; retry in 120s")
            if args.once:
                return 2
            time.sleep(120)
            continue

        while True:
            st = poll_status(name)
            if st is None:
                print("poll failed -> session likely dead")
                break
            waiting = st.get("waiting_dataset")
            if waiting:
                upload_dataset(name, waiting, datasets_dir)
                continue
            done = len(st.get("done", []))
            failed = len(st.get("failed", {}))
            print(f"[{time.strftime('%H:%M:%S')}] phase={st.get('phase')} "
                  f"model={st.get('model')} epoch={st.get('epoch')} "
                  f"done={done} failed={failed}")
            if st.get("phase") == "all_done":
                print("queue finished")
                return 0
            time.sleep(poll)

        if args.once:
            return 2
        print("session lost; relaunching in 60s")
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
