"""Local orchestrator for RVC training on Colab.

Owns all persistent state: checkpoints are pulled to <checkpoints_dir>/<model>/
and pushed back when a session is recreated. Per model: upload dataset zip +
checkpoint mirror -> detached worker -> poll status + pull new checkpoints ->
on 'trained', fetch weights/index into the RVC library. Session death just
re-runs the same model with the mirrored checkpoints.

Usage: python run_cloud.py [--gpu T4] [--model NAME] [--once]
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from common import REPO_DIR, load_config, load_manifest

# files mirrored continuously vs only at export time
CKPT_RE = re.compile(r"^(G|D)_\d+\.pth$|^\d+e_\d+s\.pth$|\.index$|config\.json$")
LIVE_RE = re.compile(r"^(G|D)_\d+\.pth$|^config\.json$")


def colab(*args, timeout=120):
    return subprocess.run(["colab", *args], capture_output=True, text=True,
                          timeout=timeout)


class Session:
    def __init__(self, name, gpu):
        self.name, self.gpu = name, gpu

    def alive(self):
        r = colab("status", "-s", self.name)
        return r.returncode == 0 and "not found" not in (r.stdout + r.stderr).lower()

    def ensure(self):
        if self.alive():
            return True
        print(f"[sess] creating {self.name} gpu={self.gpu}")
        r = colab("new", "-s", self.name, "--gpu", self.gpu, timeout=600)
        ok = r.returncode == 0
        print(f"[sess] {'ready' if ok else 'FAILED: ' + (r.stdout + r.stderr)[-300:]}")
        return ok

    def exec(self, script: str, timeout=60):
        p = REPO_DIR / ".state" / "exec_tmp.py"
        p.parent.mkdir(exist_ok=True)
        p.write_text(script)
        return colab("exec", "-s", self.name, "-f", str(p), timeout=timeout)

    def upload(self, local: Path, remote: str, timeout=3600):
        return colab("upload", "-s", self.name, str(local), remote,
                     timeout=timeout).returncode == 0

    def download(self, remote: str, local: Path, timeout=3600):
        local.parent.mkdir(parents=True, exist_ok=True)
        return colab("download", "-s", self.name, remote, str(local),
                     timeout=timeout).returncode == 0

    def remote_files(self, path: str):
        """{name: size} for files directly under remote dir."""
        r = self.exec(
            "import os,json\n"
            f"p={json.dumps(path)}\n"
            "try:\n"
            " d={f:[os.path.getsize(os.path.join(p,f)),"
            "       os.path.getmtime(os.path.join(p,f))] for f in os.listdir(p)\n"
            "    if os.path.isfile(os.path.join(p,f))}\n"
            "except OSError: d={}\n"
            "print('LS::'+json.dumps(d))\n"
        )
        for line in r.stdout.splitlines():
            if line.startswith("LS::"):
                try:
                    return json.loads(line[4:])
                except json.JSONDecodeError:
                    pass
        return None

    def status(self):
        r = self.exec(
            "import os\n"
            "p='/content/rvc_work/status.json'\n"
            "print('STATUS::'+(open(p).read() if os.path.exists(p) else '{}'))\n"
        )
        for line in r.stdout.splitlines():
            if line.startswith("STATUS::"):
                try:
                    return json.loads(line[8:])
                except json.JSONDecodeError:
                    pass
        return None


def push_inputs(sess: Session, cfg, model):
    state = REPO_DIR / ".state"
    state.mkdir(exist_ok=True)
    (state / "rvc_config.json").write_text(json.dumps({
        "applio_repo": cfg["applio_repo"],
        "applio_commit": cfg["applio_commit"],
    }))
    (state / "model.json").write_text(json.dumps(model))
    for local, remote in [
        (REPO_DIR / "remote" / "worker.py", "/content/worker.py"),
        (REPO_DIR / "remote" / "bootstrap.py", "/content/bootstrap.py"),
        (state / "rvc_config.json", "/content/rvc_config.json"),
        (state / "model.json", "/content/model.json"),
    ]:
        if not sess.upload(local, remote):
            return False
    return True


def push_checkpoints(sess: Session, mirror: Path, model_name):
    if not mirror.is_dir():
        return
    for f in mirror.iterdir():
        if f.is_file() and CKPT_RE.match(f.name):
            remote = f"/content/rvc_work/restore/{model_name}/{f.name}"
            if not sess.upload(f, remote):
                print(f"  ckpt push failed: {f.name}")


def pull_checkpoints(sess: Session, mirror: Path, model_name, full=False):
    remote = sess.remote_files(f"/content/Applio/logs/{model_name}")
    if remote is None:
        return
    pat = CKPT_RE if full else LIVE_RE
    for name, (size, mtime) in remote.items():
        if not pat.match(name):
            continue
        local = mirror / name
        if (not local.exists() or local.stat().st_size != size
                or abs(local.stat().st_mtime - mtime) > 2):
            print(f"  pull {name} ({size/1e6:.0f} MB)")
            sess.download(f"/content/Applio/logs/{model_name}/{name}", local)


def finalize_model(sess: Session, cfg, model, mirror: Path):
    """Copy final weights+index from mirror into the RVC library dir."""
    weights = sorted(mirror.glob("*e_*s.pth"), key=lambda f: f.stat().st_mtime)
    index = sorted(mirror.glob("*.index"), key=lambda f: f.stat().st_mtime)
    if not weights:
        print(f"  !! no weights pulled for {model['name']}")
        return False
    dst = Path(cfg["rvc_models_dir"]) / model["char"]
    dst.mkdir(parents=True, exist_ok=True)
    for src in [weights[-1], *(index[-1:] if index else [])]:
        ext = src.suffix
        target = dst / f"{model['name']}{ext}"
        v = 2
        while target.exists():
            target = dst / f"{model['name']}_v{v}{ext}"
            v += 1
        target.write_bytes(src.read_bytes())
        print(f"  exported -> {target}")
    (mirror / ".trained").write_text("1")
    sess.exec(f"import shutil;shutil.rmtree('/content/Applio/logs/{model['name']}',ignore_errors=True)")
    return True


def bootstrap(sess: Session):
    r = sess.exec((REPO_DIR / "remote" / "bootstrap.py").read_text(), timeout=60)
    out = r.stdout + r.stderr
    return "BOOTSTRAP::" in out


def train_model(sess: Session, cfg, model, mirror: Path, datasets_dir: Path, poll: int):
    """Train one model to completion, recreating the session as needed."""
    while True:
        if not sess.ensure():
            print("[sess] unavailable; retry 300s")
            time.sleep(300)
            continue
        model["run"] = str(int(time.time()))
        if not push_inputs(sess, cfg, model):
            time.sleep(60)
            continue
        # upload API 500s if the parent dir does not exist yet
        sess.exec("import os;os.makedirs('/content/rvc_work/upload',exist_ok=True);"
                  f"os.makedirs('/content/rvc_work/restore/{model['name']}',exist_ok=True)")
        push_checkpoints(sess, mirror, model["name"])
        z = datasets_dir / f"{model['name']}.zip"
        if not z.exists():
            print(f"  !! {z} missing (run tools/prepare_datasets.py)")
            return False
        print(f"  upload {z.name} ({z.stat().st_size/1e6:.0f} MB)")
        if not sess.upload(z, f"/content/rvc_work/upload/{z.name}"):
            print("  dataset upload failed; recreating session")
            time.sleep(30)
            continue
        if not bootstrap(sess):
            time.sleep(60)
            continue

        last_seen = 0.0
        stale = 0
        pull_checkpoints.last = 0.0
        while True:
            st = sess.status()
            if st is None:
                print("[sess] lost; will recreate")
                break
            # ignore status files left over from a previous model
            if st.get("model") not in (None, model["name"]):
                stale += 1
                if stale > 30:
                    print("[poll] no matching status; re-bootstrapping")
                    break
                time.sleep(10)
                continue
            # adopt a live worker from an earlier bootstrap of the same model
            if st.get("run") and st.get("run") != model["run"]:
                print(f"[poll] adopting worker run={st['run']}")
                model["run"] = st["run"]
            stale = 0
            if time.time() - getattr(pull_checkpoints, "last", 0) > cfg["colab"].get("ckpt_pull_seconds", 900):
                pull_checkpoints(sess, mirror, model["name"])
                pull_checkpoints.last = time.time()
            upd = st.get("updated", 0)
            if upd == last_seen and time.time() - upd > 600:
                print("[poll] worker silent >10min; re-bootstrapping")
                break
            last_seen = upd
            print(f"[{time.strftime('%H:%M:%S')}] {model['name']} "
                  f"phase={st.get('phase')} epoch={st.get('epoch')}")
            if st.get("phase") in ("trained", "failed"):
                pull_checkpoints(sess, mirror, model["name"], full=True)
                if st["phase"] == "trained":
                    return finalize_model(sess, cfg, model, mirror)
                print(f"  !! worker failed: {st.get('error')}")
                return False
            time.sleep(poll)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu")
    ap.add_argument("--model", help="train only this model name")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    sess = Session(cfg["colab"]["session_name"], args.gpu or cfg["colab"]["gpu"])
    poll = cfg["colab"]["poll_seconds"]
    ckpt_root = Path(cfg["checkpoints_dir"])
    datasets_dir = Path(cfg["datasets_dir"])

    manifest = load_manifest()
    if args.model:
        manifest = [m for m in manifest if m["name"] == args.model]
    pending = [m for m in manifest
               if not (ckpt_root / m["name"] / ".trained").exists()]
    print(f"{len(manifest)} models, {len(pending)} pending")

    for m in pending:
        mirror = ckpt_root / m["name"]
        mirror.mkdir(parents=True, exist_ok=True)
        print(f"=== {m['name']} target={m['epochs']} files={m['files']} ===")
        ok = train_model(sess, cfg, m, mirror, datasets_dir, poll)
        print(f"=== {m['name']} {'done' if ok else 'FAILED'} ===")
        if args.once and not ok:
            return 2
    print("queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
