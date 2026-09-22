"""Pull exported models from Drive into the local RVC model library.

Exported files land in <drive_root>/exported/<Model>/ on Drive, reachable
through the mounted session at /content/drive/MyDrive/... . Local naming:
RVC/<Char>/<Char>_JP.pth/.index; if the folder already contains a same-named
file, the new file is suffixed _v2, _v3, ...

Requires a live Colab session (run_cloud.py keeps one up; or `colab new`).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import REPO_DIR, load_config, load_manifest


def colab(*args, timeout=300):
    return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout)


def remote_ls(session, path):
    script = REPO_DIR / ".state" / "ls.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text(
        "import os,json\n"
        f"p={json.dumps(path)}\n"
        "print('LS::'+json.dumps(sorted(os.listdir(p)) if os.path.isdir(p) else []))\n"
    )
    r = colab("exec", "-s", session, "-f", str(script), timeout=60)
    for line in r.stdout.splitlines():
        if line.startswith("LS::"):
            return json.loads(line[4:])
    return None


def free_name(folder: Path, stem: str, ext: str) -> Path:
    cand = folder / f"{stem}{ext}"
    v = 2
    while cand.exists():
        cand = folder / f"{stem}_v{v}{ext}"
        v += 1
    return cand


def main():
    cfg = load_config()
    session = cfg["colab"]["session_name"]
    rvc_dir = Path(cfg["rvc_models_dir"])
    drive = f"/content/drive/MyDrive/{cfg['drive_root']}"

    models = {m["name"]: m for m in load_manifest()}
    exported = remote_ls(session, f"{drive}/exported") or []
    print(f"{len(exported)} exported model dirs on Drive")

    pulled = 0
    for name in exported:
        if name not in models:
            print(f"  skip {name} (not in manifest)")
            continue
        char = models[name]["char"]
        files = remote_ls(session, f"{drive}/exported/{name}") or []
        dst_dir = rvc_dir / char
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            ext = Path(f).suffix
            if ext not in (".pth", ".index"):
                continue
            # local stem: <Char>_JP ; suffix _v2+ if name taken
            target = free_name(dst_dir, name, ext)
            r = colab("download", "-s", session,
                      f"{drive}/exported/{name}/{f}", str(target), timeout=3600)
            if r.returncode == 0:
                pulled += 1
                print(f"  {f} -> {target}")
            else:
                print(f"  FAIL {f}: {r.stderr.strip()}")
    print(f"pulled {pulled} files")


if __name__ == "__main__":
    main()
