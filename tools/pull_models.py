"""Collect exported models into the local RVC model library.

Download Drive's RVC-Train/exported/ folder to local disk first (web UI or
rclone), then run:

    python tools/pull_models.py --src F:/Dev/AI-Models/exported

Each exported model dir (<Char>_JP/) is copied into RVC/<Char>/ as
<Char>_JP.pth / <Char>_JP.index; if that name is taken, _v2, _v3, ...
is appended. Existing files are never touched.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_manifest


def free_name(folder: Path, stem: str, ext: str) -> Path:
    cand = folder / f"{stem}{ext}"
    v = 2
    while cand.exists():
        cand = folder / f"{stem}_v{v}{ext}"
        v += 1
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="local copy of Drive's exported/ dir")
    ap.add_argument("--dst", default=r"F:\Dev\AI-Models\RVC")
    args = ap.parse_args()

    src, dst_root = Path(args.src), Path(args.dst)
    models = {m["name"]: m for m in load_manifest()}

    pulled = 0
    for name in sorted(p.name for p in src.iterdir() if p.is_dir()):
        if name not in models:
            print(f"  skip {name} (not in manifest)")
            continue
        dst_dir = dst_root / models[name]["char"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in (src / name).iterdir():
            if f.suffix not in (".pth", ".index"):
                continue
            target = free_name(dst_dir, name, f.suffix)
            target.write_bytes(f.read_bytes())
            pulled += 1
            print(f"  {f.name} -> {target}")
    print(f"pulled {pulled} files")


if __name__ == "__main__":
    main()
