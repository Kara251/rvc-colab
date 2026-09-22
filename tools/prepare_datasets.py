"""Pack each model's JP audio into a flat zip for upload to Drive."""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, load_manifest


def build_zip(voice_data: Path, zip_path: Path, char: str) -> int:
    tmp = zip_path.with_suffix(".zip.tmp")
    count = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for sub in sorted((voice_data / char / "audios").glob("jp_*")):
            if not sub.is_dir():
                continue
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() == ".ogg":
                    # Flat layout; prefix costume dir only on collision.
                    arc = f.name
                    if arc in zf.namelist():
                        arc = f"{sub.name}__{f.name}"
                    zf.write(f, arc)
                    count += 1
    tmp.replace(zip_path)
    return count


def main():
    cfg = load_config()
    voice_data = Path(cfg["voice_data"])
    out_dir = Path(cfg["datasets_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    models = load_manifest()

    done = skipped = 0
    for m in models:
        zip_path = out_dir / f"{m['name']}.zip"
        if zip_path.exists():
            skipped += 1
            continue
        n = build_zip(voice_data, zip_path, m["char"])
        done += 1
        print(f"{m['name']}: {n} files -> {zip_path.name} ({zip_path.stat().st_size/1e6:.1f} MB)")
    print(f"done={done} skipped={skipped} dir={out_dir}")


if __name__ == "__main__":
    main()
