"""Scan VoiceData and generate models.yaml (one model per character, JP only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import REPO_DIR, load_config

# Multi-speaker / non-character collections that must not become voice models.
EXCLUDE_DIRS = {"Main", "Tutorial"}
# Directory-name prefixes that are story compilations rather than a speaker.
EXCLUDE_PREFIXES = ("ST",)


def iter_models(voice_data: Path, cfg: dict):
    d = cfg["defaults"]
    for char_dir in sorted(p for p in voice_data.iterdir() if p.is_dir()):
        name = char_dir.name
        if name in EXCLUDE_DIRS or name.startswith(EXCLUDE_PREFIXES):
            continue
        ogg_files = [
            f
            for sub in char_dir.glob("audios/jp_*")
            if sub.is_dir()
            for f in sub.iterdir()
            if f.suffix.lower() == ".ogg"
        ]
        if not ogg_files:
            continue
        n = len(ogg_files)
        epochs = round(d["target_steps"] * d["batch_size"] / n)
        epochs = max(d["min_epochs"], min(d["max_epochs"], epochs))
        yield {
            "name": f"{name}_JP",
            "char": name,
            "files": n,
            "epochs": epochs,
            "sample_rate": d["sample_rate"],
            "batch_size": d["batch_size"],
            "save_every_epoch": d["save_every_epoch"],
            "f0_method": d["f0_method"],
            "embedder_model": d["embedder_model"],
            "vocoder": d["vocoder"],
        }


def main():
    import yaml

    cfg = load_config()
    models = list(iter_models(Path(cfg["voice_data"]), cfg))
    out = REPO_DIR / "models.yaml"
    out.write_text(
        "models:\n"
        + "".join(
            yaml.safe_dump([m], default_flow_style=False, sort_keys=False)
            for m in models
        ),
        encoding="utf-8",
    )
    total_files = sum(m["files"] for m in models)
    print(f"{len(models)} models, {total_files} jp files -> {out}")
    for m in models:
        print(f"  {m['name']:<28} {m['files']:>5} files  {m['epochs']:>4} epochs")


if __name__ == "__main__":
    main()
