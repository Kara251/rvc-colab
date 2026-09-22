"""Shared config loading for rvc-colab tools."""
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_DIR / "config.yaml"
MANIFEST_PATH = REPO_DIR / "models.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("models", []) if isinstance(data, dict) else data
