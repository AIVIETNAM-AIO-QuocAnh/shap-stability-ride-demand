from pathlib import Path
import yaml
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def resolve_path(cfg: dict, key: str) -> Path:

    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Config is missing the 'paths' section.")
    if key not in paths:
        available = ", ".join(sorted(paths)) or "(empty)"
        raise KeyError(f"Path '{key}' not found in config. Available: {available}")

    return PROJECT_ROOT / paths[key]
