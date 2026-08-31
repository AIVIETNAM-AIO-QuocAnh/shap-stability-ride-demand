from pathlib import Path
import json

import yaml
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

def load_data(fold: str, variant: str | None = None):
    """Load one prepared train/evaluation fold for the modeling pipeline."""
    valid_folds = {"fold1", "fold2", "fold3", "fold4", "hpo", "final_test"}
    if fold not in valid_folds:
        available = ", ".join(sorted(valid_folds))
        raise ValueError(f"Invalid fold '{fold}'. Expected one of: {available}")

    fold_dir = PROJECT_ROOT / "data" / "folds" / fold
    evaluation_name = "test.csv" if fold == "final_test" else "val.csv"
    train_path = fold_dir / "train.csv"
    evaluation_path = fold_dir / evaluation_name

    for path in (train_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")

    train = pd.read_csv(train_path, parse_dates=["target_datetime"])
    evaluation = pd.read_csv(evaluation_path, parse_dates=["target_datetime"])

    required_columns = {"PULocationID", "target_datetime", "demand"}
    for name, frame in (("train", train), ("evaluation", evaluation)):
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} data is missing required columns: {', '.join(sorted(missing))}"
            )

    feature_map_path = PROJECT_ROOT / "data" / "processed" / "variant_feature_map.json"
    with open(feature_map_path, "r", encoding="utf-8") as file:
        feature_map = json.load(file)

    feature_columns = list(feature_map["base_features"])
    if variant is None:
        feature_columns += [
            column
            for column in train.columns
            if column not in required_columns and column not in feature_columns
        ]
    else:
        variants = feature_map.get("variants", {})
        if variant not in variants:
            available = ", ".join(sorted(variants))
            raise ValueError(
                f"Invalid variant '{variant}'. Expected one of: {available}"
            )
        feature_columns += variants[variant]["weekly_features"]

    for name, frame in (("train", train), ("evaluation", evaluation)):
        missing = set(feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} data is missing feature columns: {', '.join(sorted(missing))}"
            )

    # Build the same one-hot zone columns for both splits, including unseen zones.
    zones = sorted(set(train["PULocationID"]) | set(evaluation["PULocationID"]))
    zone_columns = [f"PULocationID_{zone}" for zone in zones]
    combined_zones = pd.concat(
        [train["PULocationID"], evaluation["PULocationID"]], ignore_index=True
    )
    zone_dummies = pd.get_dummies(combined_zones, prefix="PULocationID", dtype="int8")
    zone_dummies = zone_dummies.reindex(columns=zone_columns, fill_value=0)

    train_zones = zone_dummies.iloc[: len(train)].set_axis(train.index)
    evaluation_zones = zone_dummies.iloc[len(train) :].set_axis(evaluation.index)

    X_train = pd.concat(
        [train[feature_columns].reset_index(drop=True), train_zones.reset_index(drop=True)],
        axis=1,
    )
    X_evaluation = pd.concat(
        [
            evaluation[feature_columns].reset_index(drop=True),
            evaluation_zones.reset_index(drop=True),
        ],
        axis=1,
    )

    return X_train, train["demand"], X_evaluation, evaluation["demand"]
