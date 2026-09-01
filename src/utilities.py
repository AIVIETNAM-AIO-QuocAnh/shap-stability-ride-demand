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


def load_data(fold: str, variant: str):
    with open(PROJECT_ROOT / "data/processed/variant_feature_map.json") as file:
        variant_map = json.load(file)
    with open(PROJECT_ROOT / "data/processed/frozen/top50_zones_frozen.json") as f:
        zone_ids = json.load(f)["zone_ids"]

    feature_cols = (
        variant_map["base_features"]
        + variant_map["variants"][variant]["weekly_features"]
    )

    fold_dir = PROJECT_ROOT / "data/folds" / fold
    evaluation_file = "test.csv" if fold == "final_test" else "val.csv"

    train = pd.read_csv(
        fold_dir / "train.csv", parse_dates=["target_datetime"]
    )
    evaluation = pd.read_csv(
        fold_dir / evaluation_file, parse_dates=["target_datetime"]
    )

    def add_zone_onehot(df, zone_ids):
        df = df.copy()
        df["PULocationID"] = pd.Categorical(df["PULocationID"], categories=zone_ids)
        dummies = pd.get_dummies(df["PULocationID"], prefix="zone")
        dummies = dummies.reindex(
            columns=[f"zone_{zone_id}" for zone_id in zone_ids],
            fill_value=0,
        )
        return pd.concat([df, dummies], axis=1), dummies.columns.tolist()

    train, zone_cols = add_zone_onehot(train, zone_ids)
    evaluation, _ = add_zone_onehot(evaluation, zone_ids)

    X_train = train[feature_cols + zone_cols].copy()
    y_train = train["demand"]
    X_evaluation = evaluation[feature_cols + zone_cols].copy()
    y_evaluation = evaluation["demand"]

    return X_train, y_train, X_evaluation, y_evaluation

def calculate_metrics(metric, y_true, y_pred):
    if metric == "wape":
        return ((y_true - y_pred) / y_true).abs().mean()
    elif metric == "rmse":
        return ((y_true - y_pred) ** 2).mean() ** 0.5
    elif metric == "mae":
        return (y_true - y_pred).abs().mean()
    else:
        raise ValueError(f"Unsupported metric: {metric}")
