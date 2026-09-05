import json

import pandas as pd

from src.configuration import load_data_config
from src.data.build_panel import add_zone_onehot

def load_data(fold: str, variant: str):
    """Load one configured temporal fold and variant with stable zone columns."""
    config = load_data_config()
    paths = config["paths"]
    with open(paths["variant_map"]) as file:
        variant_map = json.load(file)
    with open(paths["frozen_zones"]) as f:
        zone_ids = json.load(f)["zone_ids"]

    feature_cols = (
        ["pu_location_id"]
        + variant_map["base_features"]
        + variant_map["variants"][variant]["weekly_features"]
    )

    fold_dir = paths["folds_dir"] / fold
    evaluation_file = "test.csv" if fold == "final_test" else "val.csv"

    train = pd.read_csv(
        fold_dir / "train.csv", parse_dates=["target_datetime"]
    )
    evaluation = pd.read_csv(
        fold_dir / evaluation_file, parse_dates=["target_datetime"]
    )

    train, zone_cols = add_zone_onehot(train, zone_ids)
    evaluation, _ = add_zone_onehot(evaluation, zone_ids)

    X_train = train[feature_cols + zone_cols].copy()
    y_train = train["demand"]
    X_evaluation = evaluation[feature_cols + zone_cols].copy()
    y_evaluation = evaluation["demand"]

    return X_train, y_train, X_evaluation, y_evaluation

def calculate_metrics(metric, y_true, y_pred):
    if metric == "wape":
        sum_abs_err = (y_true - y_pred).abs().sum()
        sum_abs_true = y_true.abs().sum()
        return sum_abs_err / sum_abs_true
    elif metric == "rmse":
        return ((y_true - y_pred) ** 2).mean() ** 0.5
    elif metric == "mae":
        return (y_true - y_pred).abs().mean()
    else:
        raise ValueError(f"Unsupported metric: {metric}")
