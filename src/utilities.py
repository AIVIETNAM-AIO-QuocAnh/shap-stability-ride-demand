"""Load validated modeling data from the frozen temporal split artifacts."""

import json
from pathlib import Path
from typing import TypedDict

import pandas as pd

from src.load_config import DataConfig, load_data_config
from src.data.build_panel import add_zone_onehot


class ModelData(TypedDict):
    """Feature matrices, targets, and semantic keys for one configured split."""

    X_train: pd.DataFrame
    y_train: pd.Series
    train_keys: pd.DataFrame
    X_evaluation: pd.DataFrame
    y_evaluation: pd.Series
    evaluation_keys: pd.DataFrame


class VariantRecord(TypedDict):
    """Validated feature names for one model variant."""

    weekly_features: list[str]
    description: str


class VariantMap(TypedDict):
    """Validated generated feature-map artifact."""

    base_features: list[str]
    variants: dict[str, VariantRecord]


def _load_variant_map(config: DataConfig) -> VariantMap:
    """Read the generated variant map used by the modeling contract."""
    with config["paths"]["variant_map"].open(encoding="utf-8") as file:
        value: object = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("variant_feature_map.json must contain a JSON object")
    base_features = value.get("base_features")
    variants = value.get("variants")
    if not isinstance(base_features, list) or not all(
        isinstance(feature, str) for feature in base_features
    ):
        raise ValueError("variant_feature_map.json has invalid base_features")
    if not isinstance(variants, dict):
        raise ValueError("variant_feature_map.json has invalid variants")
    validated_variants: dict[str, VariantRecord] = {}
    for variant in ("A", "B", "C"):
        record = variants.get(variant)
        if not isinstance(record, dict):
            raise ValueError(f"variant_feature_map.json is missing variant {variant}")
        weekly_features = record.get("weekly_features")
        description = record.get("description")
        if not isinstance(weekly_features, list) or not all(
            isinstance(feature, str) for feature in weekly_features
        ):
            raise ValueError(f"Variant {variant} has invalid weekly feature names")
        if not isinstance(description, str):
            raise ValueError(f"Variant {variant} has an invalid description")
        validated_variants[variant] = {
            "weekly_features": weekly_features,
            "description": description,
        }
    return {"base_features": base_features, "variants": validated_variants}


def _load_split_frame(
    path: Path, key_columns: tuple[str, str], target_column: str
) -> pd.DataFrame:
    """Read and validate one train or evaluation frame."""
    if not path.exists():
        raise FileNotFoundError(f"Model split file not found: {path}")
    frame = pd.read_csv(path, parse_dates=[key_columns[1]])
    required_columns = set(key_columns) | {target_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required modeling columns: {missing_columns}")
    if frame[list(key_columns)].duplicated().any():
        raise ValueError(f"{path} contains duplicate semantic row keys")
    return frame


def _feature_columns(
    config: DataConfig, variant_map: VariantMap, variant: str
) -> list[str]:
    """Resolve the ordered non-key feature columns for one variant."""
    variant_record = variant_map["variants"].get(variant)
    if variant_record is None:
        raise ValueError(f"Variant {variant} is missing from variant_feature_map.json")
    configured_base = list(config["panel"]["base_features"])
    if variant_map["base_features"] != configured_base:
        raise ValueError(
            f"variant_feature_map.json base_features differ from configuration: "
            f"map={variant_map['base_features']}, config={configured_base}"
        )
    return configured_base + variant_record["weekly_features"]


def load_frozen_zone_ids(config: DataConfig) -> list[int]:
    """Read the frozen zone vocabulary with runtime type validation."""
    with config["paths"]["frozen_zones"].open(encoding="utf-8") as file:
        frozen_zones: object = json.load(file)
    if not isinstance(frozen_zones, dict):
        raise ValueError("Frozen-zone artifact must contain a JSON object")
    zone_ids = frozen_zones.get("zone_ids")
    if not isinstance(zone_ids, list) or not all(isinstance(zone_id, int) for zone_id in zone_ids):
        raise ValueError("Frozen-zone artifact must contain an integer zone_ids list")
    if len(zone_ids) != config["selection"]["top_zones"]:
        raise ValueError(
            f"Frozen-zone artifact contains {len(zone_ids)} zones; "
            f"expected {config['selection']['top_zones']}"
        )
    return zone_ids


def _prepare_frame(
    frame: pd.DataFrame,
    zone_ids: list[int],
    feature_columns: list[str],
    key_columns: tuple[str, str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Encode frozen zones and separate model features, target, and semantic keys."""
    encoded, zone_columns = add_zone_onehot(frame, zone_ids)
    required_features = feature_columns + zone_columns
    missing_features = sorted(set(required_features) - set(encoded.columns))
    if missing_features:
        raise ValueError(f"Split frame is missing model features: {missing_features}")
    keys = frame[list(key_columns)].copy()
    features = encoded[required_features].copy()
    target = frame[target_column].copy()
    if list(features.columns) != required_features:
        raise ValueError("Model feature order does not match the validated feature contract")
    return features, target, keys


def load_data(fold: str, variant: str) -> ModelData:
    """Load one validated temporal split and variant without raw zone-id features."""
    config = load_data_config()
    if fold not in config["splits"]:
        raise ValueError(f"Unknown fold '{fold}'; expected one of {sorted(config['splits'])}")
    if variant not in config["panel"]["variants"]:
        raise ValueError(f"Unknown variant '{variant}'; expected A, B, or C")

    variant_map = _load_variant_map(config)
    feature_columns = _feature_columns(config, variant_map, variant)
    key_columns = config["modeling"]["row_key_columns"]
    target_column = config["modeling"]["target_column"]
    zone_ids = load_frozen_zone_ids(config)

    fold_dir = config["paths"]["folds_dir"] / fold
    evaluation_name = "test.csv" if fold == "final_test" else "val.csv"
    train_frame = _load_split_frame(fold_dir / "train.csv", key_columns, target_column)
    evaluation_frame = _load_split_frame(fold_dir / evaluation_name, key_columns, target_column)
    train_features, train_target, train_keys = _prepare_frame(
        train_frame, zone_ids, feature_columns, key_columns, target_column
    )
    evaluation_features, evaluation_target, evaluation_keys = _prepare_frame(
        evaluation_frame, zone_ids, feature_columns, key_columns, target_column
    )
    if list(train_features.columns) != list(evaluation_features.columns):
        raise ValueError("Train and evaluation feature sets differ")
    return {
        "X_train": train_features,
        "y_train": train_target,
        "train_keys": train_keys,
        "X_evaluation": evaluation_features,
        "y_evaluation": evaluation_target,
        "evaluation_keys": evaluation_keys,
    }
