"""Đọc modeling data đã validate từ các temporal split artifact đã freeze."""

import json
from pathlib import Path
from typing import TypedDict

import pandas as pd

from src.load_config import DataConfig, load_data_config
from src.data.build_panel import add_zone_onehot


class ModelData(TypedDict):
    """Feature matrix, target và semantic key của một split theo config."""

    X_train: pd.DataFrame
    y_train: pd.Series
    train_keys: pd.DataFrame
    X_evaluation: pd.DataFrame
    y_evaluation: pd.Series
    evaluation_keys: pd.DataFrame


class VariantRecord(TypedDict):
    """Tên feature đã validate của một model variant."""

    weekly_features: list[str]
    description: str


class VariantMap(TypedDict):
    """Feature-map artifact đã sinh và validate."""

    base_features: list[str]
    variants: dict[str, VariantRecord]


def _load_variant_map(config: DataConfig) -> VariantMap:
    """Đọc variant map đã sinh, được dùng trong modeling contract."""
    with config["paths"]["variant_map"].open(encoding="utf-8") as file:
        value: object = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("variant_feature_map.json phải chứa một JSON object")
    base_features = value.get("base_features")
    variants = value.get("variants")
    if not isinstance(base_features, list) or not all(
        isinstance(feature, str) for feature in base_features
    ):
        raise ValueError("variant_feature_map.json có base_features không hợp lệ")
    if not isinstance(variants, dict):
        raise ValueError("variant_feature_map.json có variants không hợp lệ")
    validated_variants: dict[str, VariantRecord] = {}
    for variant in ("A", "B", "C"):
        record = variants.get(variant)
        if not isinstance(record, dict):
            raise ValueError(f"variant_feature_map.json thiếu variant {variant}")
        weekly_features = record.get("weekly_features")
        description = record.get("description")
        if not isinstance(weekly_features, list) or not all(
            isinstance(feature, str) for feature in weekly_features
        ):
            raise ValueError(f"Variant {variant} có tên weekly feature không hợp lệ")
        if not isinstance(description, str):
            raise ValueError(f"Variant {variant} có description không hợp lệ")
        validated_variants[variant] = {
            "weekly_features": weekly_features,
            "description": description,
        }
    return {"base_features": base_features, "variants": validated_variants}


def _load_split_frame(
    path: Path, key_columns: tuple[str, str], target_column: str
) -> pd.DataFrame:
    """Đọc và validate một train hoặc evaluation frame."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model split file: {path}")
    frame = pd.read_csv(path, parse_dates=[key_columns[1]])
    required_columns = set(key_columns) | {target_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} thiếu modeling column bắt buộc: {missing_columns}")
    if frame[list(key_columns)].duplicated().any():
        raise ValueError(f"{path} chứa semantic row key trùng lặp")
    return frame


def _feature_columns(
    config: DataConfig, variant_map: VariantMap, variant: str
) -> list[str]:
    """Xác định các feature column không phải key theo đúng thứ tự của một variant."""
    variant_record = variant_map["variants"].get(variant)
    if variant_record is None:
        raise ValueError(f"variant_feature_map.json thiếu Variant {variant}")
    configured_base = list(config["panel"]["base_features"])
    if variant_map["base_features"] != configured_base:
        raise ValueError(
            f"base_features trong variant_feature_map.json khác configuration: "
            f"map={variant_map['base_features']}, config={configured_base}"
        )
    return configured_base + variant_record["weekly_features"]


def load_frozen_zone_ids(config: DataConfig) -> list[int]:
    """Đọc frozen zone vocabulary và validate type ở runtime."""
    with config["paths"]["frozen_zones"].open(encoding="utf-8") as file:
        frozen_zones: object = json.load(file)
    if not isinstance(frozen_zones, dict):
        raise ValueError("Frozen-zone artifact phải chứa một JSON object")
    zone_ids = frozen_zones.get("zone_ids")
    if not isinstance(zone_ids, list) or not all(isinstance(zone_id, int) for zone_id in zone_ids):
        raise ValueError("Frozen-zone artifact phải chứa list zone_ids dạng integer")
    if len(zone_ids) != config["selection"]["top_zones"]:
        raise ValueError(
            f"Frozen-zone artifact chứa {len(zone_ids)} zone; "
            f"kỳ vọng {config['selection']['top_zones']}"
        )
    return zone_ids


def _prepare_frame(
    frame: pd.DataFrame,
    zone_ids: list[int],
    feature_columns: list[str],
    key_columns: tuple[str, str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Encode frozen zone và tách model feature, target, semantic key."""
    encoded, zone_columns = add_zone_onehot(frame, zone_ids)
    required_features = feature_columns + zone_columns
    missing_features = sorted(set(required_features) - set(encoded.columns))
    if missing_features:
        raise ValueError(f"Split frame thiếu model feature: {missing_features}")
    keys = frame[list(key_columns)].copy()
    features = encoded[required_features].copy()
    target = frame[target_column].copy()
    if list(features.columns) != required_features:
        raise ValueError("Thứ tự model feature không khớp feature contract đã validate")
    return features, target, keys


def load_data(fold: str, variant: str) -> ModelData:
    """Đọc một temporal split và variant đã validate, không đưa raw zone-id vào feature."""
    config = load_data_config()
    if fold not in config["splits"]:
        raise ValueError(f"Fold không xác định '{fold}'; kỳ vọng một trong {sorted(config['splits'])}")
    if variant not in config["panel"]["variants"]:
        raise ValueError(f"Variant không xác định '{variant}'; kỳ vọng A, B hoặc C")

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
        raise ValueError("Tập feature train và evaluation khác nhau")
    return {
        "X_train": train_features,
        "y_train": train_target,
        "train_keys": train_keys,
        "X_evaluation": evaluation_features,
        "y_evaluation": evaluation_target,
        "evaluation_keys": evaluation_keys,
    }
