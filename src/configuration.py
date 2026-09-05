"""Load and validate the repository's data and model configuration."""

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import TypedDict

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


class DataPaths(TypedDict):
    raw_dir: Path
    monthly_aggregates_dir: Path
    frozen_zones: Path
    feature_table: Path
    variant_map: Path
    lag_examples: Path
    correlation_by_zone: Path
    correlation_summary: Path
    data_dictionary: Path
    folds_dir: Path


class SourceConfig(TypedDict):
    base_url: str
    filename_pattern: str
    request_datetime_column: str
    pickup_zone_column: str
    normalized_zone_column: str
    required_columns: tuple[str, str]


class SelectionConfig(TypedDict):
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    top_zones: int
    period_label: str
    selection_rule: str


class VariantConfig(TypedDict):
    weekly_features: tuple[str, ...]
    description: str


class PanelConfig(TypedDict):
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    calendar_features: tuple[str, ...]
    short_lags_hours: tuple[int, ...]
    weekly_lags_hours: tuple[int, ...]
    all_lags_hours: tuple[int, ...]
    base_features: tuple[str, ...]
    median_feature: str
    variants: dict[str, VariantConfig]


class CorrelationConfig(TypedDict):
    start: pd.Timestamp
    end_exclusive: pd.Timestamp


class SplitConfig(TypedDict):
    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end_exclusive: pd.Timestamp
    eval_kind: str


class DataConfig(TypedDict):
    dataset_name: str
    year: int
    months: tuple[str, ...]
    source: SourceConfig
    paths: DataPaths
    selection: SelectionConfig
    panel: PanelConfig
    correlation: CorrelationConfig
    splits: dict[str, SplitConfig]


ModelScalar = str | int | float | bool
ModelParameters = dict[str, ModelScalar]


class SearchRange(TypedDict, total=False):
    low: float
    high: float
    log: bool


SearchValue = list[ModelScalar] | SearchRange
SearchSpace = dict[str, SearchValue]


class ModelPaths(TypedDict):
    results: Path
    results_hpo: Path
    results_stats: Path


class HpoConfig(TypedDict):
    n_trials: int
    search_space: dict[str, SearchSpace]


class ShapConfig(TypedDict):
    zone_sample_size: int


class ModelConfig(TypedDict):
    seed: int
    paths: ModelPaths
    models: dict[str, ModelParameters]
    hpo: HpoConfig
    shap: ShapConfig


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a mapping with string keys")
    return value


def _required(mapping: Mapping[str, object], key: str, context: str) -> object:
    if key not in mapping:
        raise ValueError(f"{context} is missing required key '{key}'")
    return mapping[key]


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def _timestamp(value: object, context: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(_string(value, context))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a valid timestamp") from exc
    if timestamp.tz is not None:
        raise ValueError(f"{context} must be timezone-naive")
    return timestamp


def _relative_path(value: object, context: str) -> Path:
    path = Path(_string(value, context))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must be a repository-relative path")
    return PROJECT_ROOT / path


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")
    return tuple(_string(item, f"{context}[{index}]") for index, item in enumerate(value))


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of integers")
    return tuple(_integer(item, f"{context}[{index}]") for index, item in enumerate(value))


def _load_yaml(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        raw: object = yaml.safe_load(stream)
    return _mapping(raw, str(path))


def _parse_data_config(raw: Mapping[str, object]) -> DataConfig:
    dataset = _mapping(_required(raw, "dataset", "data config"), "data config.dataset")
    source_raw = _mapping(_required(raw, "source", "data config"), "data config.source")
    paths_raw = _mapping(_required(raw, "paths", "data config"), "data config.paths")
    selection_raw = _mapping(_required(raw, "selection", "data config"), "data config.selection")
    panel_raw = _mapping(_required(raw, "panel", "data config"), "data config.panel")
    correlation_raw = _mapping(_required(raw, "correlation", "data config"), "data config.correlation")
    splits_raw = _mapping(_required(raw, "splits", "data config"), "data config.splits")

    year = _integer(_required(dataset, "year", "data config.dataset"), "data config.dataset.year")
    months = tuple(f"{year}-{month:02d}" for month in range(1, 13))
    request_column = _string(
        _required(source_raw, "request_datetime_column", "data config.source"),
        "data config.source.request_datetime_column",
    )
    pickup_column = _string(
        _required(source_raw, "pickup_zone_column", "data config.source"),
        "data config.source.pickup_zone_column",
    )
    normalized_column = _string(
        _required(source_raw, "normalized_zone_column", "data config.source"),
        "data config.source.normalized_zone_column",
    )

    data_paths: DataPaths = {
        name: _relative_path(
            _required(paths_raw, name, "data config.paths"), f"data config.paths.{name}"
        )
        for name in DataPaths.__annotations__
    }

    selection_start = _timestamp(
        _required(selection_raw, "start", "data config.selection"),
        "data config.selection.start",
    )
    selection_end = _timestamp(
        _required(selection_raw, "end_exclusive", "data config.selection"),
        "data config.selection.end_exclusive",
    )
    top_zones = _integer(
        _required(selection_raw, "top_zones", "data config.selection"),
        "data config.selection.top_zones",
    )
    if top_zones <= 0 or selection_end <= selection_start:
        raise ValueError("data config.selection must contain a positive top_zones and ordered dates")
    period_label = f"{selection_start:%Y-%m-%d} to {(selection_end - pd.Timedelta(days=1)):%Y-%m-%d}"
    selection_rule = (
        f"top {top_zones} {normalized_column} by total trip_count where hour is in "
        f"[{selection_start:%Y-%m-%d %H:%M}, {selection_end:%Y-%m-%d %H:%M}); "
        f"ties use {normalized_column} ascending"
    )

    panel_start = _timestamp(
        _required(panel_raw, "start", "data config.panel"), "data config.panel.start"
    )
    panel_end = _timestamp(
        _required(panel_raw, "end_exclusive", "data config.panel"),
        "data config.panel.end_exclusive",
    )
    calendar_features = _string_tuple(
        _required(panel_raw, "calendar_features", "data config.panel"),
        "data config.panel.calendar_features",
    )
    short_lags = _integer_tuple(
        _required(panel_raw, "short_lags_hours", "data config.panel"),
        "data config.panel.short_lags_hours",
    )
    weekly_lags = _integer_tuple(
        _required(panel_raw, "weekly_lags_hours", "data config.panel"),
        "data config.panel.weekly_lags_hours",
    )
    median_feature = _string(
        _required(panel_raw, "median_feature", "data config.panel"),
        "data config.panel.median_feature",
    )
    if not short_lags or not weekly_lags or any(lag <= 0 for lag in short_lags + weekly_lags):
        raise ValueError("data config.panel lag lists must contain positive integers")
    if panel_end <= panel_start:
        raise ValueError("data config.panel dates must be ordered")

    variants_raw = _mapping(
        _required(panel_raw, "variants", "data config.panel"), "data config.panel.variants"
    )
    variants: dict[str, VariantConfig] = {}
    for variant_name in ("A", "B", "C"):
        variant_raw = _mapping(
            _required(variants_raw, variant_name, "data config.panel.variants"),
            f"data config.panel.variants.{variant_name}",
        )
        variants[variant_name] = {
            "weekly_features": _string_tuple(
                _required(
                    variant_raw,
                    "weekly_features",
                    f"data config.panel.variants.{variant_name}",
                ),
                f"data config.panel.variants.{variant_name}.weekly_features",
            ),
            "description": _string(
                _required(variant_raw, "description", f"data config.panel.variants.{variant_name}"),
                f"data config.panel.variants.{variant_name}.description",
            ),
        }

    all_lags = short_lags + weekly_lags
    base_features = calendar_features + tuple(f"lag_{lag}" for lag in short_lags)
    valid_weekly_features = {f"lag_{lag}" for lag in weekly_lags} | {median_feature}
    for variant_name, variant in variants.items():
        if not set(variant["weekly_features"]).issubset(valid_weekly_features):
            raise ValueError(
                f"data config.panel.variants.{variant_name}.weekly_features contains an unknown feature"
            )

    correlation_start = _timestamp(
        _required(correlation_raw, "start", "data config.correlation"),
        "data config.correlation.start",
    )
    correlation_end = _timestamp(
        _required(correlation_raw, "end_exclusive", "data config.correlation"),
        "data config.correlation.end_exclusive",
    )
    if correlation_end <= correlation_start:
        raise ValueError("data config.correlation dates must be ordered")

    expected_split_names = ("hpo", "fold1", "fold2", "fold3", "fold4", "final_test")
    if set(splits_raw) != set(expected_split_names):
        raise ValueError(f"data config.splits must contain exactly {list(expected_split_names)}")
    splits: dict[str, SplitConfig] = {}
    for split_name in expected_split_names:
        split_raw = _mapping(splits_raw[split_name], f"data config.splits.{split_name}")
        split: SplitConfig = {
            key: _timestamp(
                _required(split_raw, key, f"data config.splits.{split_name}"),
                f"data config.splits.{split_name}.{key}",
            )
            for key in (
                "train_start",
                "train_end_exclusive",
                "eval_start",
                "eval_end_exclusive",
            )
        }
        split["eval_kind"] = _string(
            _required(split_raw, "eval_kind", f"data config.splits.{split_name}"),
            f"data config.splits.{split_name}.eval_kind",
        )
        if not (
            split["train_start"] < split["train_end_exclusive"]
            and split["train_end_exclusive"] == split["eval_start"]
            and split["eval_start"] < split["eval_end_exclusive"]
        ):
            raise ValueError(f"data config.splits.{split_name} has invalid or non-adjacent windows")
        splits[split_name] = split

    warmup_start = panel_start + pd.Timedelta(hours=max(all_lags))
    if splits["hpo"]["train_start"] != warmup_start:
        raise ValueError("data config.splits.hpo.train_start must equal the panel warm-up boundary")
    if splits["final_test"]["eval_end_exclusive"] != panel_end:
        raise ValueError("data config.splits.final_test.eval_end_exclusive must equal panel.end_exclusive")

    return {
        "dataset_name": _string(_required(dataset, "name", "data config.dataset"), "data config.dataset.name"),
        "year": year,
        "months": months,
        "source": {
            "base_url": _string(_required(source_raw, "base_url", "data config.source"), "data config.source.base_url"),
            "filename_pattern": _string(
                _required(source_raw, "filename_pattern", "data config.source"),
                "data config.source.filename_pattern",
            ),
            "request_datetime_column": request_column,
            "pickup_zone_column": pickup_column,
            "normalized_zone_column": normalized_column,
            "required_columns": (request_column, pickup_column),
        },
        "paths": data_paths,
        "selection": {
            "start": selection_start,
            "end_exclusive": selection_end,
            "top_zones": top_zones,
            "period_label": period_label,
            "selection_rule": selection_rule,
        },
        "panel": {
            "start": panel_start,
            "end_exclusive": panel_end,
            "calendar_features": calendar_features,
            "short_lags_hours": short_lags,
            "weekly_lags_hours": weekly_lags,
            "all_lags_hours": all_lags,
            "base_features": base_features,
            "median_feature": median_feature,
            "variants": variants,
        },
        "correlation": {"start": correlation_start, "end_exclusive": correlation_end},
        "splits": splits,
    }


def _model_scalar(value: object, context: str) -> ModelScalar:
    if isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"{context} must be a scalar model parameter")


def _parse_search_space(value: object, context: str) -> SearchSpace:
    raw = _mapping(value, context)
    parsed: SearchSpace = {}
    for name, spec in raw.items():
        if isinstance(spec, list):
            if not spec:
                raise ValueError(f"{context}.{name} must not be empty")
            parsed[name] = [_model_scalar(item, f"{context}.{name}") for item in spec]
            continue
        range_raw = _mapping(spec, f"{context}.{name}")
        low = _number(_required(range_raw, "low", f"{context}.{name}"), f"{context}.{name}.low")
        high = _number(_required(range_raw, "high", f"{context}.{name}"), f"{context}.{name}.high")
        if low >= high:
            raise ValueError(f"{context}.{name} requires low < high")
        range_config: SearchRange = {"low": low, "high": high}
        if "log" in range_raw:
            log = range_raw["log"]
            if not isinstance(log, bool):
                raise ValueError(f"{context}.{name}.log must be boolean")
            range_config["log"] = log
        parsed[name] = range_config
    return parsed


def _parse_model_config(raw: Mapping[str, object]) -> ModelConfig:
    paths_raw = _mapping(_required(raw, "paths", "model config"), "model config.paths")
    models_raw = _mapping(_required(raw, "models", "model config"), "model config.models")
    hpo_raw = _mapping(_required(raw, "hpo", "model config"), "model config.hpo")
    shap_raw = _mapping(_required(raw, "shap", "model config"), "model config.shap")

    model_paths: ModelPaths = {
        name: _relative_path(
            _required(paths_raw, name, "model config.paths"), f"model config.paths.{name}"
        )
        for name in ModelPaths.__annotations__
    }
    models: dict[str, ModelParameters] = {}
    for model_name in ("lightgbm", "xgboost"):
        model_raw = _mapping(
            _required(models_raw, model_name, "model config.models"),
            f"model config.models.{model_name}",
        )
        models[model_name] = {
            name: _model_scalar(value, f"model config.models.{model_name}.{name}")
            for name, value in model_raw.items()
        }

    search_raw = _mapping(
        _required(hpo_raw, "search_space", "model config.hpo"),
        "model config.hpo.search_space",
    )
    search_space = {
        model_name: _parse_search_space(
            _required(search_raw, model_name, "model config.hpo.search_space"),
            f"model config.hpo.search_space.{model_name}",
        )
        for model_name in ("lightgbm", "xgboost")
    }
    n_trials = _integer(_required(hpo_raw, "n_trials", "model config.hpo"), "model config.hpo.n_trials")
    sample_size = _integer(
        _required(shap_raw, "zone_sample_size", "model config.shap"),
        "model config.shap.zone_sample_size",
    )
    if n_trials <= 0 or sample_size <= 0:
        raise ValueError("model config hpo.n_trials and shap.zone_sample_size must be positive")

    return {
        "seed": _integer(_required(raw, "seed", "model config"), "model config.seed"),
        "paths": model_paths,
        "models": models,
        "hpo": {"n_trials": n_trials, "search_space": search_space},
        "shap": {"zone_sample_size": sample_size},
    }


@cache
def load_data_config() -> DataConfig:
    """Load and validate the shared data protocol configuration once per process."""
    return _parse_data_config(_load_yaml(CONFIG_DIR / "data.yaml"))


@cache
def load_model_config() -> ModelConfig:
    """Load and validate the shared model protocol configuration once per process."""
    return _parse_model_config(_load_yaml(CONFIG_DIR / "model.yaml"))
