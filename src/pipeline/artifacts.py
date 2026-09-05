"""Validated, non-overwriting writers for Pipeline artifacts."""

import json
import pickle
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Mapping, TypeAlias

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.load_config import ModelParameters
from src.pipeline.train_test import ModelEstimator, PredictionMetrics


JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _ensure_absent(path: Path) -> None:
    """Fail before replacing an existing artifact."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def write_json(path: Path, value: Mapping[str, JsonValue]) -> None:
    """Write one JSON object without replacing an existing file."""
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)


def load_model_parameters(path: Path) -> ModelParameters:
    """Read and validate scalar frozen model parameters."""
    if not path.is_file():
        raise FileNotFoundError(f"Frozen model parameters not found: {path}")
    with path.open(encoding="utf-8") as stream:
        value: object = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Frozen model parameters must be a JSON object: {path}")
    parameters: ModelParameters = {}
    for name, parameter in value.items():
        if not isinstance(name, str) or not isinstance(parameter, (str, int, float, bool)):
            raise ValueError(f"Frozen model parameters contain an invalid value: {path}")
        parameters[name] = parameter
    return parameters


def package_snapshot() -> dict[str, JsonValue]:
    """Capture runtime versions required to interpret the generated artifacts."""
    packages = ("numpy", "pandas", "lightgbm", "xgboost", "optuna", "shap")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
    }


def save_pickle_model(path: Path, model: ModelEstimator) -> None:
    """Save a trusted model snapshot using the project-approved pickle format."""
    if not isinstance(model, (XGBRegressor, LGBMRegressor)):
        raise TypeError(f"Expected a supported model estimator, got {type(model).__name__}")
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(model, stream)


def load_pickle_model(path: Path) -> ModelEstimator:
    """Load a trusted project model snapshot."""
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with path.open("rb") as stream:
        model: object = pickle.load(stream)
    if not isinstance(model, (XGBRegressor, LGBMRegressor)):
        raise TypeError(f"Model artifact is not a supported estimator: {path}")
    return model


def save_shap_values(path: Path, values: shap.Explanation) -> None:
    """Save one SHAP Explanation without replacing an existing artifact."""
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(values, stream)


def load_prediction_metrics(path: Path) -> PredictionMetrics:
    """Read and validate the three locked prediction metrics."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction metrics not found: {path}")
    with path.open(encoding="utf-8") as stream:
        value: object = json.load(stream)
    if not isinstance(value, dict) or set(value) != {"mae", "rmse", "wape"}:
        raise ValueError(f"Prediction metrics must contain exactly mae, rmse, and wape: {path}")
    metrics: PredictionMetrics = {}
    for name in ("mae", "rmse", "wape"):
        metric = value[name]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not np.isfinite(metric):
            raise ValueError(f"Prediction metric {name} is not a finite number: {path}")
        metrics[name] = float(metric)
    return metrics


def load_weekly_group_importance(path: Path) -> float:
    """Read and validate one weekly-group SHAP importance artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Weekly-group artifact not found: {path}")
    with path.open(encoding="utf-8") as stream:
        value: object = json.load(stream)
    if not isinstance(value, dict) or set(value) != {"weekly_group_importance"}:
        raise ValueError(f"Weekly-group artifact has an invalid schema: {path}")
    importance = value["weekly_group_importance"]
    if isinstance(importance, bool) or not isinstance(importance, (int, float)) or not np.isfinite(importance):
        raise ValueError(f"Weekly-group importance is not a finite number: {path}")
    return float(importance)


def save_prediction_artifacts(
    directory: Path,
    keys: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    metrics: PredictionMetrics,
) -> None:
    """Write keyed predictions and metrics without replacing existing files."""
    if len(keys) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Prediction artifacts have inconsistent row counts")
    expected_keys = ["pu_location_id", "target_datetime"]
    if list(keys.columns) != expected_keys:
        raise ValueError(f"Prediction keys must contain exactly {expected_keys}")
    prediction_frame = keys.copy()
    prediction_frame["y_true"] = np.asarray(y_true)
    prediction_frame["y_pred"] = y_pred
    prediction_path = directory / "y_pred.csv"
    metrics_path = directory / "metrics.json"
    _ensure_absent(prediction_path)
    _ensure_absent(metrics_path)
    directory.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(prediction_path, index=False)
    write_json(metrics_path, {name: float(value) for name, value in metrics.items()})


def write_snapshot(path: Path, snapshot: Mapping[str, JsonValue]) -> None:
    """Write one validated run configuration snapshot."""
    write_json(path, snapshot)
