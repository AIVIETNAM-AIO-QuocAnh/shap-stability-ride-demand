"""Writer cho Pipeline artifact đã validate, không ghi đè file hiện có."""

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
    """Báo lỗi trước khi thay thế artifact hiện có."""
    if path.exists():
        raise FileExistsError(f"Từ chối ghi đè artifact đã tồn tại: {path}")


def write_json(path: Path, value: Mapping[str, JsonValue]) -> None:
    """Ghi một JSON object mà không thay thế file hiện có."""
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)


def load_model_parameters(path: Path) -> ModelParameters:
    """Đọc và validate scalar parameter của model đã freeze."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy parameter của model đã freeze: {path}")
    with path.open(encoding="utf-8") as stream:
        value: object = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Parameter của model đã freeze phải là JSON object: {path}")
    parameters: ModelParameters = {}
    for name, parameter in value.items():
        if not isinstance(name, str) or not isinstance(parameter, (str, int, float, bool)):
            raise ValueError(f"Parameter của model đã freeze chứa giá trị không hợp lệ: {path}")
        parameters[name] = parameter
    return parameters


def package_snapshot() -> dict[str, JsonValue]:
    """Ghi nhận runtime version cần để diễn giải artifact đã sinh."""
    packages = ("numpy", "pandas", "lightgbm", "xgboost", "optuna", "shap")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
    }


def save_pickle_model(path: Path, model: ModelEstimator) -> None:
    """Lưu model snapshot đáng tin cậy theo pickle format đã được project duyệt."""
    if not isinstance(model, (XGBRegressor, LGBMRegressor)):
        raise TypeError(f"Kỳ vọng supported model estimator, nhưng nhận {type(model).__name__}")
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(model, stream)


def load_pickle_model(path: Path) -> ModelEstimator:
    """Đọc model snapshot đáng tin cậy của project."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model artifact: {path}")
    with path.open("rb") as stream:
        model: object = pickle.load(stream)
    if not isinstance(model, (XGBRegressor, LGBMRegressor)):
        raise TypeError(f"Model artifact không phải supported estimator: {path}")
    return model


def save_shap_values(path: Path, values: shap.Explanation) -> None:
    """Lưu một SHAP Explanation mà không thay thế artifact hiện có."""
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(values, stream)


def load_prediction_metrics(path: Path) -> PredictionMetrics:
    """Đọc và validate ba prediction metric đã khoá."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy prediction metric: {path}")
    with path.open(encoding="utf-8") as stream:
        value: object = json.load(stream)
    if not isinstance(value, dict) or set(value) != {"mae", "rmse", "wape"}:
        raise ValueError(f"Prediction metric phải chứa đúng mae, rmse và wape: {path}")
    metrics: PredictionMetrics = {}
    for name in ("mae", "rmse", "wape"):
        metric = value[name]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not np.isfinite(metric):
            raise ValueError(f"Prediction metric {name} không phải finite number: {path}")
        metrics[name] = float(metric)
    return metrics


def load_weekly_group_importance(path: Path) -> float:
    """Đọc và validate một weekly-group SHAP importance artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy weekly-group artifact: {path}")
    with path.open(encoding="utf-8") as stream:
        value: object = json.load(stream)
    if not isinstance(value, dict) or set(value) != {"weekly_group_importance"}:
        raise ValueError(f"Weekly-group artifact có schema không hợp lệ: {path}")
    importance = value["weekly_group_importance"]
    if isinstance(importance, bool) or not isinstance(importance, (int, float)) or not np.isfinite(importance):
        raise ValueError(f"Weekly-group importance không phải finite number: {path}")
    return float(importance)


def save_prediction_artifacts(
    directory: Path,
    keys: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    metrics: PredictionMetrics,
) -> None:
    """Ghi prediction có key và metric mà không thay thế file hiện có."""
    if len(keys) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Prediction artifact có số row không nhất quán")
    expected_keys = ["pu_location_id", "target_datetime"]
    if list(keys.columns) != expected_keys:
        raise ValueError(f"Prediction key phải chứa đúng {expected_keys}")
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
    """Ghi một run configuration snapshot đã validate."""
    write_json(path, snapshot)
