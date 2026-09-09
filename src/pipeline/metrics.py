"""Các pure prediction metric function được model pipeline sử dụng."""

from typing import TypeAlias

import numpy as np
import pandas as pd


NumericVector: TypeAlias = pd.Series | np.ndarray


def _validated_vectors(
    y_true: NumericVector, y_pred: NumericVector
) -> tuple[np.ndarray, np.ndarray]:
    """Chuyển hai numeric vector thành floating-point array đã validate."""
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Metric input phải là vector một chiều")
    if len(true_values) != len(predicted_values):
        raise ValueError(
            f"Metric input có độ dài khác nhau: y_true={len(true_values)}, "
            f"y_pred={len(predicted_values)}"
        )
    if not np.isfinite(true_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Metric input chỉ được chứa finite value")
    return true_values, predicted_values


def mean_absolute_error(y_true: NumericVector, y_pred: NumericVector) -> float:
    """Trả về mean absolute error của evaluation vector."""
    true_values, predicted_values = _validated_vectors(y_true, y_pred)
    return float(np.abs(true_values - predicted_values).mean())


def root_mean_squared_error(y_true: NumericVector, y_pred: NumericVector) -> float:
    """Trả về root mean squared error của evaluation vector."""
    true_values, predicted_values = _validated_vectors(y_true, y_pred)
    return float(np.sqrt(np.square(true_values - predicted_values).mean()))


def weighted_absolute_percentage_error(y_true: NumericVector, y_pred: NumericVector) -> float:
    """Trả về WAPE dạng phần trăm của evaluation vector."""
    true_values, predicted_values = _validated_vectors(y_true, y_pred)
    denominator = float(np.abs(true_values).sum())
    if denominator == 0:
        raise ValueError("WAPE không xác định vì tổng target tuyệt đối bằng zero")
    return float(np.abs(true_values - predicted_values).sum() / denominator * 100.0)
