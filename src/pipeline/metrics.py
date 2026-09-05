"""Pure prediction metric functions used by the model pipeline."""

from typing import TypeAlias

import numpy as np
import pandas as pd


NumericVector: TypeAlias = pd.Series | np.ndarray


def _validated_vectors(
    y_true: NumericVector, y_pred: NumericVector
) -> tuple[np.ndarray, np.ndarray]:
    """Convert two numeric vectors to validated floating-point arrays."""
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional vectors")
    if len(true_values) != len(predicted_values):
        raise ValueError(
            f"Metric inputs have different lengths: y_true={len(true_values)}, "
            f"y_pred={len(predicted_values)}"
        )
    if not np.isfinite(true_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Metric inputs must contain only finite values")
    return true_values, predicted_values


def mean_absolute_error(y_true: NumericVector, y_pred: NumericVector) -> float:
    """Return mean absolute error for an evaluation vector."""
    true_values, predicted_values = _validated_vectors(y_true, y_pred)
    return float(np.abs(true_values - predicted_values).mean())


def root_mean_squared_error(y_true: NumericVector, y_pred: NumericVector) -> float:
    """Return root mean squared error for an evaluation vector."""
    true_values, predicted_values = _validated_vectors(y_true, y_pred)
    return float(np.sqrt(np.square(true_values - predicted_values).mean()))


def weighted_absolute_percentage_error(y_true: NumericVector, y_pred: NumericVector) -> float:
    """Return WAPE as a percentage for an evaluation vector."""
    true_values, predicted_values = _validated_vectors(y_true, y_pred)
    denominator = float(np.abs(true_values).sum())
    if denominator == 0:
        raise ValueError("WAPE is undefined because the absolute target sum is zero")
    return float(np.abs(true_values - predicted_values).sum() / denominator * 100.0)
