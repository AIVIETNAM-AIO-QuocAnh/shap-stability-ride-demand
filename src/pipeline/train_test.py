"""Train and evaluate the configured tree models."""

import logging
from typing import TypeAlias, TypedDict

import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.load_config import ModelParameters, load_model_config
from src.pipeline.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.utilities import ModelData


logger = logging.getLogger(__name__)

ModelEstimator: TypeAlias = XGBRegressor | LGBMRegressor


class PredictionMetrics(TypedDict):
    """Prediction metrics stored for one evaluation run."""

    mae: float
    rmse: float
    wape: float


class TrainingResult(TypedDict):
    """Trained estimator output used by artifact writers and SHAP."""

    model: ModelEstimator
    metrics: PredictionMetrics
    y_pred: np.ndarray


def build_model(model_key: str, parameters: ModelParameters) -> ModelEstimator:
    """Build one supported estimator from explicit parameters."""
    if model_key == "xgboost":
        return XGBRegressor(**parameters)
    if model_key == "lightgbm":
        return LGBMRegressor(**parameters)
    raise ValueError(f"Unsupported model '{model_key}'; expected xgboost or lightgbm")


def model_parameters(model_key: str, tuned_parameters: ModelParameters | None) -> ModelParameters:
    """Merge baseline parameters with an explicit frozen parameter set."""
    config = load_model_config()
    if model_key not in config["models"]:
        raise ValueError(f"Unsupported model '{model_key}'; expected xgboost or lightgbm")
    parameters: ModelParameters = {
        "random_state": config["seed"],
        **config["models"][model_key],
    }
    if tuned_parameters is not None:
        parameters.update(tuned_parameters)
    return parameters


def train_and_evaluate(
    data: ModelData,
    model_key: str,
    tuned_parameters: ModelParameters | None,
    fold: str,
) -> TrainingResult:
    """Fit one model and calculate all locked prediction metrics."""
    parameters = model_parameters(model_key, tuned_parameters)
    model = build_model(model_key, parameters)
    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_evaluation"])
    metrics: PredictionMetrics = {
        "mae": mean_absolute_error(data["y_evaluation"], y_pred),
        "rmse": root_mean_squared_error(data["y_evaluation"], y_pred),
        "wape": weighted_absolute_percentage_error(data["y_evaluation"], y_pred),
    }
    logger.info(
        "model_evaluated",
        extra={"model": model_key, "fold": fold, "metrics": metrics},
    )
    return {"model": model, "metrics": metrics, "y_pred": y_pred}
