"""Validate đầy đủ Pipeline artifact matrix trước analysis."""

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from src.load_config import load_data_config, load_model_config
from src.pipeline.artifacts import load_model_parameters, load_prediction_metrics
from src.pipeline.build_pipeline import CORE_FOLDS, MODELS, VARIANTS
from src.pipeline.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.pipeline.run_shap import load_sample_keys
from src.utilities import load_data, load_frozen_zone_ids


FOLDS = CORE_FOLDS
METRIC_RTOL = 1e-7
METRIC_ATOL = 1e-7


class MatrixRow(TypedDict):
    """Một experiment unit dự kiến và trạng thái artifact của nó."""

    stage: str
    model: str
    variant: str
    fold: str
    status: str
    missing_artifacts: str


def _required_artifacts(stage: str) -> tuple[str, ...]:
    """Trả về tên artifact bắt buộc cho một matrix stage."""
    if stage == "hpo":
        return ("best_params.json", "trials.csv", "run_config.json")
    if stage in ("baseline_hpo", "tuned_hpo"):
        return ("model.pkl", "metrics.json", "y_pred.csv", "run_config.json")
    if stage == "core":
        return (
            "model.pkl",
            "metrics.json",
            "y_pred.csv",
            "run_config.json",
            "shap_values.pkl",
            "shap_bar.png",
            "shap_beeswarm.png",
            "shap_importance.csv",
            "shap_weekly_group.json",
            "shap_config.json",
        )
    raise ValueError(f"Matrix stage không được hỗ trợ: {stage}")


def _path_for(stage: str, model: str, variant: str, fold: str) -> Path:
    """Xác định canonical artifact directory của một stage."""
    config = load_model_config()
    if stage == "hpo":
        return config["paths"]["results_hpo"] / model
    if stage in ("baseline_hpo", "tuned_hpo"):
        directory_name = f"{model}_baseline" if stage == "baseline_hpo" else model
        return config["paths"]["results"] / "A" / "hpo" / directory_name
    return config["paths"]["results"] / variant / fold / model


def _matrix_rows() -> list[MatrixRow]:
    """Tạo status row cho hai HPO mode và 30 core unit."""
    rows: list[MatrixRow] = []
    for model in MODELS:
        for stage in ("hpo", "baseline_hpo", "tuned_hpo"):
            directory = _path_for(stage, model, "A", "hpo")
            missing = [name for name in _required_artifacts(stage) if not (directory / name).is_file()]
            rows.append(
                {
                    "stage": stage,
                    "model": model,
                    "variant": "A",
                    "fold": "hpo",
                    "status": "complete" if not missing else "incomplete",
                    "missing_artifacts": ";".join(missing),
                }
            )
    for variant in VARIANTS:
        for fold in FOLDS:
            for model in MODELS:
                directory = _path_for("core", model, variant, fold)
                missing = [
                    name for name in _required_artifacts("core") if not (directory / name).is_file()
                ]
                rows.append(
                    {
                        "stage": "core",
                        "model": model,
                        "variant": variant,
                        "fold": fold,
                        "status": "complete" if not missing else "incomplete",
                        "missing_artifacts": ";".join(missing),
                    }
                )
    return rows


def _validate_trials() -> None:
    """Yêu cầu đúng số complete Optuna trial theo config."""
    config = load_model_config()
    for model in MODELS:
        path = config["paths"]["results_hpo"] / model / "trials.csv"
        if not path.is_file():
            continue
        trials = pd.read_csv(path)
        expected = config["hpo"]["n_trials"]
        if len(trials) != expected:
            raise ValueError(f"{path} chứa {len(trials)} trial; kỳ vọng {expected}")
        if set(trials["state"].astype(str)) != {"COMPLETE"}:
            raise ValueError(f"{path} chứa trial có state không phải COMPLETE")


def _validate_sample_keys() -> None:
    """Yêu cầu semantic sample-key shape deterministic cho mọi core fold."""
    data_config = load_data_config()
    model_config = load_model_config()
    zone_ids = load_frozen_zone_ids(data_config)
    expected_rows = len(zone_ids) * model_config["shap"]["zone_sample_size"]
    for fold in FOLDS:
        sample_keys = load_sample_keys(fold)
        evaluation_keys = load_data(fold, "A")["evaluation_keys"]
        available_keys = evaluation_keys[["pu_location_id", "target_datetime"]]
        joined = sample_keys.merge(
            available_keys,
            on=["pu_location_id", "target_datetime"],
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"{fold} chứa SHAP key không có trong evaluation row")
        if len(sample_keys) != expected_rows:
            raise ValueError(f"{fold} có {len(sample_keys)} SHAP row; kỳ vọng {expected_rows}")
        if sample_keys[["pu_location_id", "target_datetime"]].duplicated().any():
            raise ValueError(f"{fold} chứa SHAP semantic key duplicate")
        counts = sample_keys["pu_location_id"].value_counts()
        if set(counts.index) != set(zone_ids) or not counts.eq(model_config["shap"]["zone_sample_size"]).all():
            raise ValueError(f"{fold} không có đúng số row theo frozen zone như config")


def _validate_prediction(directory: Path, fold: str, variant: str) -> None:
    """Validate prediction row có key và recompute ba metric."""
    data = load_data(fold, variant)
    prediction_path = directory / "y_pred.csv"
    metrics_path = directory / "metrics.json"
    predictions = pd.read_csv(prediction_path, parse_dates=["target_datetime"])
    expected_columns = ["pu_location_id", "target_datetime", "y_true", "y_pred"]
    if list(predictions.columns) != expected_columns:
        raise ValueError(f"{prediction_path} phải chứa đúng {expected_columns}")
    expected_keys = data["evaluation_keys"].reset_index(drop=True)
    actual_keys = predictions[["pu_location_id", "target_datetime"]]
    if not actual_keys.equals(expected_keys):
        raise ValueError(f"Key của {prediction_path} không khớp evaluation row theo config")
    metrics = load_prediction_metrics(metrics_path)
    calculated = {
        "mae": mean_absolute_error(predictions["y_true"], predictions["y_pred"]),
        "rmse": root_mean_squared_error(predictions["y_true"], predictions["y_pred"]),
        "wape": weighted_absolute_percentage_error(
            predictions["y_true"], predictions["y_pred"]
        ),
    }
    for name, expected in calculated.items():
        value = metrics[name]
        if not isinstance(value, (int, float)) or not np.isclose(
            float(value), expected, rtol=METRIC_RTOL, atol=METRIC_ATOL
        ):
            raise ValueError(
                f"{metrics_path} có {name} không nhất quán: stored={value}, "
                f"calculated={expected}, rtol={METRIC_RTOL}, atol={METRIC_ATOL}"
            )


def _validate_core_contract() -> None:
    """Validate mọi core prediction, snapshot và model signature đã freeze."""
    signatures: dict[str, tuple[str, ...]] = {}
    for variant in VARIANTS:
        for fold in FOLDS:
            for model in MODELS:
                directory = _path_for("core", model, variant, fold)
                _validate_prediction(directory, fold, variant)
                with (directory / "run_config.json").open(encoding="utf-8") as stream:
                    snapshot: object = json.load(stream)
                if not isinstance(snapshot, dict) or snapshot.get("stage") != "core":
                    raise ValueError(f"{directory / 'run_config.json'} không phải core snapshot")
                if (
                    snapshot.get("model") != model
                    or snapshot.get("variant") != variant
                    or snapshot.get("fold") != fold
                ):
                    raise ValueError(f"{directory / 'run_config.json'} xác định sai run")
                feature_columns = snapshot.get("feature_columns")
                if not isinstance(feature_columns, list) or "pu_location_id" in feature_columns:
                    raise ValueError(f"{directory / 'run_config.json'} để lộ raw pu_location_id")
                parameters = snapshot.get("parameters")
                if not isinstance(parameters, dict):
                    raise ValueError(f"{directory / 'run_config.json'} không có parameter snapshot")
                signature = tuple(f"{key}={parameters[key]}" for key in sorted(parameters))
                signatures.setdefault(model, signature)
                if signatures[model] != signature:
                    raise ValueError(f"Frozen parameter khác nhau giữa các core run của {model}")


def _validate_hpo_contract() -> None:
    """Validate prediction có key cho cả hai HPO evaluation mode."""
    for model in MODELS:
        load_model_parameters(_path_for("hpo", model, "A", "hpo") / "best_params.json")
        for stage in ("baseline_hpo", "tuned_hpo"):
            _validate_prediction(_path_for(stage, model, "A", "hpo"), "hpo", "A")


def validate_matrix() -> pd.DataFrame:
    """Validate artifact cần có, semantics, metric và frozen parameter."""
    rows = pd.DataFrame(_matrix_rows())
    incomplete = rows[rows["status"] != "complete"]
    if not incomplete.empty:
        raise FileNotFoundError(
            "Pipeline artifact matrix chưa đầy đủ: "
            + "; ".join(
                f"{row.stage}:{row.variant}/{row.fold}/{row.model} missing {row.missing_artifacts}"
                for row in incomplete.itertuples()
            )
        )
    _validate_trials()
    _validate_sample_keys()
    _validate_hpo_contract()
    _validate_core_contract()
    return rows


def write_run_matrix() -> Path:
    """Validate và ghi canonical experiment matrix một lần."""
    config = load_model_config()
    path = config["paths"]["results"] / "run_matrix.csv"
    if path.exists():
        raise FileExistsError(f"Từ chối ghi đè matrix đã tồn tại: {path}")
    matrix = validate_matrix()
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, index=False)
    return path
