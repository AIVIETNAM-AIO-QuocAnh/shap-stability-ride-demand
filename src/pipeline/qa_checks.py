"""Validate the complete Pipeline artifact matrix before analysis."""

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
    """One expected experiment unit and its artifact status."""

    stage: str
    model: str
    variant: str
    fold: str
    status: str
    missing_artifacts: str


def _required_artifacts(stage: str) -> tuple[str, ...]:
    """Return the artifact names required for one matrix stage."""
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
    raise ValueError(f"Unsupported matrix stage: {stage}")


def _path_for(stage: str, model: str, variant: str, fold: str) -> Path:
    """Resolve one stage's canonical artifact directory."""
    config = load_model_config()
    if stage == "hpo":
        return config["paths"]["results_hpo"] / model
    if stage in ("baseline_hpo", "tuned_hpo"):
        directory_name = f"{model}_baseline" if stage == "baseline_hpo" else model
        return config["paths"]["results"] / "A" / "hpo" / directory_name
    return config["paths"]["results"] / variant / fold / model


def _matrix_rows() -> list[MatrixRow]:
    """Build status rows for the two HPO modes and 30 core units."""
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
    """Require exactly the configured number of complete Optuna trials."""
    config = load_model_config()
    for model in MODELS:
        path = config["paths"]["results_hpo"] / model / "trials.csv"
        if not path.is_file():
            continue
        trials = pd.read_csv(path)
        expected = config["hpo"]["n_trials"]
        if len(trials) != expected:
            raise ValueError(f"{path} contains {len(trials)} trials; expected {expected}")
        if set(trials["state"].astype(str)) != {"COMPLETE"}:
            raise ValueError(f"{path} contains a non-COMPLETE trial state")


def _validate_sample_keys() -> None:
    """Require deterministic semantic sample-key shape for every core fold."""
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
            raise ValueError(f"{fold} contains SHAP keys absent from the evaluation rows")
        if len(sample_keys) != expected_rows:
            raise ValueError(f"{fold} has {len(sample_keys)} SHAP rows; expected {expected_rows}")
        if sample_keys[["pu_location_id", "target_datetime"]].duplicated().any():
            raise ValueError(f"{fold} contains duplicate SHAP semantic keys")
        counts = sample_keys["pu_location_id"].value_counts()
        if set(counts.index) != set(zone_ids) or not counts.eq(model_config["shap"]["zone_sample_size"]).all():
            raise ValueError(f"{fold} does not contain the configured rows per frozen zone")


def _validate_prediction(directory: Path, fold: str, variant: str) -> None:
    """Validate keyed prediction rows and recompute their three metrics."""
    data = load_data(fold, variant)
    prediction_path = directory / "y_pred.csv"
    metrics_path = directory / "metrics.json"
    predictions = pd.read_csv(prediction_path, parse_dates=["target_datetime"])
    expected_columns = ["pu_location_id", "target_datetime", "y_true", "y_pred"]
    if list(predictions.columns) != expected_columns:
        raise ValueError(f"{prediction_path} must contain exactly {expected_columns}")
    expected_keys = data["evaluation_keys"].reset_index(drop=True)
    actual_keys = predictions[["pu_location_id", "target_datetime"]]
    if not actual_keys.equals(expected_keys):
        raise ValueError(f"{prediction_path} keys do not match the configured evaluation rows")
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
                f"{metrics_path} has inconsistent {name}: stored={value}, "
                f"calculated={expected}, rtol={METRIC_RTOL}, atol={METRIC_ATOL}"
            )


def _validate_core_contract() -> None:
    """Validate all core predictions, snapshots, and frozen model signatures."""
    signatures: dict[str, tuple[str, ...]] = {}
    for variant in VARIANTS:
        for fold in FOLDS:
            for model in MODELS:
                directory = _path_for("core", model, variant, fold)
                _validate_prediction(directory, fold, variant)
                with (directory / "run_config.json").open(encoding="utf-8") as stream:
                    snapshot: object = json.load(stream)
                if not isinstance(snapshot, dict) or snapshot.get("stage") != "core":
                    raise ValueError(f"{directory / 'run_config.json'} is not a core snapshot")
                if (
                    snapshot.get("model") != model
                    or snapshot.get("variant") != variant
                    or snapshot.get("fold") != fold
                ):
                    raise ValueError(f"{directory / 'run_config.json'} identifies the wrong run")
                feature_columns = snapshot.get("feature_columns")
                if not isinstance(feature_columns, list) or "pu_location_id" in feature_columns:
                    raise ValueError(f"{directory / 'run_config.json'} exposes raw pu_location_id")
                parameters = snapshot.get("parameters")
                if not isinstance(parameters, dict):
                    raise ValueError(f"{directory / 'run_config.json'} has no parameter snapshot")
                signature = tuple(f"{key}={parameters[key]}" for key in sorted(parameters))
                signatures.setdefault(model, signature)
                if signatures[model] != signature:
                    raise ValueError(f"Frozen parameters differ across core runs for {model}")


def _validate_hpo_contract() -> None:
    """Validate keyed predictions for both HPO evaluation modes."""
    for model in MODELS:
        load_model_parameters(_path_for("hpo", model, "A", "hpo") / "best_params.json")
        for stage in ("baseline_hpo", "tuned_hpo"):
            _validate_prediction(_path_for(stage, model, "A", "hpo"), "hpo", "A")


def validate_matrix() -> pd.DataFrame:
    """Validate expected artifact presence, semantics, metrics, and frozen parameters."""
    rows = pd.DataFrame(_matrix_rows())
    incomplete = rows[rows["status"] != "complete"]
    if not incomplete.empty:
        raise FileNotFoundError(
            "Incomplete Pipeline artifact matrix: "
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
    """Validate and write the canonical experiment matrix once."""
    config = load_model_config()
    path = config["paths"]["results"] / "run_matrix.csv"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing matrix: {path}")
    matrix = validate_matrix()
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, index=False)
    return path
