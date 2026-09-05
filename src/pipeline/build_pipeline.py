"""Pure orchestration functions for the configured Pipeline stages."""

import logging
from itertools import product
from pathlib import Path
from typing import Final

from tqdm import tqdm

from src.load_config import load_data_config, load_model_config
from src.pipeline.artifacts import (
    JsonValue,
    load_model_parameters,
    package_snapshot,
    save_pickle_model,
    save_prediction_artifacts,
    write_snapshot,
)
from src.pipeline.run_hpo import run_hpo
from src.pipeline.run_shap import save_sample_keys, run_shap
from src.pipeline.train_test import TrainingResult, model_parameters, train_and_evaluate
from src.utilities import ModelData, load_data


logger = logging.getLogger(__name__)

MODELS: Final[tuple[str, str]] = ("xgboost", "lightgbm")
VARIANTS: Final[tuple[str, str, str]] = ("A", "B", "C")
CORE_FOLDS: Final[tuple[str, str, str, str, str]] = (
    "fold1",
    "fold2",
    "fold3",
    "fold4",
    "final_test",
)


def _snapshot(
    stage: str,
    model_key: str,
    variant: str,
    fold: str,
    data: ModelData,
    parameters: dict[str, str | int | float | bool],
) -> dict[str, JsonValue]:
    """Build one auditable run configuration snapshot."""
    data_config = load_data_config()
    model_config = load_model_config()
    split = data_config["splits"][fold]
    return {
        "stage": stage,
        "model": model_key,
        "variant": variant,
        "fold": fold,
        "seed": model_config["seed"],
        "target_column": data_config["modeling"]["target_column"],
        "row_key_columns": list(data_config["modeling"]["row_key_columns"]),
        "feature_columns": list(data["X_train"].columns),
        "split": {
            name: value.isoformat()
            for name, value in split.items()
            if name != "eval_kind"
        },
        "evaluation_kind": split["eval_kind"],
        "parameters": parameters,
        "package_versions": package_snapshot(),
    }


def _save_training_run(
    stage: str,
    model_key: str,
    variant: str,
    fold: str,
    directory: Path,
    data: ModelData,
    result: TrainingResult,
    parameters: dict[str, str | int | float | bool],
) -> None:
    """Persist one model, keyed prediction, metrics, and snapshot."""
    save_pickle_model(directory / "model.pkl", result["model"])
    save_prediction_artifacts(
        directory,
        data["evaluation_keys"],
        data["y_evaluation"],
        result["y_pred"],
        result["metrics"],
    )
    write_snapshot(
        directory / "run_config.json",
        _snapshot(stage, model_key, variant, fold, data, parameters),
    )


def run_hpo_stage(model_key: str) -> None:
    """Run auditable Variant A HPO for one model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'")
    data = load_data("hpo", "A")
    run_hpo(model_key, data)


def _save_hpo_evaluation(
    model_key: str,
    tuned_parameters: dict[str, str | int | float | bool] | None,
    stage: str,
    directory_name: str,
) -> None:
    """Evaluate one explicitly selected HPO configuration."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'")
    data = load_data("hpo", "A")
    model_config = load_model_config()
    result = train_and_evaluate(model_key=model_key, data=data, tuned_parameters=tuned_parameters, fold="hpo")
    target_directory = model_config["paths"]["results"] / "A" / "hpo" / directory_name
    _save_training_run(
        stage,
        model_key,
        "A",
        "hpo",
        target_directory,
        data,
        result,
        model_parameters(model_key, tuned_parameters),
    )


def run_baseline_hpo(model_key: str) -> None:
    """Evaluate one baseline model on the HPO split."""
    _save_hpo_evaluation(model_key, None, "baseline_hpo", f"{model_key}_baseline")


def run_tuned_hpo(model_key: str) -> None:
    """Evaluate one frozen HPO model on the HPO split."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'")
    model_config = load_model_config()
    tuned_parameters = load_model_parameters(
        model_config["paths"]["results_hpo"] / model_key / "best_params.json"
    )
    _save_hpo_evaluation(model_key, tuned_parameters, "tuned_hpo", model_key)


def run_core_stage(model_key: str, variant: str, fold: str) -> None:
    """Train and persist one tuned model-variant-fold combination."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'")
    if fold not in CORE_FOLDS:
        raise ValueError(f"Unknown core fold '{fold}'")
    data = load_data(fold, variant)
    model_config = load_model_config()
    frozen = load_model_parameters(
        model_config["paths"]["results_hpo"] / model_key / "best_params.json"
    )
    result = train_and_evaluate(data, model_key, frozen, fold)
    directory = model_config["paths"]["results"] / variant / fold / model_key
    _save_training_run(
        "core",
        model_key,
        variant,
        fold,
        directory,
        data,
        result,
        model_parameters(model_key, frozen),
    )


def run_shap_stage(model_key: str, variant: str, fold: str) -> None:
    """Run SHAP for one existing core model artifact."""
    if model_key not in MODELS or variant not in VARIANTS or fold not in CORE_FOLDS:
        raise ValueError(f"Invalid SHAP selection: model={model_key}, variant={variant}, fold={fold}")
    data = load_data(fold, variant)
    run_shap(fold, variant, model_key, data)


def generate_all_sample_keys() -> None:
    """Generate semantic SHAP sample keys for every core fold."""
    for fold in CORE_FOLDS:
        save_sample_keys(fold, load_data(fold, "A"))


def run_all_stages() -> None:
    """Run sampling, HPO, baseline, core training, and SHAP in protocol order."""
    generate_all_sample_keys()
    for model_key in MODELS:
        run_hpo_stage(model_key)
        run_baseline_hpo(model_key)
        run_tuned_hpo(model_key)
    core_runs = product(VARIANTS, CORE_FOLDS, MODELS)
    for variant, fold, model_key in tqdm(
        core_runs,
        total=len(VARIANTS) * len(CORE_FOLDS) * len(MODELS),
        desc="Core training and SHAP",
        unit="run",
    ):
        run_core_stage(model_key, variant, fold)
        run_shap_stage(model_key, variant, fold)
    logger.info("pipeline_completed", extra={"models": MODELS, "variants": VARIANTS})
