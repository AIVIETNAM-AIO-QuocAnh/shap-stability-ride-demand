"""Optuna HPO cho Variant A HPO split đã khoá."""

import logging
from pathlib import Path

import optuna
import pandas as pd

from src.load_config import ModelParameters, load_data_config, load_model_config
from src.pipeline.artifacts import JsonValue, package_snapshot, write_json
from src.pipeline.metrics import mean_absolute_error
from src.pipeline.train_test import build_model, model_parameters
from src.utilities import ModelData


logger = logging.getLogger(__name__)


def _suggest_parameters(trial: optuna.Trial, model_key: str) -> ModelParameters:
    """Xác định categorical và numeric search parameter theo config."""
    config = load_model_config()
    search_space = config["hpo"]["search_space"].get(model_key)
    if search_space is None:
        raise ValueError(f"Model '{model_key}' chưa có HPO search space trong config")
    tuned: ModelParameters = {}
    for name, specification in search_space.items():
        if isinstance(specification, list):
            tuned[name] = trial.suggest_categorical(name, specification)
        elif isinstance(specification, dict):
            tuned[name] = trial.suggest_float(
                name,
                specification["low"],
                specification["high"],
                log=specification.get("log", False),
            )
        else:
            raise ValueError(f"HPO specification không được hỗ trợ cho {model_key}.{name}")
    return tuned


def _objective(
    trial: optuna.Trial, data: ModelData, model_key: str
) -> float:
    """Train một trial và trả về MAE trên July evaluation window đã khoá."""
    tuned = _suggest_parameters(trial, model_key)
    model = build_model(model_key, model_parameters(model_key, tuned))
    model.fit(data["X_train"], data["y_train"])
    return mean_absolute_error(data["y_evaluation"], model.predict(data["X_evaluation"]))


def _trial_rows(study: optuna.Study) -> list[dict[str, JsonValue]]:
    """Chuyển toàn bộ Optuna trial history thành CSV record đã validate."""
    rows: list[dict[str, JsonValue]] = []
    for trial in study.trials:
        row: dict[str, JsonValue] = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "mae": None if trial.value is None else float(trial.value),
        }
        row.update({name: value for name, value in trial.params.items()})
        rows.append(row)
    return rows


def run_hpo(model_key: str, data: ModelData) -> ModelParameters:
    """Chạy đúng HPO study theo config và lưu evidence có thể audit."""
    config = load_model_config()
    if model_key not in config["models"]:
        raise ValueError(f"Model '{model_key}' không được hỗ trợ; kỳ vọng xgboost hoặc lightgbm")
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: _objective(trial, data, model_key),
        n_trials=config["hpo"]["n_trials"],
        show_progress_bar=True,
    )
    if len(study.trials) != config["hpo"]["n_trials"]:
        raise RuntimeError(
            f"HPO cho {model_key} tạo {len(study.trials)} trial; "
            f"kỳ vọng {config['hpo']['n_trials']}"
        )

    output_dir: Path = config["paths"]["results_hpo"] / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    best_parameters: ModelParameters = {
        name: value for name, value in study.best_params.items()
    }
    write_json(
        output_dir / "best_params.json",
        {name: value for name, value in best_parameters.items()},
    )
    trials_path = output_dir / "trials.csv"
    if trials_path.exists():
        raise FileExistsError(f"Từ chối ghi đè artifact đã tồn tại: {trials_path}")
    pd.DataFrame(_trial_rows(study)).to_csv(trials_path, index=False)
    write_json(
        output_dir / "run_config.json",
        {
            "stage": "hpo",
            "model": model_key,
            "variant": "A",
            "fold": "hpo",
            "seed": config["seed"],
            "n_trials": config["hpo"]["n_trials"],
            "objective": "mae",
            "sampler": "TPESampler",
            "target_column": load_data_config()["modeling"]["target_column"],
            "row_key_columns": list(load_data_config()["modeling"]["row_key_columns"]),
            "feature_columns": list(data["X_train"].columns),
            "split": {
                name: value.isoformat()
                for name, value in load_data_config()["splits"]["hpo"].items()
                if name != "eval_kind"
            },
            "evaluation_kind": load_data_config()["splits"]["hpo"]["eval_kind"],
            "package_versions": package_snapshot(),
        },
    )
    logger.info("hpo_completed", extra={"model": model_key, "trials": len(study.trials)})
    return best_parameters
