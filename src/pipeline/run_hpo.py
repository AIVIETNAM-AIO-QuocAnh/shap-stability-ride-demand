"""Optuna HPO for the locked Variant A HPO split."""

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
    """Resolve the configured categorical and numeric search parameters."""
    config = load_model_config()
    search_space = config["hpo"]["search_space"].get(model_key)
    if search_space is None:
        raise ValueError(f"No HPO search space configured for model '{model_key}'")
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
            raise ValueError(f"Unsupported HPO specification for {model_key}.{name}")
    return tuned


def _objective(
    trial: optuna.Trial, data: ModelData, model_key: str
) -> float:
    """Train one trial and return MAE on the locked July evaluation window."""
    tuned = _suggest_parameters(trial, model_key)
    model = build_model(model_key, model_parameters(model_key, tuned))
    model.fit(data["X_train"], data["y_train"])
    return mean_absolute_error(data["y_evaluation"], model.predict(data["X_evaluation"]))


def _trial_rows(study: optuna.Study) -> list[dict[str, JsonValue]]:
    """Convert the complete Optuna trial history to validated CSV records."""
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
    """Run exactly the configured HPO study and persist auditable evidence."""
    config = load_model_config()
    if model_key not in config["models"]:
        raise ValueError(f"Unsupported model '{model_key}'; expected xgboost or lightgbm")
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: _objective(trial, data, model_key),
        n_trials=config["hpo"]["n_trials"],
        show_progress_bar=True,
    )
    if len(study.trials) != config["hpo"]["n_trials"]:
        raise RuntimeError(
            f"HPO for {model_key} produced {len(study.trials)} trials; "
            f"expected {config['hpo']['n_trials']}"
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
        raise FileExistsError(f"Refusing to overwrite existing artifact: {trials_path}")
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
