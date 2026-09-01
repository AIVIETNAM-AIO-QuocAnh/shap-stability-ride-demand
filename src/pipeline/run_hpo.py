import json
from pathlib import Path

import optuna
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

from src.utilities import load_config, calculate_metrics

cfg = load_config()


class RunHpo:
    def __init__(self, X_train, y_train, X_test, y_test, model_key):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_key = model_key

    def objective(self, trial: optuna.Trial):
        common_params = {}
        if self.model_key == "xgboost":
            params = {
                **common_params,
                # @TODO
            }
            model = XGBRegressor(**params)
        else:
            params = {
                **common_params,
                # @TODO
            }
            model = LGBMRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return calculate_metrics("mae", self.y_test, y_pred)

    def save_best_params(self, best_params: dict):
        hpo_dir = resolve_path(cfg, "results_hpo")

        file_path = hpo_dir / self.model_key / "best_params.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False, sort_keys=True)

        return file_path

    def run(self):
        sampler = optuna.samplers.TPESampler(seed=cfg["seed"])
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self.objective, n_trials=cfg["models"]["optuna"]["n_trials"])

        best_params = study.best_params
        self.save_best_params(best_params)
        return best_params
