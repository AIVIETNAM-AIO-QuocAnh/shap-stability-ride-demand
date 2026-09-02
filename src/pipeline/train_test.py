import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

from src.utilities import load_config, resolve_path, calculate_metrics

cfg = load_config()


class TrainTest:
    def __init__(self, train_data, test_data, model_key, use_tuned=True):
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.model_key = model_key
        self.metrics = []

        common_params = cfg["models"][self.model_key]
        self.hyperparams = {
            "random_state": cfg["seed"],
            **common_params
        }

        if use_tuned:
            hpo_dir = resolve_path(cfg, "results_hpo")
            with open(hpo_dir / self.model_key / "best_params.json") as f:
                tuned_params = json.load(f)
            self.hyperparams.update(tuned_params)

    def run(self, fold):
        if self.model_key == "xgboost":
            model = XGBRegressor(**self.hyperparams)
        else:
            model = LGBMRegressor(**self.hyperparams)

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        mae = calculate_metrics("mae", self.y_test, y_pred)
        rmse = calculate_metrics("rmse", self.y_test, y_pred)
        wape = calculate_metrics("wape", self.y_test, y_pred)

        self.metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "wape": float(wape),
        }

        print(f"Metrics for {self.model_key} on fold {fold}: {self.metrics}")
        return model, self.metrics, y_pred

