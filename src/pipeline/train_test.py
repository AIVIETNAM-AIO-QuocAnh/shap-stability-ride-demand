import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.utilities import load_config, resolve_path, calculate_metrics

cfg = load_config()


class TrainTest:
    def __init__(self, train_data, test_data, model_key):
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.model_key = model_key
        self.metrics = []
        self.hyperparams = cfg["models"][model_key]  # @TODO: thay bằng hyperparams từ file JSON của HPO sau khi freeze

    def run(self, fold):
        print(self.model_key)
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

