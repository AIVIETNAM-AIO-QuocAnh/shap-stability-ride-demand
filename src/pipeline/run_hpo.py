import optuna
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

from src.utilities import load_config, calculate_metrics

cfg = load_config()

class RunHpo: 
    def __init__(self, X_train, y_train, X_test, y_test, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
    def objective(self, trial : optuna.Trial):
        common_params = {}
        if self.model == "xgboost":
            params = {
                **common_params,
            }
            model = XGBRegressor(**params)
        else :
            params = {
                **common_params,
            }
            model = LGBMRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return calculate_metrics("mae", self.y_test, y_pred)
    def run(self):
        sampler = optuna.samplers.TPESampler(seed=cfg["seed"])
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self.objective, n_trials=cfg["models"]["optuna"]["n_trials"])
        return study.best_params
