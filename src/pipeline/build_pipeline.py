from src.utilities import load_config, load_data, resolve_path
from src.pipeline.train_test import TrainTest
from src.pipeline.run_hpo import RunHpo

cfg = load_config()
model_keys = ["xgboost", "lightgbm"]

class BuildPipeline:
    def __init__(self, variant):
        self.variant = variant
        self.results = []
    def run_hpo(self):
        X_train, y_train, X_test, y_test = load_data(fold="hpo", variant=self.variant)
        for model_key in model_keys:
            hpo = RunHpo(X_train, y_train, X_test, y_test, model_key=model_key)
            hpo.run()
    def run_shap(self, fold):
        pass
    def run_fold(self, fold):
        pass