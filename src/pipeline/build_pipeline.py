from src.utilities import load_config, load_data, resolve_path
from src.pipeline.train_test import TrainTest
cfg = load_config()
model_keys = ["xgboost", "lightgbm"]
class BuildPipeline:
    def __init__(self, variant):
        self.variant = variant
        self.results = []

    def run(self, fold):
        
        print(f"Running build pipeline for variant {self.variant} on fold {fold}")

        X_train, y_train, X_test, y_test = load_data(fold, self.variant)

        for model_key in model_keys:
            train_test = TrainTest([X_train, y_train], [X_test, y_test], model_key)
            train_test.run(fold)
            train_test.log(model_name=model_key, variant=self.variant, fold=fold)
