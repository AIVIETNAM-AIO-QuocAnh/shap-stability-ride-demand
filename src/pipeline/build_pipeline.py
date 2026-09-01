from src.utilities import load_config, load_data, resolve_path
from src.pipeline.train_test import TrainTest
from src.pipeline.run_shap import RunShap
cfg = load_config()
model_keys = ["xgboost", "lightgbm"]
class BuildPipeline:
    def __init__(self, variant):
        self.variant = variant
        self.results = []

    def run(self, fold):
        
        print(f"Running build pipeline for variant {self.variant} on fold {fold}")

        self.X_train, self.y_train, self.X_test, self.y_test = load_data(fold, self.variant)
        
        # print(X_train.head())

        # for model_key in model_keys:
        #     train_test = TrainTest([X_train.drop(columns=["PULocationID"]), y_train], [X_test.drop(columns=["PULocationID"]), y_test], model_key)
        #     train_test.run(fold)
        #     train_test.log(model_name=model_key, variant=self.variant, fold=fold)
    def run_shap(self, fold):
        shap_analysis = RunShap(variant=self.variant, fold=fold)
        shap_analysis.run(self.X_train, self.y_train, self.X_test, self.y_test)