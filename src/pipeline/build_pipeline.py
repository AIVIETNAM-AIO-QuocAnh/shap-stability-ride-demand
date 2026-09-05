import json
import pickle
from pathlib import Path
import pandas as pd

from src.configuration import load_model_config
from src.utilities import load_data
from src.pipeline.train_test import TrainTest
from src.pipeline.run_hpo import RunHpo
from src.pipeline.run_shap import RunShap

cfg = load_model_config()
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
        X_train, y_train, X_test, y_test = load_data(fold=fold, variant=self.variant)
        for model_key in model_keys:
            shap_runner = RunShap(fold=fold, model_key=model_key, variant=self.variant)
            shap_runner.run(X_train, y_train, X_test, y_test)
        pass

    def run_fold(self, fold, tuned=True):
        X_train, y_train, X_test, y_test = load_data(fold=fold, variant=self.variant)

        for model_key in model_keys:
            train_test = TrainTest(
                [X_train.drop(columns=["pu_location_id"]), y_train],
                [X_test.drop(columns=["pu_location_id"]), y_test],
                model_key=model_key,
                use_tuned=tuned,
            )
            model, metrics, y_pred = train_test.run(fold=fold)

            path = self.save_artifacts(
                model=model,
                metrics=metrics,
                y_pred=y_pred,
                y_true=y_test,
                fold=fold,
                model_key=model_key,
                tuned=tuned,
            )
            print(f"Saved results to: {path}")

    def save_artifacts(self, model, metrics, y_pred, y_true, fold, model_key, tuned=True):
        model_dir_name = model_key if tuned else f"{model_key}_baseline"
        base_path = cfg["paths"]["results"] / self.variant / fold / model_dir_name
        base_path.mkdir(parents=True, exist_ok=True)

        model_path = base_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        metrics_path = base_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, sort_keys=True)

        pred_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
        })
        pred_df.to_csv(base_path / "y_pred.csv", index=False)

        return base_path
