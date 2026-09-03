import pickle
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utilities import load_config, load_data, resolve_path

cfg = load_config()


class RunShap:
    def __init__(self, fold: str, model_key: str, variant: str):
        self.fold = fold
        self.model_key = model_key
        self.variant = variant

    def load_saved_model(self):
        model_dir = resolve_path(cfg, "results") / self.variant / self.fold / self.model_key
        model_path = model_dir / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print(f"Loaded model from: {model_path}")
        return model

    def load_sample_indices(self):
        sample_path = (resolve_path(cfg, "data_fold") / self.fold/ "sample_indices.csv")
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Sample indices not found at: {sample_path}. Run main() first."
            )

        sample_indices = pd.read_csv(sample_path)["row_index"].tolist()
        print(f"Loaded {len(sample_indices)} sample indices from: {sample_path}")
        return sample_indices
    def run(self, X_train, y_train, X_test, y_test):
        print(f"Running SHAP analysis on fold {self.fold}")

        sampled_indices = self.load_sample_indices()
        missing_indices = set(sampled_indices) - set(X_test.index)
        if missing_indices:
            raise KeyError(
                f"Sample indices are not present in X_test: {sorted(missing_indices)[:5]}"
            )

        X_sample = X_test.loc[sampled_indices].reset_index(drop=True)
        X_sample_model = X_sample.drop(columns=["pu_location_id"])
        model = self.load_saved_model()
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

        shap_values = explainer(X_sample_model)

        model_dir = resolve_path(cfg, "results") / self.variant / self.fold / self.model_key
        self.save_shap_artifacts(shap_values, X_sample_model, model_dir)
        self.save_importance(shap_values, X_sample_model, model_dir)

        return shap_values

def main():
    # chạy main để tạo ra các sample_indices (shap_row_keys) để phân tích shap trên các row này thay vì toàn bộ 
    for fold in ["fold1", "fold2", "fold3", "fold4"]:
        _, _, X_test, _ = load_data(fold=fold, variant="A")
        sample_indices = []
        rng = np.random.default_rng(cfg["seed"])
        for _, group in X_test.groupby("pu_location_id", observed=False, sort=False):
            sample_size = min(cfg["shap"]["zone_sample_size"], len(group))
            sample_indices.extend(group.sample(n=sample_size, random_state=rng).index)

        sample_path = resolve_path(cfg, "data_fold") / fold / "sample_indices.csv"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row_index": sample_indices}).to_csv(sample_path, index=False)
        print(f"Saved {len(sample_indices)} sample indices to: {sample_path}")

if __name__ == "__main__":
    main()