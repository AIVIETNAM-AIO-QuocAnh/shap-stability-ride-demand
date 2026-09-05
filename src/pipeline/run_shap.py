import pickle
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.configuration import load_data_config, load_model_config
from src.utilities import load_data

data_cfg = load_data_config()
model_cfg = load_model_config()


class RunShap:
    def __init__(self, fold: str, model_key: str, variant: str):
        self.fold = fold
        self.model_key = model_key
        self.variant = variant

    def save_shap_artifacts(self, shap_values, X_sample_model, model_dir: Path):
        model_dir.mkdir(parents=True, exist_ok=True)

        shap_values_path = model_dir / "shap_values.pkl"
        with open(shap_values_path, "wb") as f:
            pickle.dump(shap_values, f)
        print(f"Saved SHAP values to: {shap_values_path}")

        plt.figure()
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(model_dir / "shap_bar.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved SHAP bar plot to: {model_dir / 'shap_bar.png'}")

        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(model_dir / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved SHAP beeswarm plot to: {model_dir / 'shap_beeswarm.png'}")

    def save_importance(self, shap_values, X_sample_model, model_dir: Path):
        importance = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": X_sample_model.columns,
            "importance": importance
        }).sort_values("importance", ascending=False)

        importance_path = model_dir / "shap_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"Saved SHAP importance to: {importance_path}")

        # Weekly-group aggregation based on variant
        variant_map_path = data_cfg["paths"]["variant_map"]
        with open(variant_map_path) as f:
            variant_map = json.load(f)

        weekly_features = variant_map["variants"][self.variant]["weekly_features"]
        weekly_importance = importance_df[importance_df["feature"].isin(weekly_features)]["importance"].sum()

        weekly_group_path = model_dir / "shap_weekly_group.json"
        with open(weekly_group_path, "w") as f:
            json.dump({"weekly_group_importance": float(weekly_importance)}, f, indent=2)
        print(f"Saved SHAP weekly-group importance to: {weekly_group_path}")

    def load_saved_model(self):
        model_dir = model_cfg["paths"]["results"] / self.variant / self.fold / self.model_key
        model_path = model_dir / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print(f"Loaded model from: {model_path}")
        return model

    def load_sample_indices(self):
        sample_path = data_cfg["paths"]["folds_dir"] / self.fold / "sample_indices.csv"
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

        model_dir = model_cfg["paths"]["results"] / self.variant / self.fold / self.model_key
        self.save_shap_artifacts(shap_values, X_sample_model, model_dir)
        self.save_importance(shap_values, X_sample_model, model_dir)

        return shap_values

def main():
    # chạy main để tạo ra các sample_indices (shap_row_keys) để phân tích shap trên các row này thay vì toàn bộ 
    for fold in tuple(name for name in data_cfg["splits"] if name != "hpo"):
        _, _, X_test, _ = load_data(fold=fold, variant="A")
        sample_indices = []
        rng = np.random.default_rng(model_cfg["seed"])
        for _, group in X_test.groupby("pu_location_id", observed=False, sort=False):
            sample_size = min(model_cfg["shap"]["zone_sample_size"], len(group))
            sample_indices.extend(group.sample(n=sample_size, random_state=rng).index)

        sample_path = data_cfg["paths"]["folds_dir"] / fold / "sample_indices.csv"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row_index": sample_indices}).to_csv(sample_path, index=False)
        print(f"Saved {len(sample_indices)} sample indices to: {sample_path}")

if __name__ == "__main__":
    main()
