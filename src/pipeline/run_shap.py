import pickle
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

    def save_shap_artifacts(self, shap_values, model_dir: Path):
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

    def run(self, X_train, y_train, X_test, y_test):
        print(f"Running SHAP analysis on fold {self.fold}")

        # @TODO : nhờ bên data viết 1 cái hàm để load X_sample từ /data thay vì làm thủ công tại đây, để sau cho QA check các sample row key có thống nhất ko
        X_sample = (
            X_test.groupby("PULocationID", group_keys=False, observed=False)
            .apply(
                lambda x: x.sample(n=cfg["shap"]["zone_sample_size"], random_state=cfg["seed"]),
                include_groups=True,
            )
            .reset_index(drop=True)
        )
        print(X_sample.head())

        X_sample_model = X_sample.drop(columns=["PULocationID"], axis=1)
        model = self.load_saved_model()
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

        shap_values = explainer(X_sample_model)
        print(shap_values)

        model_dir = resolve_path(cfg, "results") / self.variant / self.fold / self.model_key
        self.save_shap_artifacts(shap_values, model_dir)

        return shap_values

