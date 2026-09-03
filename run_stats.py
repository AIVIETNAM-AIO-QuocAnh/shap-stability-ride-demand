from src.utilities import load_config, resolve_path
import pandas as pd
import json

cfg = load_config()

class RunStats:
    def __init__(self, variants=("A","B","C"), models=("xgboost","lightgbm"),
                 folds=("fold1","fold2","fold3","fold4","final_test")):
        self.variants = variants
        self.models = models
        self.folds = folds
        self.result_base_path = resolve_path(cfg, "results")
        self.stats_folder = resolve_path(cfg, "results_stats")
        self.stats_folder.mkdir(parents=True, exist_ok=True)
        self.variant_map_file = resolve_path(cfg, "variant_map")
        with open(self.variant_map_file, "r") as f:
            self.variant_map = json.load(f)
    
    def load_metrics_df(self):
        metrics = []

        for variant in self.variants:
            for fold in self.folds:
                for model in self.models:
                   result_path = self.result_base_path / variant / fold / model / "metrics.json"
                   with open(result_path, 'r') as f:
                        data = json.load(f)
                        metrics.append({
                                "variant": variant,
                                "fold": fold,
                                "model": model,
                                **data
                            }
                        )
        metrics_df = pd.DataFrame(metrics)
        metrics_df["fold"] = pd.Categorical(metrics_df["fold"], categories=self.folds, ordered=True)

        metrics_df.to_csv(self.stats_folder / "performance_summary.csv", index=False)

        return metrics_df

    def load_shap_long_df(self):
        shap_frames = []

        for variant in self.variants:
            variant_features = self.variant_map["variants"][variant]["weekly_features"]
            for fold in self.folds:
                for model in self.models:
                    result_path = self.result_base_path / variant / fold / model / "shap_importance.csv"
                    df = pd.read_csv(result_path)
                    df = df[df["feature"].isin(variant_features)].copy()

                    df["variant"] = variant
                    df["fold"] = fold
                    df["model"] = model

                    shap_frames.append(df)

        shap_long_df = pd.concat(shap_frames, ignore_index=True)
        shap_long_df = shap_long_df[["variant", "model", "fold", "feature", "importance"]]
        shap_long_df["fold"] = pd.Categorical(shap_long_df["fold"], categories=self.folds, ordered=True)

        expected_rows = len(self.folds) * len(self.models) * sum(
            len(self.variant_map["variants"][variant]["weekly_features"])
            for variant in self.variants
        )
        if len(shap_long_df) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} weekly-lag SHAP rows but got {len(shap_long_df)}. "
                "Check that weekly_features in variant_map match the feature names in shap_importance.csv."
            )

        return shap_long_df

    def compute_performance_summary(self, metrics_df):
        performance_aggregated_df = (
            metrics_df.groupby(["variant", "model"], observed=True)[["mae", "rmse", "wape"]]
            .agg(["mean", "std"])
        )
        performance_aggregated_df.columns = [
            f"{metric}_{stat}" for metric, stat in performance_aggregated_df.columns
        ]
        performance_aggregated_df = performance_aggregated_df.reset_index()

        performance_aggregated_df.to_csv(self.stats_folder / "performance_aggregated.csv", index=False)

        return performance_aggregated_df

    def compute_feature_stability(self, shap_long_df):
        feature_stability_df = (
            shap_long_df.groupby(["variant", "model", "feature"], observed=True)["importance"]
            .agg(mean_importance="mean", std_importance="std")
            .reset_index()
        )
        feature_stability_df["cv"] = feature_stability_df["std_importance"] / feature_stability_df["mean_importance"]

        feature_stability_df.to_csv(self.stats_folder / "feature_importance_stability.csv", index=False)

        return feature_stability_df

    def compute_group_stability(self):
        group_importance = []

        for variant in self.variants:
            for fold in self.folds:
                for model in self.models:
                    result_path = self.result_base_path / variant / fold / model / "shap_weekly_group.json"
                    with open(result_path, "r") as f:
                        data = json.load(f)
                    group_importance.append({
                        "variant": variant,
                        "fold": fold,
                        "model": model,
                        "group_importance": data["weekly_group_importance"],
                    })

        group_importance_df = pd.DataFrame(group_importance)

        group_stability_df = (
            group_importance_df.groupby(["variant", "model"], observed=True)["group_importance"]
            .agg(mean_group_importance="mean", std_group_importance="std")
            .reset_index()
        )
        group_stability_df["cv"] = group_stability_df["std_group_importance"] / group_stability_df["mean_group_importance"]

        group_stability_df.to_csv(self.stats_folder / "weekly_group_stability.csv", index=False)

        return group_stability_df


run_stats = RunStats()
metrics_df = run_stats.load_metrics_df()
shap_long_df = run_stats.load_shap_long_df()

performance_aggregated_df = run_stats.compute_performance_summary(metrics_df)
feature_stability_df = run_stats.compute_feature_stability(shap_long_df)
group_stability_df = run_stats.compute_group_stability()

print(performance_aggregated_df)
print(feature_stability_df)
print(group_stability_df)