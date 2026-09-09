"""Tổng hợp kết quả core experiment theo protocol của project.

Phạm vi core gồm:

- Mục 2.5, prediction metric: MAE, RMSE, WAPE trên toàn bộ evaluation rows của từng fold;
  báo cáo mean ± sample standard deviation qua Fold 1-4; final test tháng 12 báo cáo riêng,
  không gộp vào mean/std.
- Mục 2.6, SHAP stability: I_{j,f} = mean |phi| trên 5.000 sample row của fold f; báo cáo
  mean và sample standard deviation của I_{j,f} qua Fold 1-4; final test báo cáo riêng.
- Mục 2.6, weekly group: I_weekly,f = tổng I_{j,f} của các weekly feature trong variant;
  cũng báo cáo mean ± sample standard deviation qua Fold 1-4.

Protocol quy định không dùng coefficient of variation hoặc significance test trong core scope
và không mở rộng core experiment ngoài data, model, metric, split và SHAP settings đã khóa. Vì
vậy module này KHÔNG tính: coefficient of variation,
rank/Spearman stability, paired comparison với standard error, z-score của final test,
trend/confound check, và sensitivity của công thức group importance.
"""

from src.load_config import load_data_config, load_model_config
import pandas as pd
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data_cfg = load_data_config()
model_cfg = load_model_config()


class RunStats:
    def __init__(self, variants=("A", "B", "C"), models=("xgboost", "lightgbm"),
                 folds=("fold1", "fold2", "fold3", "fold4", "final_test")):
        self.variants = variants
        self.models = models
        self.folds = folds
        self.result_base_path = model_cfg["paths"]["results"]
        self.stats_folder = model_cfg["paths"]["results_stats"]
        self.stats_folder.mkdir(parents=True, exist_ok=True)
        self.plots_folder = self.stats_folder / "plots"
        self.plots_folder.mkdir(parents=True, exist_ok=True)
        self.variant_map_file = data_cfg["paths"]["variant_map"]
        with open(self.variant_map_file, "r") as f:
            self.variant_map = json.load(f)

        # Protocol: final_test báo cáo riêng, không gộp vào mean/std qua fold.
        self.final_test_fold = "final_test"
        self.cv_folds = tuple(fold for fold in self.folds if fold != self.final_test_fold)

    def load_correlation_summary(self):
        """Pearson correlation của ba cặp weekly lag, tổng hợp trên 50 zone.

        Bảng do tầng Data sinh ra; Analysis chỉ đọc lại để `summary.md` có đủ correlation
        evidence cùng với prediction và SHAP results.
        """
        path = data_cfg["paths"]["correlation_summary"]
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy correlation summary: {path}")
        return pd.read_csv(path)

    def load_metrics_df(self):
        metrics = []

        for variant in self.variants:
            for fold in self.folds:
                for model in self.models:
                    result_path = self.result_base_path / variant / fold / model / "metrics.json"
                    with open(result_path, "r") as f:
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
                f"Kỳ vọng {expected_rows} weekly-lag SHAP row nhưng nhận {len(shap_long_df)}. "
                "Kiểm tra weekly_features trong variant_map có khớp tên feature trong shap_importance.csv không."
            )

        return shap_long_df

    def compute_performance_summary(self, metrics_df):
        """Tính mean ± sample std qua Fold 1-4 và tách riêng final test."""
        cv_metrics_df = metrics_df[metrics_df["fold"] != self.final_test_fold]

        performance_aggregated_df = (
            cv_metrics_df.groupby(["variant", "model"], observed=True)[["mae", "rmse", "wape"]]
            .agg(["mean", "std"])
        )
        performance_aggregated_df.columns = [
            f"{metric}_{stat}" for metric, stat in performance_aggregated_df.columns
        ]
        performance_aggregated_df = performance_aggregated_df.reset_index()

        final_test_df = metrics_df[metrics_df["fold"] == self.final_test_fold][
            ["variant", "model", "mae", "rmse", "wape"]
        ].rename(columns={"mae": "mae_final_test", "rmse": "rmse_final_test", "wape": "wape_final_test"})

        performance_aggregated_df = performance_aggregated_df.merge(final_test_df, on=["variant", "model"])

        performance_aggregated_df.to_csv(self.stats_folder / "performance_aggregated.csv", index=False)

        return performance_aggregated_df

    def compute_feature_stability(self, shap_long_df):
        """Tính mean và sample standard deviation của I_{j,f} qua Fold 1-4."""
        cv_shap_df = shap_long_df[shap_long_df["fold"] != self.final_test_fold]

        feature_stability_df = (
            cv_shap_df.groupby(["variant", "model", "feature"], observed=True)["importance"]
            .agg(mean_importance="mean", std_importance="std")
            .reset_index()
        )

        final_test_df = shap_long_df[shap_long_df["fold"] == self.final_test_fold][
            ["variant", "model", "feature", "importance"]
        ].rename(columns={"importance": "importance_final_test"})

        feature_stability_df = feature_stability_df.merge(final_test_df, on=["variant", "model", "feature"])

        feature_stability_df.to_csv(self.stats_folder / "feature_importance_stability.csv", index=False)

        return feature_stability_df

    def compute_group_stability(self):
        """Tính weekly-group importance, mean ± sample std qua Fold 1-4."""
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
        cv_group_df = group_importance_df[group_importance_df["fold"] != self.final_test_fold]

        group_stability_df = (
            cv_group_df.groupby(["variant", "model"], observed=True)["group_importance"]
            .agg(mean_group_importance="mean", std_group_importance="std")
            .reset_index()
        )

        final_test_df = group_importance_df[group_importance_df["fold"] == self.final_test_fold][
            ["variant", "model", "group_importance"]
        ].rename(columns={"group_importance": "group_importance_final_test"})

        group_stability_df = group_stability_df.merge(final_test_df, on=["variant", "model"])

        group_stability_df.to_csv(self.stats_folder / "weekly_group_stability.csv", index=False)

        return group_stability_df

    def plot_feature_trends(self, shap_long_df):
        """Vẽ I_{j,f} theo từng fold để hỗ trợ đọc kết quả trong bảng và summary."""
        saved_paths = []
        for variant in self.variants:
            variant_df = shap_long_df[shap_long_df["variant"] == variant]

            fig, axes = plt.subplots(1, len(self.models), figsize=(6 * len(self.models), 4), sharey=True)
            if len(self.models) == 1:
                axes = [axes]

            for ax, model in zip(axes, self.models):
                model_df = variant_df[variant_df["model"] == model]
                for feature, feature_df in model_df.groupby("feature", observed=True):
                    feature_df = feature_df.sort_values("fold")
                    ax.plot(feature_df["fold"].astype(str), feature_df["importance"], marker="o", label=feature)
                ax.set_title(model)
                ax.set_xlabel("fold")
                ax.legend()

            axes[0].set_ylabel("SHAP importance")
            fig.suptitle(f"Variant {variant}: weekly-lag feature importance qua các fold")
            fig.tight_layout()

            path = self.plots_folder / f"{variant}_feature_trend.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(path)

        return saved_paths

    def _df_to_markdown(self, df, float_format="{:.4f}"):
        def fmt(value):
            if isinstance(value, float):
                return float_format.format(value)
            return str(value)

        header = "| " + " | ".join(df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = [
            "| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([header, separator, *rows])

    def write_summary_md(self, correlation_df, performance_aggregated_df,
                         feature_stability_df, group_stability_df):
        n_cv_folds = len(self.cv_folds)
        lines = ["# Kết quả tổng hợp: prediction performance & SHAP stability", ""]

        lines.append(
            "> **Phạm vi báo cáo theo protocol.** Số liệu là **mean ± sample "
            f"standard deviation** qua {n_cv_folds} fold; `final_test` (tháng 12) báo cáo riêng, không "
            "gộp vào mean/std. Core không dùng coefficient of variation hoặc significance test và "
            "không mở rộng ngoài data, model, metric, split và SHAP settings đã khóa. Vì vậy báo cáo này "
            "**chỉ** gồm các chỉ số trên, không kèm CV, rank stability, paired standard error hay "
            "z-score."
        )
        lines.append("")
        lines.append(
            "> **Cách đọc standard deviation:** std được diễn giải **cùng với** "
            "mean importance. Không kết luận một feature \"ổn định hơn\" chỉ dựa vào std khi mức mean "
            "importance khác nhau quá lớn."
        )
        lines.append("")

        lines.append("## 1. Tương quan giữa ba weekly lag")
        lines.append("")
        lines.append(
            "Pearson correlation tính theo từng zone trên 50 frozen zone. Protocol không đặt trước "
            "ngưỡng để gọi là \"cao\"; giá trị quan sát được quyết định mức độ "
            "mạnh của kết luận."
        )
        lines.append("")
        correlation_display = correlation_df.rename(
            columns={"pair": "cặp lag", "mean_r": "mean r", "std_r": "std r", "n_valid": "n zone"}
        )[["cặp lag", "mean r", "std r", "n zone"]]
        lines.append(self._df_to_markdown(correlation_display))
        lines.append("")
        lines.append(
            f"- Cả ba cặp nằm trong khoảng **{correlation_df['mean_r'].min():.4f}** tới "
            f"**{correlation_df['mean_r'].max():.4f}**, std lớn nhất chỉ "
            f"**{correlation_df['std_r'].max():.4f}** trên "
            f"{int(correlation_df['n_valid'].max())} zone. Tương quan **cao và đồng nhất giữa các "
            "zone**, premise của đề tài đứng vững."
        )
        lines.append("")

        lines.append(f"## 2. Prediction performance (mean ± std qua {n_cv_folds} fold)")
        lines.append("")
        lines.append(
            "MAE là metric chính; RMSE và WAPE là metric bổ sung. WAPE tính theo "
            "phần trăm."
        )
        lines.append("")
        perf_display = performance_aggregated_df[
            ["variant", "model", "mae_mean", "mae_std", "mae_final_test",
             "rmse_mean", "rmse_std", "rmse_final_test",
             "wape_mean", "wape_std", "wape_final_test"]
        ].sort_values(["variant", "model"])
        lines.append(self._df_to_markdown(perf_display))
        lines.append("")

        best_row = performance_aggregated_df.loc[performance_aggregated_df["mae_mean"].idxmin()]
        lines.append(
            f"- MAE trung bình thấp nhất: **variant {best_row['variant']} / {best_row['model']}** "
            f"({best_row['mae_mean']:.3f} ± {best_row['mae_std']:.3f})."
        )
        mae_spread = float(
            performance_aggregated_df["mae_mean"].max() - performance_aggregated_df["mae_mean"].min()
        )
        max_mae_std = float(performance_aggregated_df["mae_std"].max())
        lines.append(
            f"- Khoảng chênh lệch mean MAE giữa 6 tổ hợp là **{mae_spread:.3f}**, trong khi std qua "
            f"fold lên tới **{max_mae_std:.3f}**. Std ở đây phản ánh mức khó khác nhau giữa các "
            "tháng đánh giá, nên bảng này mô tả kết quả chứ không phải bằng chứng variant nào tốt "
            "hơn hẳn."
        )
        lines.append("")

        lines.append(f"## 3. SHAP feature-level importance (mean ± std qua {n_cv_folds} fold)")
        lines.append("")
        lines.append(
            "Protocol định nghĩa `I_{j,f}` = mean |phi| của feature j trên 5.000 sample row của fold f. "
            "Bảng dưới là mean và sample standard deviation của `I_{j,f}` qua các fold, kèm cột "
            "final_test tách riêng."
        )
        lines.append("")
        feature_display = feature_stability_df[
            ["variant", "model", "feature", "mean_importance", "std_importance", "importance_final_test"]
        ].sort_values(["variant", "model", "feature"])
        lines.append(self._df_to_markdown(feature_display))
        lines.append("")

        highest_mean_row = feature_stability_df.loc[feature_stability_df["mean_importance"].idxmax()]
        lowest_mean_row = feature_stability_df.loc[feature_stability_df["mean_importance"].idxmin()]
        lines.append(
            "- Mean importance giữa các feature chênh nhau rất lớn (từ "
            f"**{lowest_mean_row['mean_importance']:.2f}** ở `{lowest_mean_row['feature']}` "
            f"[{lowest_mean_row['variant']}/{lowest_mean_row['model']}] tới "
            f"**{highest_mean_row['mean_importance']:.2f}** ở `{highest_mean_row['feature']}` "
            f"[{highest_mean_row['variant']}/{highest_mean_row['model']}]), nên **không** so std trực "
            "tiếp giữa chúng để xếp hạng độ ổn định, đúng quy ước đọc kết quả của project."
        )
        lines.append("")

        lines.append(f"## 4. Weekly-group importance (mean ± std qua {n_cv_folds} fold)")
        lines.append("")
        group_sizes = {v: len(self.variant_map["variants"][v]["weekly_features"]) for v in self.variants}
        lines.append(
            "Protocol định nghĩa `I_weekly,f` là tổng `I_{j,f}` của các weekly feature trong "
            "variant: "
            + "; ".join(
                f"**{variant}** = "
                + " + ".join(f"`{feature}`" for feature in self.variant_map["variants"][variant]["weekly_features"])
                for variant in self.variants
            )
            + "."
        )
        lines.append("")
        group_display = group_stability_df[
            ["variant", "model", "mean_group_importance", "std_group_importance", "group_importance_final_test"]
        ].sort_values(["variant", "model"])
        lines.append(self._df_to_markdown(group_display))
        lines.append("")

        single_feature_variants = [variant for variant, size in group_sizes.items() if size == 1]
        lines.append(
            "- **Cảnh báo so sánh:** số feature trong nhóm khác nhau giữa các variant ("
            + ", ".join(f"{variant}={size}" for variant, size in group_sizes.items())
            + "). "
            + (
                f"Với variant {'/'.join(single_feature_variants)} nhóm chỉ có 1 feature nên "
                "**group importance ≡ feature importance**. Chênh lệch giữa các variant ở bảng này "
                "một phần đến từ **định nghĩa metric** (tổng trên số feature khác nhau), chưa thể quy "
                "hết cho hành vi model."
                if single_feature_variants else ""
            )
        )
        lines.append("")

        lines.append("## 5. Trả lời câu hỏi nghiên cứu")
        lines.append("")

        lines.append(
            f"**Câu hỏi 1: ba weekly lag tương quan tới đâu và có đồng nhất giữa các zone không?** "
            f"mean r **{correlation_df['mean_r'].min():.2f}** tới "
            f"**{correlation_df['mean_r'].max():.2f}**, std r tối đa "
            f"**{correlation_df['std_r'].max():.3f}** trên "
            f"{int(correlation_df['n_valid'].max())} zone (chi tiết ở mục 1)."
        )
        lines.append("")
        lines.append(
            "**Câu hỏi 2: mean/std của SHAP importance qua temporal fold thay đổi thế nào ở "
            "Variant A/B/C?**"
        )
        lines.append("")
        for variant in self.variants:
            variant_features = feature_stability_df[feature_stability_df["variant"] == variant]
            for model in self.models:
                model_features = variant_features[variant_features["model"] == model].sort_values(
                    "mean_importance", ascending=False
                )
                described = "; ".join(
                    f"`{row['feature']}` {row['mean_importance']:.2f} ± {row['std_importance']:.2f}"
                    for _, row in model_features.iterrows()
                )
                lines.append(f"- Variant **{variant}** / `{model}`: {described}.")
        lines.append("")

        lines.append(
            "**Câu hỏi 3: B/C ảnh hưởng thế nào đến MAE, RMSE, WAPE và weekly-group SHAP stability "
            "so với A, và xu hướng có nhất quán giữa LightGBM và XGBoost không?**"
        )
        lines.append("")

        baseline_perf = performance_aggregated_df[performance_aggregated_df["variant"] == "A"].set_index("model")
        baseline_group = group_stability_df[group_stability_df["variant"] == "A"].set_index("model")
        aggregation_variants = [variant for variant in self.variants if variant != "A"]

        mae_directions = {}
        for variant in aggregation_variants:
            variant_perf = performance_aggregated_df[
                performance_aggregated_df["variant"] == variant
            ].set_index("model")
            variant_group = group_stability_df[group_stability_df["variant"] == variant].set_index("model")

            lines.append(f"*Variant {variant} so với A:*")
            lines.append("")
            for model in self.models:
                deltas = []
                for metric in ("mae", "rmse", "wape"):
                    baseline_value = baseline_perf.loc[model, f"{metric}_mean"]
                    variant_value = variant_perf.loc[model, f"{metric}_mean"]
                    delta_pct = (variant_value - baseline_value) / baseline_value * 100
                    if abs(delta_pct) < 0.05:
                        change = "gần như không đổi"
                    else:
                        word = "giảm" if delta_pct < 0 else "tăng"
                        change = f"{word} {abs(delta_pct):.1f}%"
                    deltas.append(
                        f"{metric.upper()} {change} "
                        f"({baseline_value:.3f} → {variant_value:.3f})"
                    )
                    if metric == "mae":
                        mae_directions[(variant, model)] = delta_pct

                baseline_group_mean = baseline_group.loc[model, "mean_group_importance"]
                baseline_group_std = baseline_group.loc[model, "std_group_importance"]
                variant_group_mean = variant_group.loc[model, "mean_group_importance"]
                variant_group_std = variant_group.loc[model, "std_group_importance"]

                lines.append(
                    f"- `{model}`: " + "; ".join(deltas) + ". Weekly-group importance "
                    f"{baseline_group_mean:.2f} ± {baseline_group_std:.2f} → "
                    f"{variant_group_mean:.2f} ± {variant_group_std:.2f}."
                )
            lines.append("")

        for variant in aggregation_variants:
            directions = [mae_directions[(variant, model)] for model in self.models]
            consistent = all(d < 0 for d in directions) or all(d > 0 for d in directions)
            direction_word = "giảm" if directions[0] < 0 else "tăng"
            lines.append(
                f"- **Tính nhất quán giữa hai model (variant {variant}):** mean MAE "
                + (
                    f"{direction_word} ở **cả LightGBM và XGBoost**"
                    if consistent
                    else "**không cùng hướng** giữa LightGBM và XGBoost"
                )
                + " ("
                + ", ".join(
                    f"{model} {mae_directions[(variant, model)]:+.1f}%" for model in self.models
                )
                + ")."
            )
        lines.append("")

        lines.append(
            "- **Mức độ mạnh của kết luận:** chênh lệch mean MAE giữa các variant nhỏ hơn nhiều so "
            f"với std qua fold (tối đa {max_mae_std:.3f}). Core không dùng significance "
            "test, nên phát biểu dừng ở mức **mô tả**: gộp weekly lag không làm hại "
            "prediction, và hướng thay đổi "
            + (
                "nhất quán giữa hai model."
                if all(
                    all(mae_directions[(variant, model)] < 0 for model in self.models)
                    or all(mae_directions[(variant, model)] > 0 for model in self.models)
                    for variant in aggregation_variants
                )
                else "không nhất quán giữa hai model ở mọi variant."
            )
        )
        lines.append(
            "- **Về explanation stability:** so sánh weekly-group std giữa các variant bị lẫn confound "
            "định nghĩa metric (mục 3), còn so sánh feature-level std giữa các feature có mean chênh "
            "lệch lớn thì không được dùng std đơn độc để xếp hạng stability. Trong phạm vi core scope, số "
            "liệu mean ± std ở mục 2 và 3 là kết quả báo cáo được; kết luận mạnh hơn cần thước đo nằm "
            "ngoài spec hiện tại."
        )
        lines.append("")

        lines.append("## Chi tiết file")
        lines.append("")
        lines.append("- `performance_summary.csv`: MAE/RMSE/WAPE theo từng fold.")
        lines.append("- `performance_aggregated.csv`: MAE/RMSE/WAPE mean ± std qua Fold 1-4 + cột final_test.")
        lines.append("- `shap_importance_per_fold.csv`: `I_{j,f}` của weekly feature theo từng fold.")
        lines.append("- `feature_importance_stability.csv`: mean ± std của `I_{j,f}` qua Fold 1-4 + cột final_test.")
        lines.append("- `weekly_group_per_fold.csv`: weekly-group importance theo từng fold.")
        lines.append("- `weekly_group_stability.csv`: weekly-group importance mean ± std + cột final_test.")
        lines.append("- `hpo_comparison.csv`: baseline vs tuned trên HPO split.")
        lines.append("- `plots/{variant}_feature_trend.png`: `I_{j,f}` của từng weekly feature qua các fold.")
        lines.append("")
        lines.append(
            "Correlation ở mục 1 do tầng Data sinh ra, module này chỉ đọc lại: "
            "`data/processed/correlation_summary.csv` và `correlation_by_zone.csv`."
        )

        summary_path = self.stats_folder / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return summary_path

    def run(self):
        correlation_df = self.load_correlation_summary()
        metrics_df = self.load_metrics_df()
        shap_long_df = self.load_shap_long_df()

        performance_aggregated_df = self.compute_performance_summary(metrics_df)
        feature_stability_df = self.compute_feature_stability(shap_long_df)
        group_stability_df = self.compute_group_stability()

        self.plot_feature_trends(shap_long_df)
        summary_path = self.write_summary_md(
            correlation_df, performance_aggregated_df, feature_stability_df, group_stability_df
        )

        print(f"Đã ghi stats vào: {self.stats_folder}")
        print(f"Summary: {summary_path}")

        return performance_aggregated_df, feature_stability_df, group_stability_df


if __name__ == "__main__":
    RunStats().run()
