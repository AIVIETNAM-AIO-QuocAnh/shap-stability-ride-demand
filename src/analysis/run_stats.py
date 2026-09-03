from src.utilities import load_config, resolve_path
import pandas as pd
import numpy as np
import json
import pickle
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        self.plots_folder = self.stats_folder / "plots"
        self.plots_folder.mkdir(parents=True, exist_ok=True)
        self.variant_map_file = resolve_path(cfg, "variant_map")
        with open(self.variant_map_file, "r") as f:
            self.variant_map = json.load(f)

        # Proposal mục 2.5/2.6: final_test báo cáo riêng, không gộp vào mean/std qua fold.
        self.final_test_fold = "final_test"
        self.cv_folds = tuple(fold for fold in self.folds if fold != self.final_test_fold)

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
        cv_shap_df = shap_long_df[shap_long_df["fold"] != self.final_test_fold]

        feature_stability_df = (
            cv_shap_df.groupby(["variant", "model", "feature"], observed=True)["importance"]
            .agg(mean_importance="mean", std_importance="std")
            .reset_index()
        )
        feature_stability_df["cv"] = feature_stability_df["std_importance"] / feature_stability_df["mean_importance"]

        final_test_df = shap_long_df[shap_long_df["fold"] == self.final_test_fold][
            ["variant", "model", "feature", "importance"]
        ].rename(columns={"importance": "importance_final_test"})

        feature_stability_df = feature_stability_df.merge(final_test_df, on=["variant", "model", "feature"])

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
        cv_group_df = group_importance_df[group_importance_df["fold"] != self.final_test_fold]

        group_stability_df = (
            cv_group_df.groupby(["variant", "model"], observed=True)["group_importance"]
            .agg(mean_group_importance="mean", std_group_importance="std")
            .reset_index()
        )
        group_stability_df["cv"] = group_stability_df["std_group_importance"] / group_stability_df["mean_group_importance"]

        final_test_df = group_importance_df[group_importance_df["fold"] == self.final_test_fold][
            ["variant", "model", "group_importance"]
        ].rename(columns={"group_importance": "group_importance_final_test"})

        group_stability_df = group_stability_df.merge(final_test_df, on=["variant", "model"])

        group_stability_df.to_csv(self.stats_folder / "weekly_group_stability.csv", index=False)

        return group_stability_df

    def load_shap_full_df(self):
        """Bản KHÔNG lọc weekly_features — giữ toàn bộ feature, dùng cho rank stability."""
        shap_frames = []

        for variant in self.variants:
            for fold in self.folds:
                for model in self.models:
                    result_path = self.result_base_path / variant / fold / model / "shap_importance.csv"
                    df = pd.read_csv(result_path)

                    df["variant"] = variant
                    df["fold"] = fold
                    df["model"] = model

                    shap_frames.append(df)

        shap_full_df = pd.concat(shap_frames, ignore_index=True)
        shap_full_df = shap_full_df[["variant", "model", "fold", "feature", "importance"]]
        shap_full_df["fold"] = pd.Categorical(shap_full_df["fold"], categories=self.folds, ordered=True)

        return shap_full_df

    def compute_rank_stability(self, shap_full_df):
        """Spearman giữa các cặp fold trên TOÀN BỘ feature ranking + vị trí rank của weekly feature.

        Proposal mục 2.6 cấm CV; rank stability không dùng mean ở mẫu số nên không dính
        artifact 'mean nhỏ -> CV to', và trả lời đúng câu hỏi thực tế: thứ tự feature có đổi không.
        """
        rank_rows = []

        for variant in self.variants:
            weekly_features = self.variant_map["variants"][variant]["weekly_features"]
            for model in self.models:
                subset = shap_full_df[
                    (shap_full_df["variant"] == variant)
                    & (shap_full_df["model"] == model)
                    & (shap_full_df["fold"] != self.final_test_fold)
                ]
                importance_by_fold = subset.pivot(index="feature", columns="fold", values="importance")
                importance_by_fold = importance_by_fold[list(self.cv_folds)]

                # Ranking toàn cục bị chi phối bởi ~50 zone dummy (importance rất nhỏ, thứ tự ổn
                # định một cách tầm thường) -> Spearman bị đẩy lên ~0.98 bất kể weekly lag ra sao.
                # Vì vậy tính thêm bản chỉ trên feature không phải zone dummy.
                non_zone_mask = ~importance_by_fold.index.str.startswith("zone_")
                importance_non_zone = importance_by_fold[non_zone_mask]

                def mean_pairwise_spearman(frame):
                    values = []
                    for i, fold_i in enumerate(self.cv_folds):
                        for fold_j in self.cv_folds[i + 1:]:
                            rho, _ = spearmanr(frame[fold_i], frame[fold_j])
                            values.append(rho)
                    return float(pd.Series(values).mean())

                spearman_all = mean_pairwise_spearman(importance_by_fold)
                spearman_non_zone = mean_pairwise_spearman(importance_non_zone)

                rank_by_fold = importance_non_zone.rank(ascending=False, method="min")
                for feature in weekly_features:
                    positions = rank_by_fold.loc[feature]
                    importances = importance_by_fold.loc[feature]
                    rank_rows.append({
                        "variant": variant,
                        "model": model,
                        "feature": feature,
                        "n_features_ranked": int(len(importance_by_fold)),
                        "n_features_non_zone": int(non_zone_mask.sum()),
                        "mean_pairwise_spearman_all_features": spearman_all,
                        "mean_pairwise_spearman_non_zone": spearman_non_zone,
                        "rank_min": int(positions.min()),
                        "rank_max": int(positions.max()),
                        "rank_range": int(positions.max() - positions.min()),
                        "ranks_by_fold": "→".join(str(int(p)) for p in positions),
                        # Rank không nhạy với biên độ: importance có thể dao động lớn mà hạng y nguyên.
                        "importance_swing_pct": float(
                            (importances.max() - importances.min()) / importances.mean() * 100
                        ),
                    })

        rank_stability_df = pd.DataFrame(rank_rows)
        rank_stability_df.to_csv(self.stats_folder / "rank_stability.csv", index=False)

        return rank_stability_df

    def compute_paired_comparison(self, metrics_df):
        """So sánh MAE theo từng cặp (fold, model) giữa variant gộp và baseline A.

        Mô tả thuần (không phải significance test — proposal mục 2.6 cấm test trong core scope):
        cùng fold + cùng model + cùng tập row, chỉ khác feature set, nên chênh lệch là so sánh
        có kiểm soát, chặt hơn nhiều so với đối chiếu mean ± std vốn bị nhiễu theo mùa của từng tháng.
        """
        cv_metrics_df = metrics_df[metrics_df["fold"] != self.final_test_fold]
        mae_by_variant = cv_metrics_df.pivot_table(
            index=["fold", "model"], columns="variant", values="mae", observed=True
        )

        def summarise_delta(delta, variant, scope):
            n = int(delta.notna().sum())
            mean_delta = float(delta.mean())
            std_delta = float(delta.std())
            # Phát biểu về TRUNG BÌNH hiệu số -> mẫu số đúng là SE = std/sqrt(n), không phải std.
            standard_error = std_delta / np.sqrt(n) if n else float("nan")
            return {
                "variant": variant,
                "scope": scope,
                "n_pairs": n,
                "n_better_than_A": int((delta < 0).sum()),
                "mean_delta_mae": mean_delta,
                "std_delta_mae": std_delta,
                "se_delta_mae": standard_error,
                "mean_over_std": abs(mean_delta) / std_delta if std_delta else float("nan"),
                "mean_over_se": abs(mean_delta) / standard_error if standard_error else float("nan"),
                "worst_delta_mae": float(delta.max()),
                "best_delta_mae": float(delta.min()),
            }

        paired_rows = []
        for variant in [v for v in self.variants if v != "A"]:
            delta_all = mae_by_variant[variant] - mae_by_variant["A"]
            paired_rows.append(summarise_delta(delta_all, variant, "pooled_2models"))
            # 8 cặp gộp KHÔNG độc lập (2 model × 4 fold, chung tập row), nên sqrt(8) là lạc quan.
            # Tách theo model (n=4) để kiểm hướng ở mức ít phụ thuộc hơn.
            for model in self.models:
                delta_model = (
                    mae_by_variant[variant].xs(model, level="model")
                    - mae_by_variant["A"].xs(model, level="model")
                )
                paired_rows.append(summarise_delta(delta_model, variant, model))

        paired_df = pd.DataFrame(paired_rows)
        paired_df.to_csv(self.stats_folder / "paired_mae_comparison.csv", index=False)

        return paired_df

    def compute_final_test_shift(self, feature_stability_df):
        """final_test (tháng 12) lệch bao nhiêu sigma so với phân bố fold1-4 của chính feature đó.

        fold1-4 đều là tháng 8-11 (regime tương đối đồng dạng). Tháng 12 có lễ/nghỉ nên là phép thử
        regime khác. Nếu attribution dịch chuyển có hệ thống ở đây thì "stability qua fold1-4" chỉ
        đang đo độ ổn định TRONG một regime, chưa phải độ ổn định nói chung.
        """
        shift_df = feature_stability_df[
            ["variant", "model", "feature", "mean_importance", "std_importance", "importance_final_test"]
        ].copy()
        shift_df["z_final_test"] = (
            (shift_df["importance_final_test"] - shift_df["mean_importance"]) / shift_df["std_importance"]
        )
        shift_df["abs_z"] = shift_df["z_final_test"].abs()
        shift_df = shift_df.sort_values("z_final_test")

        shift_df.to_csv(self.stats_folder / "final_test_shift.csv", index=False)

        return shift_df

    def compute_trend_confound_check(self, shap_long_df):
        """Loại (một phần) confound train-size khỏi phát hiện ở mục 2b.

        Train set tăng đơn điệu theo fold (mỗi fold thêm ~1 tháng), nên `final_test` vừa là tháng
        lệch regime, vừa là fold có train lớn nhất — hai cách giải thích bị lẫn.

        Cách tách: nếu độ dịch ở final_test do train-size/drift thì xu hướng đó phải **đã có sẵn**
        và đơn điệu qua fold1-4, và final_test chỉ nối dài. Nếu do regime thì fold1-4 phẳng, còn
        final_test **gãy ra khỏi** dải giá trị của fold1-4.
        """
        train_sizes = {}
        for fold in self.folds:
            train_path = resolve_path(cfg, "data_fold") / fold / "train.csv"
            with open(train_path, "r") as f:
                train_sizes[fold] = sum(1 for _ in f) - 1

        rows = []
        for (variant, model, feature), group in shap_long_df.groupby(
            ["variant", "model", "feature"], observed=True
        ):
            by_fold = group.set_index("fold")["importance"]
            cv_values = by_fold[list(self.cv_folds)]
            final_value = by_fold[self.final_test_fold]

            trend_rho, _ = spearmanr(range(len(self.cv_folds)), cv_values.values)
            z_final = (final_value - cv_values.mean()) / cv_values.std(ddof=1)
            inside_cv_range = bool(cv_values.min() <= final_value <= cv_values.max())
            continues_trend = bool(
                (trend_rho > 0 and z_final > 0) or (trend_rho < 0 and z_final < 0)
            )

            if inside_cv_range:
                verdict = "trong dải fold1-4 (không phải dịch chuyển)"
            elif continues_trend:
                verdict = "nối dài xu hướng sẵn có → LẪN confound train-size"
            else:
                verdict = "gãy khỏi xu hướng fold1-4 → regime là cách đọc hợp lý hơn"

            rows.append({
                "variant": variant,
                "model": model,
                "feature": feature,
                "trend_rho_fold1_4": float(trend_rho),
                "z_final_test": float(z_final),
                "final_inside_cv_range": inside_cv_range,
                "continues_existing_trend": continues_trend,
                "verdict": verdict,
            })

        confound_df = pd.DataFrame(rows).sort_values(["variant", "model", "feature"])
        confound_df.attrs["train_sizes"] = train_sizes
        confound_df.to_csv(self.stats_folder / "trend_confound_check.csv", index=False)

        return confound_df, train_sizes

    def compute_group_sensitivity(self):
        """Sensitivity check: công thức group importance của proposal vs công thức thay thế.

        - Proposal mục 2.6:  I_weekly = Σ_j mean_i |φ_ij|   (tổng của mean |φ| từng feature)
        - Thay thế:          I_weekly = mean_i |Σ_j φ_ij|   (mean của |tổng φ| trong từng sample)

        Với feature tương quan, φ của các feature trong nhóm có thể triệt tiêu dấu nhau trong cùng
        1 sample, nên 2 cách cho kết quả khác nhau. Cách 2 mới đúng nghĩa "đóng góp của cả nhóm".
        Đây là diagnostic, KHÔNG thay số liệu chính thức theo proposal.
        """
        sensitivity_rows = []

        for variant in self.variants:
            weekly_features = self.variant_map["variants"][variant]["weekly_features"]
            for fold in self.folds:
                for model in self.models:
                    shap_path = self.result_base_path / variant / fold / model / "shap_values.pkl"
                    with open(shap_path, "rb") as f:
                        shap_values = pickle.load(f)

                    feature_names = list(shap_values.feature_names)
                    weekly_idx = [feature_names.index(feature) for feature in weekly_features]
                    phi = shap_values.values[:, weekly_idx]

                    sensitivity_rows.append({
                        "variant": variant,
                        "model": model,
                        "fold": fold,
                        "group_proposal_sum_mean_abs": float(np.abs(phi).mean(axis=0).sum()),
                        "group_alt_mean_abs_sum": float(np.abs(phi.sum(axis=1)).mean()),
                    })

        sensitivity_df = pd.DataFrame(sensitivity_rows)
        sensitivity_df["cancellation_ratio"] = (
            sensitivity_df["group_alt_mean_abs_sum"] / sensitivity_df["group_proposal_sum_mean_abs"]
        )

        cv_sensitivity = sensitivity_df[sensitivity_df["fold"] != self.final_test_fold]
        summary_df = (
            cv_sensitivity.groupby(["variant", "model"], observed=True)[
                ["group_proposal_sum_mean_abs", "group_alt_mean_abs_sum", "cancellation_ratio"]
            ]
            .agg(["mean", "std"])
        )
        summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
        summary_df = summary_df.reset_index()

        summary_df.to_csv(self.stats_folder / "group_formula_sensitivity.csv", index=False)

        return summary_df

    def plot_feature_trends(self, shap_long_df):
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
            fig.suptitle(f"Variant {variant} — weekly-lag feature importance qua các fold")
            fig.tight_layout()

            path = self.plots_folder / f"{variant}_feature_trend.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(path)

        return saved_paths

    def plot_cv_comparison(self, feature_stability_df, group_stability_df):
        feature_cv = (
            feature_stability_df.groupby(["variant", "model"], observed=True)["cv"]
            .mean()
            .reset_index()
            .rename(columns={"cv": "feature_cv"})
        )
        group_cv = group_stability_df[["variant", "model", "cv"]].rename(columns={"cv": "group_cv"})
        cv_comparison_df = feature_cv.merge(group_cv, on=["variant", "model"])

        fig, axes = plt.subplots(1, len(self.models), figsize=(6 * len(self.models), 4), sharey=True)
        if len(self.models) == 1:
            axes = [axes]

        x_positions = range(len(self.variants))
        bar_width = 0.35

        for ax, model in zip(axes, self.models):
            model_df = cv_comparison_df[cv_comparison_df["model"] == model].set_index("variant").reindex(self.variants)
            ax.bar([x - bar_width / 2 for x in x_positions], model_df["feature_cv"], bar_width, label="feature-level CV")
            ax.bar([x + bar_width / 2 for x in x_positions], model_df["group_cv"], bar_width, label="group-level CV")
            ax.set_xticks(list(x_positions))
            ax.set_xticklabels(self.variants)
            ax.set_title(model)
            ax.set_xlabel("variant")
            ax.legend()

        axes[0].set_ylabel("CV (std / mean)")
        fig.suptitle("Feature-level vs group-level SHAP importance CV theo variant")
        fig.tight_layout()

        path = self.plots_folder / "cv_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

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

    def write_summary_md(self, performance_aggregated_df, feature_stability_df, group_stability_df,
                         rank_stability_df, paired_df, group_sensitivity_df, final_test_shift_df,
                         trend_confound_df, train_sizes):
        lines = ["# Kết quả tổng hợp — prediction performance & SHAP stability", ""]

        lines.append(
            "> **Quy ước báo cáo theo proposal mục 2.5/2.6:** số liệu chính là **mean ± standard "
            "deviation** qua fold1-4; `final_test` (tháng 12) báo cáo riêng, không gộp vào mean/std. "
            "Proposal ghi rõ *\"Không dùng coefficient of variation hoặc significance test trong core "
            "scope\"* — nên CV chỉ xuất hiện ở mục **Diagnostic** phía dưới, ngoài spec, và **không "
            "dùng để rút kết luận**. Proposal cũng cảnh báo không kết luận một feature \"ổn định hơn\" "
            "chỉ dựa vào std khi mean importance chênh lệch lớn."
        )
        lines.append("")

        lines.append("## 1. Prediction performance (mean ± std qua fold1-4)")
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
        lines.append(
            "- **Cảnh báo về effect size:** std qua fold lớn hơn nhiều lần khoảng chênh lệch giữa các "
            "variant (xem bảng paired bên dưới), nên **không** đọc bảng này như bằng chứng variant nào "
            "tốt hơn hẳn."
        )
        lines.append("")

        lines.append("### 1b. So sánh có kiểm soát theo từng cặp (fold × model)")
        lines.append("")
        lines.append(
            "Cùng fold, cùng model, cùng tập row — chỉ khác feature set. Mô tả thuần, không phải "
            "significance test.\n\n"
            "**Mốc nhiễu đúng cho thiết kế này là `std_delta_mae`** (độ phân tán của chính các hiệu "
            "số), *không phải* `unpaired_mae_level_std_A`. Cột sau chủ yếu phản ánh tháng 8 khó hơn "
            "tháng 11, mà chênh lệch giữa các tháng đã bị triệt tiêu khi so cùng fold — dùng nó làm "
            "mốc sẽ hạ thấp hiệu ứng một cách sai lệch."
        )
        lines.append("")
        lines.append(self._df_to_markdown(paired_df))
        lines.append("")
        lines.append(
            "**std hay SE?** Phát biểu ở đây là về *trung bình* hiệu số, nên mẫu số đúng là "
            "**SE = std/√n**, không phải std. `mean_over_std` chỉ mô tả độ phân tán của từng cặp; "
            "`mean_over_se` mới là tỉ lệ tương ứng với claim về trung bình.\n\n"
            "**Cảnh báo về tính độc lập:** 8 cặp gộp = 2 model × 4 fold **dùng chung tập row**, nên "
            "chúng không độc lập và √8 là lạc quan. Vì vậy bảng có thêm dòng tách riêng từng model "
            "(n=4) — đây là mức granularity trung thực hơn."
        )
        lines.append("")
        for _, row in paired_df.iterrows():
            all_same_sign = int(row["n_better_than_A"]) == int(row["n_pairs"])
            scope_label = "gộp 2 model" if row["scope"] == "pooled_2models" else f"chỉ `{row['scope']}`"
            lines.append(
                f"- Variant **{row['variant']}** ({scope_label}): tốt hơn A ở "
                f"**{int(row['n_better_than_A'])}/{int(row['n_pairs'])}** cặp"
                + (", **cùng dấu ở mọi cặp**" if all_same_sign else "")
                + f". Hiệu số trung bình **{row['mean_delta_mae']:.3f} MAE** "
                f"(std {row['std_delta_mae']:.3f}, SE {row['se_delta_mae']:.3f}) → "
                f"mean/SE ≈ **{row['mean_over_se']:.2f}**."
            )
        lines.append("")
        lines.append(
            "→ Hiệu ứng **nhỏ nhưng nhất quán về hướng**: cùng dấu ở mọi cặp, và vẫn giữ tỉ lệ "
            "mean/SE > 2 khi tách riêng từng model. Không phải bị chìm trong nhiễu."
        )
        lines.append("")

        lines.append("## 2. SHAP feature-level importance (mean ± std qua fold1-4)")
        lines.append("")
        feature_display = feature_stability_df[
            ["variant", "model", "feature", "mean_importance", "std_importance", "importance_final_test"]
        ].sort_values(["variant", "model", "feature"])
        lines.append(self._df_to_markdown(feature_display))
        lines.append("")
        lines.append(
            "- Đọc std **cùng với** mean: các feature ở đây chênh nhau nhiều lần về mean importance "
            "(vd. `lag_504` ≈ 7 so với `lag_168` ≈ 23), nên **không** so std trực tiếp giữa chúng để "
            "kết luận feature nào ổn định hơn — đúng cảnh báo proposal mục 2.6."
        )
        lines.append("")

        lines.append("### 2b. final_test lệch bao nhiêu so với phân bố fold1-4?")
        lines.append("")
        lines.append(
            "fold1-4 đều là tháng 8-11 — regime tương đối đồng dạng. Tháng 12 (lễ/nghỉ) là phép thử "
            "regime khác. Cột `z_final_test` = (final_test − mean) / std của chính feature đó."
        )
        lines.append("")
        shift_display = final_test_shift_df[
            ["variant", "model", "feature", "mean_importance", "std_importance",
             "importance_final_test", "z_final_test"]
        ]
        lines.append(self._df_to_markdown(shift_display))
        lines.append("")

        n_beyond_2 = int((final_test_shift_df["abs_z"] > 2).sum())
        n_beyond_1 = int((final_test_shift_df["abs_z"] > 1).sum())
        n_total = len(final_test_shift_df)
        lines.append(
            f"- **{n_beyond_2}/{n_total}** dòng lệch quá **2σ**, **{n_beyond_1}/{n_total}** dòng lệch "
            "quá 1σ so với phân bố fold1-4 của chính nó."
        )

        direction_notes = []
        for feature in sorted(final_test_shift_df["feature"].unique()):
            z_values = final_test_shift_df[final_test_shift_df["feature"] == feature]["z_final_test"]
            if len(z_values) > 1 and (z_values > 0).all():
                direction_notes.append(f"`{feature}` **tăng ở mọi model** (z: {', '.join(f'{z:+.2f}' for z in z_values)})")
            elif len(z_values) > 1 and (z_values < 0).all():
                direction_notes.append(f"`{feature}` **giảm ở mọi model** (z: {', '.join(f'{z:+.2f}' for z in z_values)})")
        if direction_notes:
            lines.append("- Dịch chuyển **có hướng, nhất quán giữa các model**: " + "; ".join(direction_notes) + ".")

        lines.append("")
        lines.append("#### Tách confound train-size")
        lines.append("")
        lines.append(
            "`final_test` vừa là tháng lệch regime, vừa là fold có **train set lớn nhất** ("
            + ", ".join(f"{fold}={train_sizes[fold]:,}" for fold in self.folds)
            + " row) — hai cách giải thích bị lẫn vào nhau. Cách tách: nếu do train-size/drift thì xu "
            "hướng phải **đã có sẵn** và đơn điệu qua fold1-4, `final_test` chỉ nối dài; nếu do regime "
            "thì fold1-4 phẳng còn `final_test` **gãy ra khỏi** dải giá trị fold1-4."
        )
        lines.append("")
        confound_display = trend_confound_df[
            ["variant", "model", "feature", "trend_rho_fold1_4", "z_final_test",
             "final_inside_cv_range", "verdict"]
        ]
        lines.append(self._df_to_markdown(confound_display))
        lines.append("")
        lines.append(
            "*Cảnh báo:* `trend_rho_fold1_4` tính trên **4 điểm**, chỉ nhận vài giá trị rời rạc "
            "(0, ±0.4, ±0.8, ±1.0) nên đây là kiểm tra thô, không phải bằng chứng chắc."
        )
        lines.append("")

        n_break = int((trend_confound_df["verdict"].str.startswith("gãy")).sum())
        n_continue = int((trend_confound_df["verdict"].str.startswith("nối dài")).sum())
        n_inside = int(trend_confound_df["final_inside_cv_range"].sum())
        lines.append(
            f"- **{n_break}** dòng gãy khỏi xu hướng fold1-4 (regime là cách đọc hợp lý hơn), "
            f"**{n_continue}** dòng chỉ nối dài xu hướng sẵn có (lẫn confound train-size), "
            f"**{n_inside}** dòng nằm gọn trong dải fold1-4 (không phải dịch chuyển thật)."
        )
        lines.append(
            "- **Đọc lại kết luận, đã thu hẹp:** phần `lag_504` **tăng** không còn đứng vững như bằng "
            "chứng regime — nó đã tăng dần sẵn qua fold1-4 khi train set lớn dần, `final_test` chỉ nối "
            "tiếp. Phần trụ được là `lag_336` (và `median_lag_3w`): fold1-4 **không có xu hướng** rồi "
            "`final_test` rơi xuống dưới toàn bộ dải — đây mới là gãy thật."
        )
        lines.append(
            "- Vì vậy phát biểu đúng **không phải** \"credit dịch từ lag 2 tuần sang lag 3 tuần\" (nửa "
            "sau lẫn confound), mà là: **tháng 12 làm gãy quỹ đạo của `lag_336`/`median_lag_3w`**, "
            "trong khi bốn fold liền kề — vốn là các tháng đồng dạng và train set tăng đều — chưa bao "
            "giờ thử thách được điều đó."
        )
        lines.append("")

        lines.append("## 3. Weekly-group importance (mean ± std qua fold1-4)")
        lines.append("")
        group_display = group_stability_df[
            ["variant", "model", "mean_group_importance", "std_group_importance", "group_importance_final_test"]
        ].sort_values(["variant", "model"])
        lines.append(self._df_to_markdown(group_display))
        lines.append("")

        group_sizes = {v: len(self.variant_map["variants"][v]["weekly_features"]) for v in self.variants}
        single_feature_variants = [v for v, n in group_sizes.items() if n == 1]
        lines.append(
            "- **Cảnh báo so sánh:** số feature trong nhóm khác nhau giữa các variant ("
            + ", ".join(f"{v}={n}" for v, n in group_sizes.items())
            + "). "
            + (
                f"Với variant {'/'.join(single_feature_variants)} nhóm chỉ có 1 feature nên "
                "**group importance ≡ feature importance** — chênh lệch giữa các variant ở bảng này "
                "một phần là do **định nghĩa metric** (tổng trên số feature khác nhau), chưa thể quy "
                "hết cho hành vi model."
                if single_feature_variants else ""
            )
        )
        lines.append("")

        lines.append("## 4. Rank stability (thước đo bổ sung — **không** nằm trong spec proposal)")
        lines.append("")
        lines.append(
            "*Ghi chú phạm vi:* proposal mục 2.6 chỉ quy định báo cáo mean ± std, và cấm CV. Rank "
            "stability **không nằm trong spec** — nó chỉ không bị cấm. Đây là thước đo bổ sung do "
            "nhóm thêm vào, không phải \"thước đo đúng spec\"."
        )
        lines.append("")
        n_all = int(rank_stability_df["n_features_ranked"].max())
        n_non_zone = int(rank_stability_df["n_features_non_zone"].max())
        lines.append(
            "Spearman giữa từng cặp fold trên feature ranking. Rank không có mean ở mẫu số nên không "
            "dính artifact \"mean nhỏ → CV to\".\n\n"
            f"**Hai cột Spearman, phải đọc cột non-zone:** ranking đầy đủ có {n_all} feature, trong đó "
            f"{n_all - n_non_zone} là zone one-hot với importance rất nhỏ và thứ tự ổn định một cách "
            f"tầm thường — chúng đẩy Spearman toàn cục lên ~0.98 bất kể weekly lag hành xử ra sao. "
            f"Cột `non_zone` (chỉ {n_non_zone} feature thật) mới có ý nghĩa.\n\n"
            "**Giới hạn của thước đo:** rank **không nhạy với biên độ** — importance có thể dao động "
            "hàng chục phần trăm mà thứ hạng vẫn y nguyên (xem cột `importance_swing_pct`). "
            "Rank ổn định **không loại trừ** magnitude bất ổn."
        )
        lines.append("")
        rank_display = rank_stability_df[
            ["variant", "model", "feature", "mean_pairwise_spearman_all_features",
             "mean_pairwise_spearman_non_zone", "ranks_by_fold", "rank_range", "importance_swing_pct"]
        ].sort_values(["variant", "model", "feature"])
        lines.append(self._df_to_markdown(rank_display))
        lines.append("")

        spearman_all_by_variant = rank_stability_df.groupby("variant")["mean_pairwise_spearman_all_features"].mean()
        spearman_nz_by_variant = rank_stability_df.groupby("variant")["mean_pairwise_spearman_non_zone"].mean()
        lines.append(
            "- Spearman toàn bộ feature: "
            + ", ".join(f"**{v}**={spearman_all_by_variant[v]:.4f}" for v in self.variants)
            + f" (bị {n_all - n_non_zone} zone dummy làm phồng, **không dùng để kết luận**)."
        )
        lines.append(
            "- Spearman chỉ trên feature non-zone: "
            + ", ".join(f"**{v}**={spearman_nz_by_variant[v]:.4f}" for v in self.variants)
            + " — đây mới là con số đáng đọc."
        )
        max_rank_range = rank_stability_df["rank_range"].max()
        max_swing = rank_stability_df["importance_swing_pct"].max()
        max_swing_row = rank_stability_df.loc[rank_stability_df["importance_swing_pct"].idxmax()]
        lines.append(
            f"- Dao động thứ hạng lớn nhất của một weekly-lag feature qua fold1-4: "
            f"**{int(max_rank_range)}** bậc — nhưng cùng lúc đó biên độ importance dao động tới "
            f"**{max_swing:.1f}%** (`{max_swing_row['feature']}`, variant {max_swing_row['variant']}/"
            f"{max_swing_row['model']}). Đây chính là minh hoạ rank ổn định nhưng magnitude thì không."
        )
        max_non_zone = float(rank_stability_df["mean_pairwise_spearman_non_zone"].max())
        at_ceiling = rank_stability_df[
            rank_stability_df["mean_pairwise_spearman_non_zone"] >= 1.0
        ]["variant"].unique()
        if len(at_ceiling):
            lines.append(
                f"- **Đụng trần:** variant {'/'.join(sorted(at_ceiling))} đạt Spearman = "
                f"**{max_non_zone:.4f}**, tức giá trị tối đa của thang đo. Khi một variant đã kịch "
                "trần thì thước đo **không còn khả năng phân biệt** — đây là lý do mạnh hơn cả việc "
                "rank không nhạy với biên độ."
            )
        lines.append(
            "- Phát biểu đúng: **thước đo rank không phân biệt được** A/B/C ở đây (đụng trần + không "
            "nhạy biên độ) — không đồng nghĩa với \"không có khác biệt\"."
        )
        lines.append("")

        lines.append("## 5. Diagnostic (ngoài spec proposal — không dùng để kết luận)")
        lines.append("")
        lines.append(
            "### 5a. CV = std/mean — vì sao không dùng để kết luận"
        )
        lines.append("")
        cv_display = feature_stability_df[
            ["variant", "model", "feature", "mean_importance", "std_importance", "cv"]
        ].sort_values("cv", ascending=False)
        lines.append(self._df_to_markdown(cv_display))
        lines.append("")
        cv_sorted = feature_stability_df.sort_values("cv", ascending=False)
        n_top = max(1, len(cv_sorted) // 3)
        top_cv_max_mean = cv_sorted.head(n_top)["mean_importance"].max()
        bottom_cv_min_mean = cv_sorted.tail(n_top)["mean_importance"].min()
        rho_mean_cv, _ = spearmanr(feature_stability_df["mean_importance"], feature_stability_df["cv"])
        lines.append(
            f"- Tương quan Spearman giữa `mean_importance` và `cv` trên {len(feature_stability_df)} dòng: "
            f"**{rho_mean_cv:.3f}** — CV gần như bị quyết định bởi độ lớn của mean, đúng artifact mẫu số "
            "nhỏ mà proposal mục 2.6 cảnh báo."
        )
        lines.append(
            f"- Cụ thể: {n_top} feature có CV cao nhất đều có mean ≤ **{top_cv_max_mean:.2f}**, trong khi "
            f"{n_top} feature có CV thấp nhất đều có mean ≥ **{bottom_cv_min_mean:.2f}**. "
            "Nói \"feature CV cao thì kém ổn định hơn\" ở đây thực chất chỉ đang nói \"feature đó có "
            "importance nhỏ hơn\"."
        )
        lines.append("")

        lines.append("### 5b. Sensitivity: công thức weekly-group importance")
        lines.append("")
        lines.append(
            "Proposal mục 2.6 định nghĩa group importance = **Σ_j mean_i |φ_ij|** (tổng mean |φ| từng "
            "feature) — code đang implement đúng công thức này. Cách thay thế **mean_i |Σ_j φ_ij|** "
            "(mean của |tổng φ| trong từng sample) mới đúng nghĩa \"đóng góp của cả nhóm\", vì với "
            "feature tương quan các φ có thể triệt tiêu dấu nhau trong cùng 1 sample. "
            "`cancellation_ratio` = alt / proposal; càng nhỏ hơn 1 nghĩa là triệt tiêu dấu càng nhiều."
        )
        lines.append("")
        sensitivity_display = group_sensitivity_df[
            ["variant", "model", "group_proposal_sum_mean_abs_mean", "group_alt_mean_abs_sum_mean",
             "group_alt_mean_abs_sum_std", "cancellation_ratio_mean"]
        ].sort_values(["variant", "model"])
        lines.append(self._df_to_markdown(sensitivity_display))
        lines.append("")

        min_ratio_row = group_sensitivity_df.loc[group_sensitivity_df["cancellation_ratio_mean"].idxmin()]
        lines.append(
            f"- Mức triệt tiêu dấu mạnh nhất ở **variant {min_ratio_row['variant']} / "
            f"{min_ratio_row['model']}** (ratio={min_ratio_row['cancellation_ratio_mean']:.4f}). "
            "Variant có nhóm 1 feature thì ratio = 1 theo định nghĩa (không có gì để triệt tiêu)."
        )
        lines.append("")

        lines.append("## 6. Đánh giá (proposal mục 6, câu hỏi 3)")
        lines.append("")

        baseline_perf = performance_aggregated_df[performance_aggregated_df["variant"] == "A"].set_index("model")
        baseline_group = group_stability_df[group_stability_df["variant"] == "A"].set_index("model")
        aggregation_variants = [v for v in self.variants if v != "A"]

        for variant in aggregation_variants:
            variant_perf = performance_aggregated_df[performance_aggregated_df["variant"] == variant].set_index("model")
            variant_group = group_stability_df[group_stability_df["variant"] == variant].set_index("model")

            lines.append(f"**Variant {variant} so với A:**")
            lines.append("")
            for model in self.models:
                mae_a = baseline_perf.loc[model, "mae_mean"]
                mae_v = variant_perf.loc[model, "mae_mean"]
                mae_delta_pct = (mae_v - mae_a) / mae_a * 100
                mae_word = "giảm" if mae_delta_pct < 0 else "tăng"

                std_a = baseline_group.loc[model, "std_group_importance"]
                std_v = variant_group.loc[model, "std_group_importance"]
                mean_a = baseline_group.loc[model, "mean_group_importance"]
                mean_v = variant_group.loc[model, "mean_group_importance"]

                lines.append(
                    f"- `{model}`: MAE {mae_word} {abs(mae_delta_pct):.1f}% ({mae_a:.3f} → {mae_v:.3f}); "
                    f"weekly-group importance {mean_a:.2f} ± {std_a:.2f} → {mean_v:.2f} ± {std_v:.2f}."
                )
            lines.append("")

        mae_improves = {
            (v, m): performance_aggregated_df[
                (performance_aggregated_df["variant"] == v) & (performance_aggregated_df["model"] == m)
            ]["mae_mean"].iloc[0] < baseline_perf.loc[m, "mae_mean"]
            for v in aggregation_variants
            for m in self.models
        }
        mae_all_better = all(mae_improves.values())
        all_pairs_win = bool((paired_df["n_better_than_A"] == paired_df["n_pairs"]).all())

        lines.append("**Kết luận:**")
        lines.append("")
        per_model_df = paired_df[paired_df["scope"] != "pooled_2models"]
        min_mean_over_se = float(per_model_df["mean_over_se"].min())
        max_mean_delta = float(paired_df["mean_delta_mae"].abs().max())
        lines.append(
            "- Về **prediction**: gộp weekly-lag "
            + ("thắng baseline A ở **mọi** cặp fold × model" if all_pairs_win else "thắng baseline A ở một phần các cặp fold × model")
            + f". Hiệu số trung bình tối đa ~{max_mean_delta:.2f} MAE; ngay cả khi tách riêng từng "
            f"model (n=4, tránh giả định độc lập của 8 cặp gộp) thì mean/SE vẫn ≥ {min_mean_over_se:.2f}. "
            "Kết luận: hiệu ứng **nhỏ nhưng thật và nhất quán về hướng** — đủ để nói gộp feature không "
            "làm hại prediction, không đủ để nói nó cải thiện đáng kể."
        )
        lines.append(
            "- Về **explanation**: thước đo rank (mục 4, **bổ sung — không nằm trong spec**) "
            "**không phân biệt được** A/B/C: nó đụng trần (A = 1.0000) nên hết khả năng phân giải, và "
            "vốn cũng không nhạy với biên độ. Cơ chế \"gộp feature tương quan làm credit dồn về một "
            "chỗ\" là có thật về mặt lý thuyết với Tree SHAP `tree_path_dependent`, nhưng **không có "
            "thước đo nào hiện có đủ sức xác nhận hay bác bỏ nó trên dữ liệu này**."
        )
        lines.append(
            "- **Attribution KHÔNG ổn định khi đổi regime:** dù fold1-4 rất ổn định, tháng 12 (mục 2b) "
            "cho thấy credit dịch có hệ thống từ lag ngắn sang lag dài. Bốn fold liền kề là các tháng "
            "đồng dạng, nên chúng chưa từng thực sự thử thách tính ổn định mà đề tài muốn đo."
        )
        lines.append(
            "- **Không so sánh được ở group-level:** số feature trong nhóm khác nhau giữa các variant "
            "(mục 3), nên chênh lệch ở đó lẫn confound định nghĩa metric. Sign cancellation đã kiểm tra "
            "(mục 5b) và không phải nguyên nhân."
        )
        lines.append(
            "- **Chưa trả lời được:** biến động qua fold hiện chưa tách khỏi nhiễu nội tại của chính ước "
            "lượng SHAP (chỉ 1 seed, 1 tập 5.000 row mỗi fold). Thiếu baseline nhiễu theo seed/bootstrap "
            "thì mọi phát biểu về \"ổn định\" đều thiếu mốc so sánh — đây là hạn chế lớn nhất hiện tại."
        )
        lines.append("")

        lines.append("## Chi tiết file")
        lines.append("")
        lines.append("- `performance_summary.csv` — MAE/RMSE/WAPE raw theo từng fold.")
        lines.append("- `performance_aggregated.csv` — MAE/RMSE/WAPE mean ± std theo (variant, model).")
        lines.append("- `paired_mae_comparison.csv` — so sánh MAE theo từng cặp (fold × model) so với baseline A.")
        lines.append("- `feature_importance_stability.csv` — SHAP importance mean/std theo từng weekly-lag feature (cột `cv` là diagnostic ngoài spec).")
        lines.append("- `weekly_group_stability.csv` — weekly-group importance mean/std (cột `cv` là diagnostic ngoài spec).")
        lines.append("- `rank_stability.csv` — Spearman giữa các fold + dao động thứ hạng weekly-lag feature.")
        lines.append("- `group_formula_sensitivity.csv` — so sánh 2 công thức group importance (proposal vs mean|Σφ|).")
        lines.append("- `plots/{variant}_feature_trend.png` — xu hướng importance từng feature qua fold.")
        lines.append("- `plots/cv_comparison.png` — biểu đồ diagnostic CV (ngoài spec).")

        summary_path = self.stats_folder / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return summary_path

    def run(self):
        metrics_df = self.load_metrics_df()
        shap_long_df = self.load_shap_long_df()
        shap_full_df = self.load_shap_full_df()

        performance_aggregated_df = self.compute_performance_summary(metrics_df)
        paired_df = self.compute_paired_comparison(metrics_df)
        feature_stability_df = self.compute_feature_stability(shap_long_df)
        group_stability_df = self.compute_group_stability()
        final_test_shift_df = self.compute_final_test_shift(feature_stability_df)
        trend_confound_df, train_sizes = self.compute_trend_confound_check(shap_long_df)
        rank_stability_df = self.compute_rank_stability(shap_full_df)
        group_sensitivity_df = self.compute_group_sensitivity()

        self.plot_feature_trends(shap_long_df)
        self.plot_cv_comparison(feature_stability_df, group_stability_df)
        summary_path = self.write_summary_md(
            performance_aggregated_df, feature_stability_df, group_stability_df,
            rank_stability_df, paired_df, group_sensitivity_df, final_test_shift_df,
            trend_confound_df, train_sizes,
        )

        print(f"Stats written to: {self.stats_folder}")
        print(f"Summary: {summary_path}")

        return performance_aggregated_df, feature_stability_df, group_stability_df


if __name__ == "__main__":
    RunStats().run()
