"""Trang Experiment: so sánh kết quả thí nghiệm, bám đúng cấu trúc README gốc.

Chỉ đọc lại artifact trong `results/stats/` và `data/processed/` do pipeline sinh ra,
không tính lại chỉ số nào, nên số trên trang này luôn khớp báo cáo.

Chỉ trình bày các chỉ số **proposal quy định** (mục 2.2, 2.5, 2.6, 3.2): correlation,
MAE/RMSE/WAPE mean ± std, SHAP importance mean ± std, weekly-group importance,
baseline vs tuned. Không có coefficient of variation, rank stability hay significance
test, đúng như phần Phạm vi báo cáo của README.

Chuỗi hiển thị bằng tiếng Anh; comment và docstring giữ tiếng Việt.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_access import (
    load_correlation_summary,
    load_stats_table,
    stats_plot_path,
)


VARIANT_ROLES = {
    "A": "Baseline, three weekly lags kept separate",
    "B": "Aggregated into their median",
    "C": "Hybrid, nearest lag plus median",
}
FORECAST_COLOR = "#d95f02"
ACTUAL_COLOR = "#1b9e77"
VARIANT_COLORS = {"A": "#4c78a8", "B": "#f58518", "C": "#54a24b"}


def _section_correlation() -> None:
    """Câu hỏi 1: ba weekly lag tương quan tới đâu và có đồng nhất giữa các zone không."""
    st.subheader("1. Correlation between the three weekly lags")
    st.caption(
        "Pearson correlation computed per zone across the 50 frozen zones "
        "(proposal section 2.2). Source: `data/processed/correlation_summary.csv`."
    )
    correlation = load_correlation_summary().copy()
    correlation["pair"] = correlation["pair"].str.replace("_vs_", " x ", regex=False)

    table_column, chart_column = st.columns([1.1, 1.0], gap="medium")
    with table_column:
        display = correlation[["pair", "mean_r", "std_r", "n_valid"]].rename(
            columns={"mean_r": "mean r", "std_r": "std r", "n_valid": "zones"}
        )
        st.dataframe(
            display, hide_index=True, width="stretch",
            column_config={
                "mean r": st.column_config.ProgressColumn(format="%.4f", min_value=0.0, max_value=1.0),
                "std r": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        lowest, highest = correlation["mean_r"].min(), correlation["mean_r"].max()
        st.markdown(
            f"All three pairs sit between **{lowest:.2f}** and **{highest:.2f}** with a standard "
            f"deviation of only **{correlation['std_r'].max():.3f}** across 50 zones. The overlap "
            "is strong and uniform, so the premise of the study holds."
        )
    with chart_column:
        figure = px.bar(
            correlation, x="mean_r", y="pair", orientation="h",
            error_x="std_r", range_x=[0.0, 1.0],
            labels={"mean_r": "mean Pearson r", "pair": ""},
        )
        figure.update_traces(marker_color="#4c78a8")
        figure.update_layout(margin={"r": 8, "t": 6, "l": 0, "b": 0}, height=190)
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _section_feature_importance() -> None:
    """Câu hỏi 2: mean ± std của SHAP importance qua Fold 1-4 ở A/B/C."""
    st.subheader("2. SHAP feature importance, mean ± std over folds 1 to 4")
    st.caption(
        "`I(j,f)` is the mean absolute SHAP value of feature j on the 5,000 sampled rows of "
        "fold f (proposal section 2.6). December `final_test` is reported separately and is "
        "never folded into the mean or the standard deviation."
    )
    stability = load_stats_table("feature_importance_stability")

    chart_column, table_column = st.columns([1.35, 1.0], gap="medium")
    with chart_column:
        stability = stability.assign(
            label=stability["variant"] + " / " + stability["model"] + " · " + stability["feature"]
        ).sort_values(["variant", "model", "mean_importance"], ascending=[True, True, False])
        figure = go.Figure()
        for variant, group in stability.groupby("variant"):
            figure.add_bar(
                x=group["mean_importance"], y=group["label"], orientation="h",
                name=f"Variant {variant}", marker_color=VARIANT_COLORS[variant],
                error_x={"type": "data", "array": group["std_importance"], "thickness": 1.2},
                customdata=group[["importance_final_test"]],
                hovertemplate=(
                    "%{y}<br>mean ± std: %{x:.2f} ± %{error_x.array:.2f}"
                    "<br>final_test: %{customdata[0]:.2f}<extra></extra>"
                ),
            )
        figure.update_layout(
            margin={"r": 8, "t": 6, "l": 0, "b": 0}, height=430,
            xaxis={"title": {"text": "mean |SHAP|", "font": {"size": 11}}},
            yaxis={"title": "", "autorange": "reversed"},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
            bargap=0.25,
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with table_column:
        display = stability[
            ["variant", "model", "feature", "mean_importance", "std_importance", "importance_final_test"]
        ].rename(
            columns={"mean_importance": "mean", "std_importance": "std",
                     "importance_final_test": "final_test"}
        )
        st.dataframe(
            display, hide_index=True, width="stretch", height=430,
            column_config={
                "mean": st.column_config.NumberColumn(format="%.2f"),
                "std": st.column_config.NumberColumn(format="%.2f"),
                "final_test": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    lowest = stability.loc[stability["mean_importance"].idxmin()]
    highest = stability.loc[stability["mean_importance"].idxmax()]
    ratio = highest["mean_importance"] / lowest["mean_importance"]
    st.info(
        f"**Read the standard deviation together with the mean.** Mean importance spans "
        f"**{lowest['mean_importance']:.2f}** (`{lowest['feature']}` in "
        f"{lowest['variant']}/{lowest['model']}) to **{highest['mean_importance']:.2f}** "
        f"(`{highest['feature']}` in {highest['variant']}/{highest['model']}), a factor of "
        f"**{ratio:.1f}**. Proposal section 2.6 warns against ranking stability by standard "
        "deviation alone when the means differ this much, so no variant is declared more stable "
        "at the feature level."
    )

    with st.expander("Per-fold trend for each variant"):
        plot_columns = st.columns(3)
        for column, variant in zip(plot_columns, ("A", "B", "C")):
            path = stats_plot_path(f"{variant}_feature_trend.png")
            if path.is_file():
                column.image(str(path), caption=f"Variant {variant}", width="stretch")
            else:
                column.warning(f"Missing {path.name}")


def _section_weekly_group() -> None:
    """Weekly-group importance: tổng I(j,f) của các weekly feature trong variant."""
    st.subheader("3. Weekly-group importance, mean ± std over folds 1 to 4")
    st.caption(
        "`I(weekly,f)` sums `I(j,f)` over the weekly features of the variant: "
        "A = lag_168 + lag_336 + lag_504, B = median_lag_3w, C = lag_168 + median_lag_3w."
    )
    group = load_stats_table("weekly_group_stability")

    chart_column, table_column = st.columns([1.35, 1.0], gap="medium")
    with chart_column:
        group = group.assign(label=group["variant"] + " / " + group["model"])
        figure = go.Figure()
        for variant, rows in group.groupby("variant"):
            figure.add_bar(
                x=rows["label"], y=rows["mean_group_importance"],
                name=f"Variant {variant}", marker_color=VARIANT_COLORS[variant],
                error_y={"type": "data", "array": rows["std_group_importance"], "thickness": 1.2},
            )
        figure.update_layout(
            margin={"r": 8, "t": 6, "l": 0, "b": 0}, height=250,
            yaxis={"title": {"text": "group importance", "font": {"size": 11}}},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with table_column:
        display = group[
            ["variant", "model", "mean_group_importance", "std_group_importance",
             "group_importance_final_test"]
        ].rename(columns={"mean_group_importance": "mean", "std_group_importance": "std",
                          "group_importance_final_test": "final_test"})
        st.dataframe(
            display, hide_index=True, width="stretch", height=250,
            column_config={
                "mean": st.column_config.NumberColumn(format="%.2f"),
                "std": st.column_config.NumberColumn(format="%.2f"),
                "final_test": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    baseline = group[group["variant"] == "A"].set_index("model")
    lines = []
    for variant in ("B", "C"):
        rows = group[group["variant"] == variant].set_index("model")
        higher = [
            model for model in rows.index
            if rows.loc[model, "std_group_importance"] > baseline.loc[model, "std_group_importance"]
        ]
        if len(higher) == len(rows):
            lines.append(f"variant **{variant}** has a *higher* standard deviation than A on **both** models")
        elif higher:
            lines.append(f"variant **{variant}** is higher than A on {', '.join(higher)} only, so the direction is **not consistent** across models")
        else:
            lines.append(f"variant **{variant}** is lower than A on both models")
    st.warning(
        "**Group sizes differ across variants (A=3, B=1, C=2 features), so this table cannot be "
        "compared directly between variants.** For variant B the group contains a single feature, "
        "so group importance is identical to feature importance. Within that caveat: "
        + "; ".join(lines) + ". Aggregating the correlated lags does not make the explanation "
        "more stable here."
    )


def _section_performance() -> None:
    """Câu hỏi 3: gộp feature ảnh hưởng thế nào tới MAE, RMSE, WAPE."""
    st.subheader("4. Prediction performance, mean ± std over folds 1 to 4")
    st.caption(
        "MAE is the primary metric; RMSE and WAPE are supporting metrics and WAPE is a "
        "percentage (proposal section 2.5). December `final_test` is reported separately."
    )
    performance = load_stats_table("performance_aggregated")
    metric = st.segmented_control(
        "Metric", ("mae", "rmse", "wape"), default="mae", selection_mode="single"
    ) or "mae"

    chart_column, table_column = st.columns([1.35, 1.0], gap="medium")
    with chart_column:
        frame = performance.assign(label=performance["variant"] + " / " + performance["model"])
        figure = go.Figure()
        for variant, rows in frame.groupby("variant"):
            figure.add_bar(
                x=rows["label"], y=rows[f"{metric}_mean"],
                name=f"Variant {variant}", marker_color=VARIANT_COLORS[variant],
                error_y={"type": "data", "array": rows[f"{metric}_std"], "thickness": 1.2},
                customdata=rows[[f"{metric}_final_test"]],
                hovertemplate="%{x}<br>mean: %{y:.3f}<br>final_test: %{customdata[0]:.3f}<extra></extra>",
            )
        figure.add_scatter(
            x=frame["label"], y=frame[f"{metric}_final_test"], mode="markers",
            name="final_test", marker={"symbol": "diamond", "size": 9, "color": "#333"},
        )
        figure.update_layout(
            margin={"r": 8, "t": 6, "l": 0, "b": 0}, height=270,
            yaxis={"title": {"text": metric.upper(), "font": {"size": 11}}},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with table_column:
        display = performance[
            ["variant", "model", f"{metric}_mean", f"{metric}_std", f"{metric}_final_test"]
        ].rename(columns={f"{metric}_mean": "mean", f"{metric}_std": "std",
                          f"{metric}_final_test": "final_test"})
        st.dataframe(
            display, hide_index=True, width="stretch", height=270,
            column_config={
                "mean": st.column_config.NumberColumn(format="%.3f"),
                "std": st.column_config.NumberColumn(format="%.3f"),
                "final_test": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    st.markdown("**Change against baseline A**")
    baseline = performance[performance["variant"] == "A"].set_index("model")
    rows = []
    for variant in ("B", "C"):
        subset = performance[performance["variant"] == variant].set_index("model")
        for model in subset.index:
            row = {"variant": variant, "model": model}
            for name in ("mae", "rmse", "wape"):
                base_value = baseline.loc[model, f"{name}_mean"]
                row[name.upper()] = (subset.loc[model, f"{name}_mean"] - base_value) / base_value * 100
            rows.append(row)
    delta = pd.DataFrame(rows)
    styled = delta.style.map(
        lambda value: f"color:{'#2b8a3e' if value < 0 else '#c92a2a'};font-weight:600",
        subset=["MAE", "RMSE", "WAPE"],
    )
    st.dataframe(
        styled, hide_index=True, width="stretch",
        column_config={name: st.column_config.NumberColumn(format="%+.2f%%")
                       for name in ("MAE", "RMSE", "WAPE")},
    )

    spread = performance["mae_mean"].max() - performance["mae_mean"].min()
    largest_std = performance["mae_std"].max()
    best = performance.loc[performance["mae_mean"].idxmin()]
    st.info(
        f"Lowest mean MAE: **variant {best['variant']} / {best['model']}** "
        f"({best['mae_mean']:.3f} ± {best['mae_std']:.3f}). Both aggregated variants lower the "
        "mean MAE on **both** models, so the direction is consistent. But the spread across all "
        f"six combinations is only **{spread:.3f}** while the fold-to-fold standard deviation "
        f"reaches **{largest_std:.3f}**, which mostly reflects how much harder some evaluation "
        "months are. Significance tests are outside the core scope, so the honest reading stops "
        "at description: aggregating the weekly lags does not hurt prediction, and the gain is "
        "too small to call one variant better."
    )


def _section_hpo() -> None:
    """Tác động của tuning trên HPO split (proposal mục 3.2)."""
    st.subheader("5. Baseline versus tuned on the HPO split")
    st.caption(
        "Twenty Optuna trials per model on variant A over July 2025, then the best "
        "configuration is frozen and reused unchanged for every variant and fold."
    )
    hpo = load_stats_table("hpo_comparison")
    pivot = hpo.pivot(index="model", columns="stage", values="mae").reset_index()
    pivot["change %"] = (pivot["tuned"] - pivot["baseline"]) / pivot["baseline"] * 100

    table_column, chart_column = st.columns([1.0, 1.0], gap="medium")
    with table_column:
        styled = pivot.style.map(
            lambda value: f"color:{'#2b8a3e' if value < 0 else '#c92a2a'};font-weight:600",
            subset=["change %"],
        )
        st.dataframe(
            styled, hide_index=True, width="stretch",
            column_config={
                "baseline": st.column_config.NumberColumn("baseline MAE", format="%.3f"),
                "tuned": st.column_config.NumberColumn("tuned MAE", format="%.3f"),
                "change %": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )
    with chart_column:
        long_frame = hpo[hpo["stage"].isin(["baseline", "tuned"])]
        figure = px.bar(
            long_frame, x="model", y="mae", color="stage", barmode="group",
            color_discrete_map={"baseline": "#adb5bd", "tuned": FORECAST_COLOR},
            labels={"mae": "MAE on July 2025", "model": "", "stage": ""},
        )
        figure.update_layout(
            margin={"r": 8, "t": 6, "l": 0, "b": 0}, height=210,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def render() -> None:
    """Dựng trang Experiment. Trang này là báo cáo nên cho phép cuộn."""
    st.title("Experiment comparison")
    st.caption(
        "Only the metrics the proposal specifies: correlation (2.2), MAE/RMSE/WAPE mean ± std "
        "(2.5), SHAP importance and weekly-group importance mean ± std (2.6), and baseline "
        "versus tuned (3.2). No coefficient of variation, rank stability or significance test, "
        "since section 2.6 rules them out of the core scope. Every number is read back from "
        "`results/stats/`, nothing is recomputed here."
    )
    roles = " · ".join(f"**{name}** {role}" for name, role in VARIANT_ROLES.items())
    st.markdown(roles)
    st.divider()

    _section_correlation()
    st.divider()
    _section_feature_importance()
    st.divider()
    _section_weekly_group()
    st.divider()
    _section_performance()
    st.divider()
    _section_hpo()
