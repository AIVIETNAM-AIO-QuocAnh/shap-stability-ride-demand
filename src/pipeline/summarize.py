"""Tạo summary mean/std đã được duyệt từ Pipeline artifact đã validate."""

from pathlib import Path

import pandas as pd

from src.load_config import load_data_config, load_model_config
from src.pipeline.artifacts import load_prediction_metrics, load_weekly_group_importance
from src.pipeline.qa_checks import FOLDS, MODELS, VARIANTS, validate_matrix


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Ghi một summary table mà không thay thế artifact hiện có."""
    if path.exists():
        raise FileExistsError(f"Từ chối ghi đè summary đã tồn tại: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _ensure_paths_absent(paths: tuple[Path, ...]) -> None:
    """Báo lỗi trước khi ghi summary nếu một target đã tồn tại."""
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Từ chối ghi đè các summary đã tồn tại: {existing}")


def _metrics_frame() -> pd.DataFrame:
    """Đọc 30 core metric artifact vào một bảng dài."""
    config = load_model_config()
    rows: list[dict[str, str | float]] = []
    for variant in VARIANTS:
        for fold in FOLDS:
            for model in MODELS:
                path = config["paths"]["results"] / variant / fold / model / "metrics.json"
                value = load_prediction_metrics(path)
                rows.append(
                    {
                        "variant": variant,
                        "model": model,
                        "fold": fold,
                        "mae": float(value["mae"]),
                        "rmse": float(value["rmse"]),
                        "wape": float(value["wape"]),
                    }
                )
    return pd.DataFrame(rows)


def _hpo_frame() -> pd.DataFrame:
    """Đọc baseline và tuned HPO metric artifact."""
    config = load_model_config()
    rows: list[dict[str, str | float]] = []
    for model in MODELS:
        for stage, directory_name in (("baseline", f"{model}_baseline"), ("tuned", model)):
            path = config["paths"]["results"] / "A" / "hpo" / directory_name / "metrics.json"
            value = load_prediction_metrics(path)
            rows.append(
                {
                    "model": model,
                    "stage": stage,
                    "mae": float(value["mae"]),
                    "rmse": float(value["rmse"]),
                    "wape": float(value["wape"]),
                }
            )
    return pd.DataFrame(rows)


def _performance_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp metric fold1-fold4 và giữ column final-test riêng."""
    cv = metrics[metrics["fold"] != "final_test"]
    aggregated = (
        cv.groupby(["variant", "model"], observed=True)[["mae", "rmse", "wape"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregated.columns
    ]
    final = metrics[metrics["fold"] == "final_test"][
        ["variant", "model", "mae", "rmse", "wape"]
    ].rename(
        columns={
            "mae": "mae_final_test",
            "rmse": "rmse_final_test",
            "wape": "wape_final_test",
        }
    )
    return aggregated.merge(final, on=["variant", "model"], validate="one_to_one")


def _shap_feature_frame() -> pd.DataFrame:
    """Đọc weekly-feature SHAP importance cho mọi core run."""
    data_config = load_data_config()
    model_config = load_model_config()
    rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        weekly_features = data_config["panel"]["variants"][variant]["weekly_features"]
        for fold in FOLDS:
            for model in MODELS:
                path = (
                    model_config["paths"]["results"]
                    / variant
                    / fold
                    / model
                    / "shap_importance.csv"
                )
                frame = pd.read_csv(path)
                expected = {"feature", "importance"}
                if set(frame.columns) != expected:
                    raise ValueError(f"SHAP importance artifact không hợp lệ: {path}")
                frame = frame[frame["feature"].isin(weekly_features)].copy()
                frame["variant"] = variant
                frame["model"] = model
                frame["fold"] = fold
                rows.append(frame[["variant", "model", "fold", "feature", "importance"]])
    return pd.concat(rows, ignore_index=True)


def _group_frame() -> pd.DataFrame:
    """Đọc weekly-group SHAP importance cho mọi core run."""
    config = load_model_config()
    rows: list[dict[str, str | float]] = []
    for variant in VARIANTS:
        for fold in FOLDS:
            for model in MODELS:
                path = config["paths"]["results"] / variant / fold / model / "shap_weekly_group.json"
                value = load_weekly_group_importance(path)
                rows.append(
                    {
                        "variant": variant,
                        "model": model,
                        "fold": fold,
                        "weekly_group_importance": value,
                    }
                )
    return pd.DataFrame(rows)


def _feature_stability(features: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp weekly-feature importance với sample standard deviation."""
    cv = features[features["fold"] != "final_test"]
    summary = (
        cv.groupby(["variant", "model", "feature"], observed=True)["importance"]
        .agg(mean_importance="mean", std_importance="std")
        .reset_index()
    )
    final = features[features["fold"] == "final_test"][
        ["variant", "model", "feature", "importance"]
    ].rename(columns={"importance": "importance_final_test"})
    return summary.merge(
        final,
        on=["variant", "model", "feature"],
        validate="one_to_one",
    )


def _group_stability(groups: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp weekly-group importance với sample standard deviation."""
    cv = groups[groups["fold"] != "final_test"]
    summary = (
        cv.groupby(["variant", "model"], observed=True)["weekly_group_importance"]
        .agg(mean_group_importance="mean", std_group_importance="std")
        .reset_index()
    )
    final = groups[groups["fold"] == "final_test"][
        ["variant", "model", "weekly_group_importance"]
    ].rename(columns={"weekly_group_importance": "group_importance_final_test"})
    return summary.merge(final, on=["variant", "model"], validate="one_to_one")


def summarize_core() -> Path:
    """Validate artifact và ghi các canonical Pipeline summary table."""
    validate_matrix()
    config = load_model_config()
    stats = config["paths"]["results_stats"]
    metrics = _metrics_frame()
    hpo = _hpo_frame()
    performance = _performance_summary(metrics)
    features = _shap_feature_frame()
    feature_stability = _feature_stability(features)
    groups = _group_frame()
    group_stability = _group_stability(groups)
    output_paths = (
        stats / "hpo_comparison.csv",
        stats / "performance_summary.csv",
        stats / "performance_aggregated.csv",
        stats / "shap_importance_per_fold.csv",
        stats / "feature_importance_stability.csv",
        stats / "weekly_group_per_fold.csv",
        stats / "weekly_group_stability.csv",
    )
    _ensure_paths_absent(output_paths)
    _write_csv(output_paths[0], hpo)
    _write_csv(output_paths[1], metrics)
    _write_csv(output_paths[2], performance)
    _write_csv(output_paths[3], features)
    _write_csv(output_paths[4], feature_stability)
    _write_csv(output_paths[5], groups)
    _write_csv(output_paths[6], group_stability)
    return stats
