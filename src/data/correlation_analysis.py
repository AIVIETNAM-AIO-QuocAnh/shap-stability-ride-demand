"""Compute and validate weekly-lag correlation evidence."""

from itertools import combinations
from pathlib import Path
import pandas as pd

from src.configuration import DataConfig, load_data_config


def load_feature_table(path: Path) -> pd.DataFrame:
    """Read the generated feature table."""
    df = pd.read_csv(path, parse_dates=["target_datetime"])
    print(f"Loaded feature table: {len(df):,} rows, {df['pu_location_id'].nunique()} zones.")
    return df


def qa_check_missing(df: pd.DataFrame, expected_columns: tuple[str, ...]) -> None:
    print("\n=== QA: MISSING VALUES ===")
    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature table is missing required columns: {missing_cols}")

    na_counts = df[list(expected_columns)].isna().sum()
    na_counts = na_counts[na_counts > 0]

    if len(na_counts) > 0:
        print("WARNING: missing values detected:")
        print(na_counts.to_string())
        raise ValueError(
            "Feature table has missing values in required columns; check build_panel."
        )
    print("No missing values in required columns.")


def qa_check_duplicate(df: pd.DataFrame) -> None:
    print("\n=== QA: DUPLICATE (zone, hour) ===")
    n_dup = df.duplicated(subset=["pu_location_id", "target_datetime"]).sum()
    if n_dup > 0:
        raise ValueError(
            f"Found {n_dup} duplicate (zone, hour) rows in the feature table."
        )
    print("No duplicate (zone, hour) rows.")


def qa_check_zero_demand(df: pd.DataFrame) -> pd.Series:
    """Report the overall and per-zone zero-demand rates."""
    print("\n=== QA: ZERO-DEMAND RATE ===")
    overall_zero_rate = (df["demand"] == 0).mean()
    print(f"Overall zero-demand rate: {overall_zero_rate:.1%}")

    zone_zero_rate = df.groupby("pu_location_id")["demand"].apply(lambda s: (s == 0).mean())
    high_zero_zones = zone_zero_rate[zone_zero_rate > 0.95]

    if len(high_zero_zones) > 0:
        print(
            f"WARNING: {len(high_zero_zones)} zones have >95% zero-demand hours "
            f"-- correlation may be NaN:"
        )
        print(high_zero_zones.round(3).to_string())
    else:
        print("No zone has an unusually high zero-demand rate (>95%).")

    return zone_zero_rate


def compute_correlation_by_zone(
    df: pd.DataFrame,
    correlation_start: pd.Timestamp,
    correlation_end: pd.Timestamp,
    lag_pairs: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """Compute Pearson correlation per zone and weekly-lag pair."""
    print(f"\n=== CORRELATION: {correlation_start} -> {correlation_end} (exclusive) ===")

    period_df = df[
        (df["target_datetime"] >= correlation_start)
        & (df["target_datetime"] < correlation_end)
    ]
    print(f"Rows in the correlation period: {len(period_df):,}")

    records = []
    n_undefined = 0

    for zone, zone_df in period_df.groupby("pu_location_id"):
        for col_a, col_b in lag_pairs:
            r = zone_df[col_a].corr(zone_df[col_b], method="pearson")
            if pd.isna(r):
                n_undefined += 1
            records.append(
                {
                    "pu_location_id": zone,
                    "pair": f"{col_a}_vs_{col_b}",
                    "pearson_r": r,
                    "n_obs": len(zone_df),
                }
            )

    corr_df = pd.DataFrame(records)

    n_zones = period_df["pu_location_id"].nunique()
    print(f"Computed correlation for {n_zones} zones x {len(lag_pairs)} lag pairs.")
    if n_undefined > 0:
        print(
            f"WARNING: {n_undefined} (zone, pair) combinations have undefined correlation (NaN)."
        )

    return corr_df


def summarize_correlation(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean, sample standard deviation, and valid Pearson counts."""
    summary = (
        corr_df.groupby("pair")["pearson_r"]
        .agg(mean_r="mean", std_r="std", n_valid="count")
        .reset_index()
    )
    summary["n_zones_total"] = corr_df.groupby("pair")["pearson_r"].size().values
    summary["n_undefined"] = summary["n_zones_total"] - summary["n_valid"]

    print("\n=== SUMMARY: MEAN +- SD PEARSON CORRELATION OVER 50 ZONES ===")
    print(summary.round(4).to_string(index=False))

    return summary


def save_data_dictionary(path: Path) -> None:
    content = """# Data Dictionary - Feature và Correlation Artifacts

Mỗi row đại diện cho một pickup zone tại một target hour. Timestamps được chuẩn hóa theo giờ.

## `feature_table.csv`

| Cột | Kiểu | Ý nghĩa | Quy ước |
|---|---|---|---|
| `pu_location_id` | int | Mã pickup zone | Một trong 50 frozen zones; chưa được one-hot encode |
| `target_datetime` | datetime | Target hour `t` | Reference key, không phải model feature |
| `demand` | int | Số request tại zone ở thời điểm `t` | Target; không âm và có thể bằng 0 |
| `hour` | int (0-23) | Giờ lấy từ `target_datetime` | Base feature |
| `dayofweek` | int (0-6) | Thứ trong tuần, Monday = 0 | Base feature |
| `is_holiday` | int (0/1) | Cờ US federal holiday | Base feature |
| `lag_1` | int | Demand tại `t - 1 hour` | Base feature |
| `lag_24` | int | Demand tại `t - 24 hours` | Base feature dùng cho mọi variant |
| `lag_168` | int | Demand tại `t - 168 hours` | Weekly lag dùng cho A và C |
| `lag_336` | int | Demand tại `t - 336 hours` | Weekly lag dùng cho A |
| `lag_504` | int | Demand tại `t - 504 hours` | Weekly lag dùng cho A |
| `median_lag_3w` | float | Median của ba weekly lags | Aggregated weekly feature dùng cho B và C |

### Các ràng buộc đã đảm bảo

- Các required columns không có missing values.
- `(pu_location_id, target_datetime)` là unique key.
- Dữ liệu bắt đầu từ `2025-01-22`, sau `lag_504` warm-up period.
- Demand bằng 0 được giữ lại như một observation hợp lệ.

## `correlation_by_zone.csv`

Pearson correlation được tính cho từng frozen zone và từng weekly-lag pair trong khoảng
`[2025-01-22 00:00, 2025-08-01 00:00)`.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `pu_location_id` | int | Mã frozen pickup zone |
| `pair` | str | Một trong ba weekly-lag pairs |
| `pearson_r` | float | Pearson coefficient trong [-1, 1]; có thể là `NaN` nếu input không đổi |
| `n_obs` | int | Số hourly observations của zone |

## `correlation_summary.csv`

Mean và sample standard deviation theo từng pair trên 50 zones.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `pair` | str | Weekly-lag pair |
| `mean_r` | float | Mean của các zone-level coefficients hợp lệ |
| `std_r` | float | Sample standard deviation của các coefficients hợp lệ |
| `n_valid` | int | Số zones có coefficient xác định |
| `n_zones_total` | int | Tổng số zones được xét; kỳ vọng bằng 50 |
| `n_undefined` | int | Số zones có coefficient không xác định |

## Artifacts liên quan

- `variant_feature_map.json`: weekly-feature contract của A/B/C.
- `frozen/top50_zones_frozen.json`: frozen zone vocabulary và selection record.
- `lag_alignment_examples.csv`: 75 full-panel checks cho alignment `target_datetime - lag`.

## `lag_alignment_examples.csv`

File gồm ba frozen zones, năm target hours đại diện và năm lag values.
Mọi giá trị `matches` phải là `True`; source timestamps giúp audit mà không cần
rebuild pre-warm-up panel.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `pu_location_id` | int | Mã frozen pickup zone |
| `target_datetime` | datetime | Target hour đang được kiểm tra |
| `lag` | int | Độ dài lag tính theo giờ |
| `source_datetime` | datetime | Source hour kỳ vọng (`target_datetime - lag`) |
| `source_demand` | int | Full-panel demand tại source hour |
| `feature_value` | int | Giá trị `lag_<lag>` được tạo tại target hour |
| `matches` | bool | Exact match giữa source value và feature value |
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nSaved data dictionary -> {path}")


def run_analysis(config: DataConfig) -> pd.DataFrame:
    """Run correlation QA and write results to the requested directory."""
    paths = config["paths"]
    panel = config["panel"]
    correlation = config["correlation"]
    expected_columns = (
        "pu_location_id",
        "target_datetime",
        "demand",
        *panel["base_features"],
        *(f"lag_{lag}" for lag in panel["weekly_lags_hours"]),
        panel["median_feature"],
    )
    weekly_columns = tuple(f"lag_{lag}" for lag in panel["weekly_lags_hours"])
    lag_pairs = tuple(combinations(weekly_columns, 2))
    df = load_feature_table(paths["feature_table"])

    qa_check_missing(df, expected_columns)
    qa_check_duplicate(df)
    qa_check_zero_demand(df)

    corr_df = compute_correlation_by_zone(
        df, correlation["start"], correlation["end_exclusive"], lag_pairs
    )
    summary_df = summarize_correlation(corr_df)

    output_dir = paths["correlation_by_zone"].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    corr_by_zone_path = paths["correlation_by_zone"]
    corr_summary_path = paths["correlation_summary"]
    data_dictionary_path = paths["data_dictionary"]
    corr_df.to_csv(corr_by_zone_path, index=False)
    summary_df.to_csv(corr_summary_path, index=False)
    print(f"\nSaved per-zone correlation -> {corr_by_zone_path}")
    print(f"Saved correlation summary -> {corr_summary_path}")

    save_data_dictionary(data_dictionary_path)

    print("\n=== HANDOFF ===")
    print("feature_table.csv passed QA and is ready for handoff.")
    print(f"  - {data_dictionary_path.name}")
    print(f"  - {corr_by_zone_path.name}")
    print(f"  - {corr_summary_path.name}")
    return summary_df


def main() -> None:
    """Run the standard project correlation analysis."""
    run_analysis(load_data_config())


if __name__ == "__main__":
    main()
