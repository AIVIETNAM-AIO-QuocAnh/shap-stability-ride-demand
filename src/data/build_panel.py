"""Xây dựng dense hourly panel và feature contract A/B/C dùng chung."""

import json
from pathlib import Path
from typing import TypedDict
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.load_config import DataConfig, VariantConfig, load_data_config


def load_frozen_zones(path: Path) -> list[int]:
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    zone_ids = record["zone_ids"]
    print(f"Đã đọc {len(zone_ids)} frozen zone từ {path.name}")
    return sorted(zone_ids)


def load_full_year_agg(agg_dir: Path, months: tuple[str, ...]) -> pd.DataFrame:
    """Đọc mười hai monthly aggregate và loại duplicate zone-hour key."""
    frames: list[pd.DataFrame] = []
    for month_label in months:
        path = agg_dir / f"agg_{month_label}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu monthly aggregate: {path}")
        df = pd.read_csv(path, parse_dates=["hour"])
        frames.append(df)

    agg_all = pd.concat(frames, ignore_index=True)
    del frames
    print(f"Đã gộp 12 monthly aggregate: {len(agg_all):,} zone-hour row.")

    n_before = len(agg_all)
    agg_all = (
        agg_all.groupby(["pu_location_id", "hour"], as_index=False)["trip_count"]
        .sum()
    )
    n_after = len(agg_all)
    if n_before != n_after:
        print(
            f"Cảnh báo: phát hiện {n_before - n_after:,} zone-hour row bị duplicate trong "
            f"các monthly file; đã cộng trước khi merge ({n_before:,} -> {n_after:,})."
        )
    else:
        print("Không có zone-hour key duplicate giữa các monthly file.")

    return agg_all


def build_hourly_grid(
    zone_ids: list[int], panel_start: pd.Timestamp, panel_end_exclusive: pd.Timestamp
) -> pd.DataFrame:
    """Xây dựng mọi tổ hợp zone-hour trong calendar year 2025."""
    hours = pd.date_range(
        start=panel_start, end=panel_end_exclusive - pd.Timedelta(hours=1), freq="h"
    )
    grid = pd.MultiIndex.from_product(
        [zone_ids, hours], names=["pu_location_id", "target_datetime"]
    ).to_frame(index=False)

    print(
        f"Đã xây dựng dense grid: {len(zone_ids)} zone x {len(hours)} hour "
        f"= {len(grid):,} row."
    )
    return grid


def merge_demand(grid: pd.DataFrame, agg_all: pd.DataFrame, zone_ids: list[int]) -> pd.DataFrame:
    """Merge demand thưa vào dense grid và điền zero cho hour bị thiếu."""
    agg_top50 = agg_all[agg_all["pu_location_id"].isin(zone_ids)].rename(
        columns={"hour": "target_datetime", "trip_count": "demand"}
    )

    panel = grid.merge(
        agg_top50, on=["pu_location_id", "target_datetime"], how="left"
    )

    if len(panel) != len(grid):
        raise ValueError(
            f"Demand merge tạo {len(panel):,} row cho grid {len(grid):,} row; "
            "monthly aggregate vẫn chứa zone-hour key duplicate."
        )

    n_zero_filled = panel["demand"].isna().sum()
    panel["demand"] = panel["demand"].fillna(0).astype("int32")

    panel = panel.sort_values(["pu_location_id", "target_datetime"]).reset_index(drop=True)

    print(
        f"Đã merge demand vào dense grid: điền zero cho {n_zero_filled:,} zone-hour "
        f"({n_zero_filled / len(panel):.1%})."
    )
    return panel


def add_calendar_features(
    panel: pd.DataFrame, panel_start: pd.Timestamp, panel_end_exclusive: pd.Timestamp
) -> pd.DataFrame:
    """Thêm feature hour, day-of-week và US federal holiday."""
    panel = panel.copy()
    panel["hour"] = panel["target_datetime"].dt.hour.astype("int8")
    panel["dayofweek"] = panel["target_datetime"].dt.dayofweek.astype("int8")

    cal = USFederalHolidayCalendar()
    holiday_dates = set(
        cal.holidays(start=panel_start, end=panel_end_exclusive).date
    )
    panel["is_holiday"] = panel["target_datetime"].dt.date.isin(holiday_dates).astype("int8")

    print(f"Đã thêm calendar feature; số US federal holiday năm 2025: {len(holiday_dates)}.")
    return panel


def add_lag_features(panel: pd.DataFrame, all_lags: tuple[int, ...]) -> pd.DataFrame:
    """Thêm toàn bộ protocol lag mà không mutate input DataFrame."""
    panel = panel.copy()
    grouped = panel.groupby("pu_location_id")["demand"]
    for lag in all_lags:
        panel[f"lag_{lag}"] = grouped.shift(lag)

    print(f"Đã thêm lag feature: {[f'lag_{l}' for l in all_lags]}")
    return panel


def add_variant_features(
    panel: pd.DataFrame, weekly_lags: tuple[int, ...], median_feature: str
) -> pd.DataFrame:
    """Thêm median của ba weekly lag."""
    panel = panel.copy()
    weekly_columns = [f"lag_{lag}" for lag in weekly_lags]
    panel[median_feature] = panel[weekly_columns].median(axis=1, skipna=False)
    print(f"Đã thêm {median_feature} là median của ba weekly lag.")
    return panel


def apply_warmup_cutoff(
    panel: pd.DataFrame, all_lags: tuple[int, ...], expected_cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Loại row không có đủ lịch sử 504-hour."""
    n_before = len(panel)
    lag_cols = [f"lag_{l}" for l in all_lags]
    panel = panel.dropna(subset=lag_cols).reset_index(drop=True)
    n_after = len(panel)

    first_valid_date = panel["target_datetime"].min()
    if first_valid_date != expected_cutoff:
        raise ValueError(
            f"Warm-up cutoff tạo ra {first_valid_date}; kỳ vọng {expected_cutoff}"
        )
    print(
        f"Warm-up cutoff: đã loại {n_before - n_after:,} row "
        f"({n_before:,} -> {n_after:,}); row đầu tiên giữ lại: {first_valid_date}."
    )

    for lag in all_lags:
        panel[f"lag_{lag}"] = panel[f"lag_{lag}"].astype("int32")

    return panel


def sanity_check(panel: pd.DataFrame, zone_ids: list[int]):
    """Kiểm tra modeling table cuối theo panel contract đã freeze."""
    print("=== KIỂM TRA HỢP LỆ ===")
    if panel["pu_location_id"].nunique() != len(zone_ids):
        raise ValueError("Final panel không chứa đúng các frozen zone")
    if panel[["pu_location_id", "target_datetime"]].duplicated().any():
        raise ValueError("Final panel chứa zone-hour row duplicate")
    if panel["demand"].min() < 0:
        raise ValueError("Final panel chứa demand âm")

    print(f"Tổng row cuối: {len(panel):,}; zone: {panel['pu_location_id'].nunique()}")
    print("Kiểm tra hợp lệ PASS.")


def save_variant_map(
    path: Path, base_features: tuple[str, ...], variants: dict[str, VariantConfig]
) -> None:
    """Lưu định nghĩa feature A/B/C cho downstream training."""
    record = {
        "base_features": list(base_features),
        "variants": {
            name: {
                "weekly_features": list(variant["weekly_features"]),
                "description": variant["description"],
            }
            for name, variant in variants.items()
        },
        "note": (
            "Mỗi variant dùng base_features cùng weekly_features tương ứng. "
            f"Variant A dùng base_features + {list(variants['A']['weekly_features'])}."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu variant feature map -> {path}")


class LagAlignmentRecord(TypedDict):
    """Record source-demand dùng để verify lag alignment."""

    pu_location_id: int
    target_datetime: pd.Timestamp
    lag: int
    source_datetime: pd.Timestamp
    source_demand: int
    feature_value: int
    matches: bool


def add_zone_onehot(df: pd.DataFrame, zone_ids: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode frozen zone với thứ tự column ổn định."""
    encoded = df.copy()
    if encoded["pu_location_id"].isna().any():
        raise ValueError("Không thể one-hot encode row thiếu pu_location_id")
    observed_ids = set(encoded["pu_location_id"].astype(int).tolist())
    unknown_ids = sorted(observed_ids - set(zone_ids))
    if unknown_ids:
        raise ValueError(f"Row chứa zone ID ngoài frozen vocabulary: {unknown_ids}")
    encoded["pu_location_id"] = pd.Categorical(
        encoded["pu_location_id"], categories=zone_ids
    )
    dummies = pd.get_dummies(encoded["pu_location_id"], prefix="zone")
    zone_columns = [f"zone_{zone_id}" for zone_id in zone_ids]
    dummies = dummies.reindex(columns=zone_columns, fill_value=False)
    return pd.concat([encoded, dummies], axis=1), zone_columns


def create_lag_alignment_examples(
    panel: pd.DataFrame,
    zone_ids: list[int],
    all_lags: tuple[int, ...],
    sample_times: tuple[pd.Timestamp, ...],
    output_path: Path,
) -> None:
    """Ghi các kiểm tra lag tại warm-up point và split boundary."""
    sample_zones = [zone_ids[0], zone_ids[len(zone_ids) // 2], zone_ids[-1]]
    lookup = panel.set_index(["pu_location_id", "target_datetime"])["demand"]
    records: list[LagAlignmentRecord] = []
    for zone_id in sample_zones:
        for target_time in sample_times:
            row = panel[
                (panel["pu_location_id"] == zone_id)
                & (panel["target_datetime"] == target_time)
            ]
            if len(row) != 1:
                raise ValueError(f"Kỳ vọng đúng một panel row cho zone={zone_id}, time={target_time}")
            for lag in all_lags:
                source_time = target_time - pd.Timedelta(hours=lag)
                expected = lookup.loc[(zone_id, source_time)]
                actual = row.iloc[0][f"lag_{lag}"]
                expected_value = int(expected)
                actual_value = int(actual)
                records.append(
                    {
                        "pu_location_id": zone_id,
                        "target_datetime": target_time,
                        "lag": lag,
                        "source_datetime": source_time,
                        "source_demand": expected_value,
                        "feature_value": actual_value,
                        "matches": expected_value == actual_value,
                    }
                )

    examples = pd.DataFrame.from_records(records)
    if not examples["matches"].all():
        raise ValueError("Lag-alignment example chứa mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output_path, index=False)


def build_feature_table(config: DataConfig) -> pd.DataFrame:
    """Xây dựng, validate và lưu các Data handoff artifact."""
    paths = config["paths"]
    panel_config = config["panel"]
    zone_ids = load_frozen_zones(paths["frozen_zones"])
    agg_all = load_full_year_agg(paths["monthly_aggregates_dir"], config["months"])

    grid = build_hourly_grid(zone_ids, panel_config["start"], panel_config["end_exclusive"])
    panel = merge_demand(grid, agg_all, zone_ids)
    del grid, agg_all

    panel = add_calendar_features(panel, panel_config["start"], panel_config["end_exclusive"])
    panel = add_lag_features(panel, panel_config["all_lags_hours"])
    panel = add_variant_features(
        panel, panel_config["weekly_lags_hours"], panel_config["median_feature"]
    )
    warmup_start = panel_config["start"] + pd.Timedelta(
        hours=max(panel_config["all_lags_hours"])
    )
    sample_times = (
        warmup_start,
        warmup_start + pd.Timedelta(hours=1),
        config["splits"]["hpo"]["eval_start"],
        config["splits"]["fold1"]["eval_start"],
        config["splits"]["final_test"]["eval_start"],
    )
    create_lag_alignment_examples(
        panel, zone_ids, panel_config["all_lags_hours"], sample_times, paths["lag_examples"]
    )
    panel = apply_warmup_cutoff(
        panel,
        panel_config["all_lags_hours"],
        warmup_start,
    )

    sanity_check(panel, zone_ids)

    paths["feature_table"].parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(paths["feature_table"], index=False)
    print(f"Đã lưu final feature table -> {paths['feature_table']}")

    save_variant_map(paths["variant_map"], panel_config["base_features"], panel_config["variants"])
    return panel


def main() -> None:
    """Xây dựng các Data artifact chuẩn từ monthly aggregate."""
    build_feature_table(load_data_config())


if __name__ == "__main__":
    main()
