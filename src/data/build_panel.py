"""Build the dense hourly panel and shared A/B/C feature contract."""

import json
from pathlib import Path
from typing import TypedDict
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.configuration import DataConfig, VariantConfig, load_data_config


def load_frozen_zones(path: Path) -> list[int]:
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    zone_ids = record["zone_ids"]
    print(f"Loaded {len(zone_ids)} frozen zones from {path.name}")
    return sorted(zone_ids)


def load_full_year_agg(agg_dir: Path, months: tuple[str, ...]) -> pd.DataFrame:
    """Read all twelve monthly aggregates and remove duplicate zone-hour keys."""
    frames: list[pd.DataFrame] = []
    for month_label in months:
        path = agg_dir / f"agg_{month_label}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing monthly aggregate: {path}")
        df = pd.read_csv(path, parse_dates=["hour"])
        frames.append(df)

    agg_all = pd.concat(frames, ignore_index=True)
    del frames
    print(f"Combined 12 monthly aggregates: {len(agg_all):,} zone-hour rows.")

    n_before = len(agg_all)
    agg_all = (
        agg_all.groupby(["pu_location_id", "hour"], as_index=False)["trip_count"]
        .sum()
    )
    n_after = len(agg_all)
    if n_before != n_after:
        print(
            f"Warning: found {n_before - n_after:,} duplicate zone-hour rows across "
            f"monthly files; summed before merging ({n_before:,} -> {n_after:,})."
        )
    else:
        print("No duplicate zone-hour keys across monthly files.")

    return agg_all


def build_hourly_grid(
    zone_ids: list[int], panel_start: pd.Timestamp, panel_end_exclusive: pd.Timestamp
) -> pd.DataFrame:
    """Build every zone-hour combination in calendar year 2025."""
    hours = pd.date_range(
        start=panel_start, end=panel_end_exclusive - pd.Timedelta(hours=1), freq="h"
    )
    grid = pd.MultiIndex.from_product(
        [zone_ids, hours], names=["pu_location_id", "target_datetime"]
    ).to_frame(index=False)

    print(
        f"Built dense grid: {len(zone_ids)} zones x {len(hours)} hours "
        f"= {len(grid):,} rows."
    )
    return grid


def merge_demand(grid: pd.DataFrame, agg_all: pd.DataFrame, zone_ids: list[int]) -> pd.DataFrame:
    """Merge sparse demand into the dense grid and fill missing hours with zero."""
    agg_top50 = agg_all[agg_all["pu_location_id"].isin(zone_ids)].rename(
        columns={"hour": "target_datetime", "trip_count": "demand"}
    )

    panel = grid.merge(
        agg_top50, on=["pu_location_id", "target_datetime"], how="left"
    )

    if len(panel) != len(grid):
        raise ValueError(
            f"Demand merge produced {len(panel):,} rows for a {len(grid):,}-row grid; "
            "monthly aggregates still contain duplicate zone-hour keys."
        )

    n_zero_filled = panel["demand"].isna().sum()
    panel["demand"] = panel["demand"].fillna(0).astype("int32")

    panel = panel.sort_values(["pu_location_id", "target_datetime"]).reset_index(drop=True)

    print(
        f"Merged demand into dense grid: filled {n_zero_filled:,} zone-hours "
        f"with zero ({n_zero_filled / len(panel):.1%})."
    )
    return panel


def add_calendar_features(
    panel: pd.DataFrame, panel_start: pd.Timestamp, panel_end_exclusive: pd.Timestamp
) -> pd.DataFrame:
    """Add hour, day-of-week, and US federal holiday features."""
    panel = panel.copy()
    panel["hour"] = panel["target_datetime"].dt.hour.astype("int8")
    panel["dayofweek"] = panel["target_datetime"].dt.dayofweek.astype("int8")

    cal = USFederalHolidayCalendar()
    holiday_dates = set(
        cal.holidays(start=panel_start, end=panel_end_exclusive).date
    )
    panel["is_holiday"] = panel["target_datetime"].dt.date.isin(holiday_dates).astype("int8")

    print(f"Added calendar features; 2025 US federal holidays: {len(holiday_dates)}.")
    return panel


def add_lag_features(panel: pd.DataFrame, all_lags: tuple[int, ...]) -> pd.DataFrame:
    """Add all protocol lags without mutating the input DataFrame."""
    panel = panel.copy()
    grouped = panel.groupby("pu_location_id")["demand"]
    for lag in all_lags:
        panel[f"lag_{lag}"] = grouped.shift(lag)

    print(f"Added lag features: {[f'lag_{l}' for l in all_lags]}")
    return panel


def add_variant_features(
    panel: pd.DataFrame, weekly_lags: tuple[int, ...], median_feature: str
) -> pd.DataFrame:
    """Add the median of the three weekly lags."""
    panel = panel.copy()
    weekly_columns = [f"lag_{lag}" for lag in weekly_lags]
    panel[median_feature] = panel[weekly_columns].median(axis=1, skipna=False)
    print(f"Added {median_feature} as the median of the three weekly lags.")
    return panel


def apply_warmup_cutoff(
    panel: pd.DataFrame, all_lags: tuple[int, ...], expected_cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Drop rows that do not have the full 504-hour history."""
    n_before = len(panel)
    lag_cols = [f"lag_{l}" for l in all_lags]
    panel = panel.dropna(subset=lag_cols).reset_index(drop=True)
    n_after = len(panel)

    first_valid_date = panel["target_datetime"].min()
    if first_valid_date != expected_cutoff:
        raise ValueError(
            f"Warm-up cutoff produced {first_valid_date}; expected {expected_cutoff}"
        )
    print(
        f"Warm-up cutoff: dropped {n_before - n_after:,} rows "
        f"({n_before:,} -> {n_after:,}); first retained row: {first_valid_date}."
    )

    for lag in all_lags:
        panel[f"lag_{lag}"] = panel[f"lag_{lag}"].astype("int32")

    return panel


def sanity_check(panel: pd.DataFrame, zone_ids: list[int]):
    """Check the final modeling table against the frozen panel contract."""
    print("=== SANITY CHECK ===")
    if panel["pu_location_id"].nunique() != len(zone_ids):
        raise ValueError("Final panel does not contain exactly the frozen zones")
    if panel[["pu_location_id", "target_datetime"]].duplicated().any():
        raise ValueError("Final panel contains duplicate zone-hour rows")
    if panel["demand"].min() < 0:
        raise ValueError("Final panel contains negative demand")

    print(f"Final rows: {len(panel):,}; zones: {panel['pu_location_id'].nunique()}")
    print("Sanity check PASS.")


def save_variant_map(
    path: Path, base_features: tuple[str, ...], variants: dict[str, VariantConfig]
) -> None:
    """Save the A/B/C feature definitions for downstream training."""
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
            "Each variant uses base_features plus its corresponding weekly_features. "
            f"Variant A uses base_features + {list(variants['A']['weekly_features'])}."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Saved variant feature map -> {path}")


class LagAlignmentRecord(TypedDict):
    """Source-demand record used to verify lag alignment."""

    pu_location_id: int
    target_datetime: pd.Timestamp
    lag: int
    source_datetime: pd.Timestamp
    source_demand: int
    feature_value: int
    matches: bool


def add_zone_onehot(df: pd.DataFrame, zone_ids: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode frozen zones with a stable column order."""
    encoded = df.copy()
    if encoded["pu_location_id"].isna().any():
        raise ValueError("Cannot one-hot encode rows with missing pu_location_id")
    observed_ids = set(encoded["pu_location_id"].astype(int).tolist())
    unknown_ids = sorted(observed_ids - set(zone_ids))
    if unknown_ids:
        raise ValueError(f"Rows contain zone IDs outside the frozen vocabulary: {unknown_ids}")
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
    """Write lag checks at the warm-up point and split boundaries."""
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
                raise ValueError(f"Expected exactly one panel row for zone={zone_id}, time={target_time}")
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
        raise ValueError("Lag-alignment examples contain a mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output_path, index=False)


def build_feature_table(config: DataConfig) -> pd.DataFrame:
    """Build, validate, and save the Data handoff artifacts."""
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
    print(f"Saved final feature table -> {paths['feature_table']}")

    save_variant_map(paths["variant_map"], panel_config["base_features"], panel_config["variants"])
    return panel


def main() -> None:
    """Build the standard Data artifacts from monthly aggregates."""
    build_feature_table(load_data_config())


if __name__ == "__main__":
    main()
