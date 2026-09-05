"""Build monthly aggregates and freeze the top 50 zones for January-June."""

import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data.aggregate_month import aggregate_month
from src.configuration import DataConfig, load_data_config


def run_all_months(config: DataConfig) -> dict[str, Path]:
    """Aggregate every required month and return the output paths."""
    source = config["source"]
    raw_dir = config["paths"]["raw_dir"]
    aggregate_dir = config["paths"]["monthly_aggregates_dir"]
    raw_paths = {
        month_label: raw_dir / source["filename_pattern"].format(
            year=config["year"], month=int(month_label[-2:])
        )
        for month_label in config["months"]
    }
    missing = [month for month, path in raw_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw HVFHV months: {missing}")

    agg_paths: dict[str, Path] = {}
    for month_label, raw_path in tqdm(
        raw_paths.items(),
        total=len(raw_paths),
        desc="Preparing monthly aggregates",
        unit="month",
    ):
        out_path = aggregate_dir / f"agg_{month_label}.csv"
        if not out_path.exists():
            out_path = aggregate_month(
                raw_path,
                month_label,
                aggregate_dir,
                source["request_datetime_column"],
                source["pickup_zone_column"],
                source["normalized_zone_column"],
            )
        else:
            tqdm.write(f"Reusing existing monthly aggregate -> {out_path}")
        agg_paths[month_label] = out_path

    return agg_paths


def select_and_freeze_top50(agg_paths: dict[str, Path], config: DataConfig) -> list[int]:
    """Select the 50 zones with the highest demand in January-June 2025."""
    months = config["months"]
    selection = config["selection"]
    missing = [m for m in months if m not in agg_paths]
    if missing:
        raise RuntimeError(f"Missing monthly aggregates {missing}; cannot freeze zones.")

    period_frames = [pd.read_csv(agg_paths[m], parse_dates=["hour"]) for m in months]
    jan_jun_all = pd.concat(period_frames, ignore_index=True)
    del period_frames
    jan_jun_all = jan_jun_all[
        (jan_jun_all["hour"] >= selection["start"])
        & (jan_jun_all["hour"] < selection["end_exclusive"])
    ]

    zone_totals = jan_jun_all.groupby("pu_location_id")["trip_count"].sum()
    zone_totals = zone_totals.rename("total_demand").reset_index()
    zone_totals = zone_totals.sort_values(
        ["total_demand", "pu_location_id"], ascending=[False, True]
    ).reset_index(drop=True)
    del jan_jun_all

    top50 = zone_totals.head(selection["top_zones"])
    top50_ids = top50["pu_location_id"].astype(int).tolist()
    if len(top50_ids) != selection["top_zones"]:
        raise ValueError(
            f"Selected {len(top50_ids)} frozen zones; expected {selection['top_zones']}"
        )
    if len(set(top50_ids)) != selection["top_zones"]:
        raise ValueError("Frozen-zone list contains duplicate IDs")

    freeze_record = {
        "selection_period": selection["period_label"],
        "selection_rule": selection["selection_rule"],
        "n_zones": selection["top_zones"],
        "zone_ids": top50_ids,
        "zone_total_demand": {
            str(int(row.pu_location_id)): int(row.total_demand)
            for row in top50.itertuples(index=False)
        },
    }

    freeze_path = config["paths"]["frozen_zones"]
    freeze_path.parent.mkdir(parents=True, exist_ok=True)

    if freeze_path.exists():
        with freeze_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != freeze_record:
            differing_fields = sorted(
                key
                for key in set(existing) | set(freeze_record)
                if existing.get(key) != freeze_record.get(key)
            )
            raise ValueError(
                "Existing frozen-zone artifact differs from the deterministic "
                f"selection for fields: {differing_fields}. "
                "Remove it and regenerate from the current approved inputs."
            )
        print(f"Validated existing frozen-zone artifact (exact match) -> {freeze_path}")
        return top50_ids

    with freeze_path.open("w", encoding="utf-8") as f:
        json.dump(freeze_record, f, indent=2)

    print(f"Froze top-{selection['top_zones']} zones -> {freeze_path}")
    print(f"Top five zones by demand: {top50_ids[:5]}")

    return top50_ids


def main() -> None:
    config = load_data_config()
    agg_paths = run_all_months(config)
    select_and_freeze_top50(agg_paths, config)


if __name__ == "__main__":
    main()
