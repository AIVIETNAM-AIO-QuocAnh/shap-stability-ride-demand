"""Xây dựng monthly aggregate và freeze top 50 zone cho January-June."""

import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data.aggregate_month import aggregate_month
from src.load_config import DataConfig, load_data_config


def run_all_months(config: DataConfig) -> dict[str, Path]:
    """Tổng hợp mọi tháng bắt buộc và trả về output path."""
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
        raise FileNotFoundError(f"Thiếu raw HVFHV month: {missing}")

    agg_paths: dict[str, Path] = {}
    for month_label, raw_path in tqdm(
        raw_paths.items(),
        total=len(raw_paths),
        desc="Đang chuẩn bị monthly aggregate",
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
            tqdm.write(f"Dùng lại monthly aggregate đã có -> {out_path}")
        agg_paths[month_label] = out_path

    return agg_paths


def select_and_freeze_top50(agg_paths: dict[str, Path], config: DataConfig) -> list[int]:
    """Chọn 50 zone có demand cao nhất trong January-June 2025."""
    months = config["months"]
    selection = config["selection"]
    missing = [m for m in months if m not in agg_paths]
    if missing:
        raise RuntimeError(f"Thiếu monthly aggregate {missing}; không thể freeze zone.")

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
            f"Đã chọn {len(top50_ids)} frozen zone; kỳ vọng {selection['top_zones']}"
        )
    if len(set(top50_ids)) != selection["top_zones"]:
        raise ValueError("Danh sách frozen zone chứa ID duplicate")

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
                "Frozen-zone artifact hiện tại khác deterministic selection "
                f"ở các field: {differing_fields}. "
                "Hãy xoá artifact và sinh lại từ input đã được duyệt."
            )
        print(f"Đã validate frozen-zone artifact hiện tại (exact match) -> {freeze_path}")
        return top50_ids

    with freeze_path.open("w", encoding="utf-8") as f:
        json.dump(freeze_record, f, indent=2)

    print(f"Đã freeze top-{selection['top_zones']} zone -> {freeze_path}")
    print(f"Năm zone có demand cao nhất: {top50_ids[:5]}")

    return top50_ids


def main() -> None:
    config = load_data_config()
    agg_paths = run_all_months(config)
    select_and_freeze_top50(agg_paths, config)


if __name__ == "__main__":
    main()
