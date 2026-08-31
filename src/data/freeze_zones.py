"""
Chạy aggregate cho cả 12 tháng của 2025.
Chọn 50 zone có tổng demand cao nhất trong 01/01-30/06/2025, freeze danh sách.
"""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from aggregate_month import aggregate_month

# ============ CONFIG ============
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data/raw"
AGG_OUTPUT_DIR = REPO_ROOT / "data/processed/monthly_agg"
FROZEN_DIR = REPO_ROOT / "data/processed/frozen"

RAW_FILENAME_PATTERN = "fhvhv_tripdata_2025-{month:02d}.csv"

MONTHS_2025 = [f"2025-{m:02d}" for m in range(1, 13)]
JAN_JUN_MONTHS = [f"2025-{m:02d}" for m in range(1, 7)]
N_TOP_ZONES = 50
# ==================================

def run_all_months() -> dict[str, Path]:
    """Aggregate lần lượt từng tháng"""
    agg_paths = {}
    for m in range(1, 13):
        month_label = f"2025-{m:02d}"
        raw_path = RAW_DATA_DIR / RAW_FILENAME_PATTERN.format(month=m)

        if not raw_path.exists():
            print(f"[{month_label}] LỖI: KHÔNG tìm thấy file: {raw_path}.")
            continue

        out_path = aggregate_month(raw_path, month_label, AGG_OUTPUT_DIR)
        agg_paths[month_label] = out_path

    return agg_paths


def select_and_freeze_top50(agg_paths: dict[str, Path]) -> list[int]:
    """
    Lấy 50 zone cao nhất trong tổng demand mỗi zone trong 01/01-30/06/2025
    """
    missing = [m for m in JAN_JUN_MONTHS if m not in agg_paths]
    if missing:
        raise RuntimeError(
            f"Thiếu aggregate của các tháng {missing} trong Jan-Jun -> "
            f"không thể chọn top-50 zone một cách hợp lệ."
        )

    jan_jun_frames = [pd.read_csv(agg_paths[m], parse_dates=["hour"]) for m in JAN_JUN_MONTHS]
    jan_jun_all = pd.concat(jan_jun_frames, ignore_index=True)
    del jan_jun_frames

    zone_totals = (
        jan_jun_all.groupby("PULocationID")["trip_count"]
        .sum()
        .sort_values(ascending=False)
    )
    del jan_jun_all

    top50 = zone_totals.head(N_TOP_ZONES)
    top50_ids = top50.index.astype(int).tolist()

    freeze_record = {
        "frozen_at_utc": datetime.isoformat() + "Z",
        "selection_period": "2025-01-01 to 2025-06-30",
        "selection_rule": "top 50 PULocationID by total request count (demand), Jan-Jun 2025",
        "n_zones": N_TOP_ZONES,
        "zone_ids": top50_ids,
        "zone_total_demand": {int(k): int(v) for k, v in top50.items()},
    }

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    freeze_path = FROZEN_DIR / "top50_zones_frozen.json"

    if freeze_path.exists():
        raise FileExistsError(
            f"{freeze_path} đã tồn tại. Danh sách zone đã được freeze trước đó; "
            f"xóa file thủ công nếu thực sự cần chọn lại."
        )

    with open(freeze_path, "w") as f:
        json.dump(freeze_record, f, indent=2)

    print(f"Đã freeze top-{N_TOP_ZONES} zone -> {freeze_path}")
    print(f"Top 5 zone theo demand: {top50_ids[:5]}")

    return top50_ids


def sanity_check(agg_paths: dict[str, Path], top50_ids: list[int]):
    """Kiểm tra nhanh"""
    print("=== SANITY CHECK ===")

    missing_months = [m for m in MONTHS_2025 if m not in agg_paths]
    if missing_months:
        print(f"CẢNH BÁO: Các tháng KHÔNG có aggregate: {missing_months}")
    else:
        print("Đủ 12/12 tháng đã được aggregate.")

    total_demand_all_year = 0
    for month_label, path in sorted(agg_paths.items()):
        df = pd.read_csv(path)
        month_total = int(df["trip_count"].sum())
        total_demand_all_year += month_total
        n_unique_zones = df["PULocationID"].nunique()
        print(
            f"  {month_label}: {len(df):,} zone-hour rows, "
            f"{n_unique_zones} zone khác nhau, tổng demand = {month_total:,}"
        )
        del df

    print(f"Tổng demand cả năm (toàn bộ zone, chưa lọc top-50): {total_demand_all_year:,}")
    print(f"Số zone đã freeze: {len(top50_ids)} (phải đúng {N_TOP_ZONES})")
    assert len(top50_ids) == N_TOP_ZONES, "Số zone freeze không đúng N_TOP_ZONES!"
    assert len(set(top50_ids)) == N_TOP_ZONES, "Có zone bị trùng trong danh sách freeze!"
    print("Sanity check PASS.")


def main():
    agg_paths = run_all_months()
    top50_ids = select_and_freeze_top50(agg_paths)
    sanity_check(agg_paths, top50_ids)


if __name__ == "__main__":
    main()