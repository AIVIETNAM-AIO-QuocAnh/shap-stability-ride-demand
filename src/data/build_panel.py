"""
Dựng complete hourly panel cho top-50 zone đã freeze.
- Grid đầy đủ: 50 zone x mọi giờ trong năm 2025 (kể cả giờ demand=0).
- Feature thời gian: hour (0-23), dayofweek (0-6, Monday=0), is_holiday.
- Lag feature: lag_1, lag_24, lag_168, lag_336, lag_504 (tính theo groupby zone
  + shift, hợp lệ vì grid đã đặc và liên tục theo giờ, không có khoảng trống).
- Variant A/B/C: định nghĩa tập cột weekly-lag dùng cho từng variant, lưu
  kèm feature table để bước train model đọc lại, không tách thành 3 file riêng
  (đúng protocol: 3 variant dùng chung 1 dataset, chỉ khác nhóm weekly lag).
- Warm-up cutoff: drop các row đầu năm bị NaN do thiếu đủ 504 giờ lịch sử
  -> tự động cắt đúng mốc 22/01/2025.
"""

import json
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# ============ CONFIG ============
REPO_ROOT = Path(__file__).resolve().parents[2]
AGG_DIR = REPO_ROOT / "data/processed/monthly_agg"
FROZEN_ZONES_PATH = REPO_ROOT / "data/processed/frozen/top50_zones_frozen.json"
OUTPUT_DIR = REPO_ROOT / "data/processed"
FEATURE_TABLE_PATH = OUTPUT_DIR / "feature_table.csv"
VARIANT_MAP_PATH = OUTPUT_DIR / "variant_feature_map.json"

MONTHS_2025 = [f"2025-{m:02d}" for m in range(1, 13)]

FULL_YEAR_START = "2025-01-01 00:00:00"
FULL_YEAR_END = "2025-12-31 23:00:00"

SHORT_LAGS = [1, 24]
WEEKLY_LAGS = [168, 336, 504]
ALL_LAGS = SHORT_LAGS + WEEKLY_LAGS

# Định nghĩa Variant A/B/C
VARIANT_FEATURE_MAP = {
    "A": {
        "weekly_features": ["lag_168", "lag_336", "lag_504"],
        "description": "Baseline: giữ nguyên các weekly lag riêng lẻ",
    },
    "B": {
        "weekly_features": ["median_lag_3w"],
        "description": "Aggregation: gộp ba weekly lag thành một đại diện (median)",
    },
    "C": {
        "weekly_features": ["lag_168", "median_lag_3w"],
        "description": "Hybrid: giữ tuần gần nhất riêng và thêm đại diện gộp",
    },
}
BASE_FEATURES = ["hour", "dayofweek", "is_holiday", "lag_1", "lag_24"]
# ==================================


def load_frozen_zones(path: Path) -> list[int]:
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    zone_ids = record["zone_ids"]
    print(f"Đã load {len(zone_ids)} zone đã freeze từ {path.name}")
    return sorted(zone_ids)


def load_full_year_agg(agg_dir: Path) -> pd.DataFrame:
    """Đọc lại 12 file aggregate đã tạo ở bước trước, gộp thành 1 bảng."""
    frames = []
    for month_label in MONTHS_2025:
        path = agg_dir / f"agg_{month_label}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu file aggregate: {path}")
        df = pd.read_csv(path, parse_dates=["hour"])
        frames.append(df)

    agg_all = pd.concat(frames, ignore_index=True)
    del frames
    print(f"Đã gộp 12 tháng aggregate: {len(agg_all):,} zone-hour rows.")

    # Gộp (sum trip_count) theo (pu_location_id, hour) để loại trùng key trước khi
    # merge vào grid, tránh 1 row trong grid bị nhân đôi khi merge.
    n_before = len(agg_all)
    agg_all = (
        agg_all.groupby(["pu_location_id", "hour"], as_index=False)["trip_count"]
        .sum()
    )
    n_after = len(agg_all)
    if n_before != n_after:
        print(
            f"CẢNH BÁO: phát hiện {n_before - n_after:,} row (zone, giờ) bị trùng "
            f"giữa các file tháng (thường do lệch ranh giới request_datetime/pickup_datetime "
            f"ở đầu-cuối tháng) -> đã cộng dồn trip_count lại. "
            f"{n_before:,} -> {n_after:,} row sau dedupe."
        )
    else:
        print("Không phát hiện (zone, giờ) trùng giữa các file tháng.")

    return agg_all


def build_hourly_grid(zone_ids: list[int]) -> pd.DataFrame:
    """
    Dựng dense grid: mọi (zone, giờ) trong năm 2025, kể cả giờ chưa từng có request nào.
    """
    hours = pd.date_range(start=FULL_YEAR_START, end=FULL_YEAR_END, freq="h")
    grid = pd.MultiIndex.from_product(
        [zone_ids, hours], names=["pu_location_id", "target_datetime"]
    ).to_frame(index=False)

    expected_rows = len(zone_ids) * len(hours)
    assert len(grid) == expected_rows, "Grid dựng sai kích thước!"
    print(
        f"Đã dựng dense grid: {len(zone_ids)} zone x {len(hours)} giờ "
        f"= {len(grid):,} row."
    )
    return grid


def merge_demand(grid: pd.DataFrame, agg_all: pd.DataFrame, zone_ids: list[int]) -> pd.DataFrame:
    """Merge demand thưa vào dense grid; giờ không có request -> demand = 0."""
    agg_top50 = agg_all[agg_all["pu_location_id"].isin(zone_ids)].rename(
        columns={"hour": "target_datetime", "trip_count": "demand"}
    )

    panel = grid.merge(
        agg_top50, on=["pu_location_id", "target_datetime"], how="left"
    )

    # Kiểm tra xem sau gộp có thừa dòng không
    if len(panel) != len(grid):
        raise ValueError(
            f"Merge tạo ra {len(panel):,} row nhưng grid gốc chỉ có {len(grid):,} row "
            f"-> agg_top50 vẫn còn (zone, giờ) bị trùng key. Kiểm tra lại bước dedupe "
            f"ở load_full_year_agg()."
        )

    n_zero_filled = panel["demand"].isna().sum()
    panel["demand"] = panel["demand"].fillna(0).astype("int32")

    panel = panel.sort_values(["pu_location_id", "target_datetime"]).reset_index(drop=True)

    print(
        f"Đã merge demand vào grid: {n_zero_filled:,} zone-hour được fill demand=0 "
        f"({n_zero_filled / len(panel):.1%} tổng số row)."
    )
    return panel


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Feature thời gian: hour (0-23), dayofweek (0=Monday..6=Sunday), is_holiday."""
    panel["hour"] = panel["target_datetime"].dt.hour.astype("int8")
    panel["dayofweek"] = panel["target_datetime"].dt.dayofweek.astype("int8")

    cal = USFederalHolidayCalendar()
    holiday_dates = set(
        cal.holidays(start=FULL_YEAR_START, end=FULL_YEAR_END).date
    )
    panel["is_holiday"] = panel["target_datetime"].dt.date.isin(holiday_dates).astype("int8")

    print(f"Đã thêm calendar feature. Số ngày lễ trong 2025: {len(holiday_dates)}.")
    return panel


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    grouped = panel.groupby("pu_location_id")["demand"]
    for lag in ALL_LAGS:
        panel[f"lag_{lag}"] = grouped.shift(lag)

    print(f"Đã tính lag feature: {[f'lag_{l}' for l in ALL_LAGS]}")
    return panel


def add_variant_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel["median_lag_3w"] = panel[["lag_168", "lag_336", "lag_504"]].median(
        axis=1, skipna=False
    )
    print("Đã tính median_lag_3w = median(lag_168, lag_336, lag_504).")
    return panel


def apply_warmup_cutoff(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Drop các row đầu năm bị NaN do chưa đủ 504 giờ lịch sử
    """
    n_before = len(panel)
    lag_cols = [f"lag_{l}" for l in ALL_LAGS]
    panel = panel.dropna(subset=lag_cols).reset_index(drop=True)
    n_after = len(panel)

    first_valid_date = panel["target_datetime"].min()
    print(
        f"Warm-up cutoff: loại {n_before - n_after:,} row "
        f"({n_before:,} -> {n_after:,}). Row đầu tiên còn lại: {first_valid_date}."
    )

    for lag in ALL_LAGS:
        panel[f"lag_{lag}"] = panel[f"lag_{lag}"].astype("int32")

    return panel


def sanity_check(panel: pd.DataFrame, zone_ids: list[int]):
    print("=== SANITY CHECK ===")
    assert panel["pu_location_id"].nunique() == len(zone_ids), "Thiếu/thừa zone trong panel!"
    assert panel[["pu_location_id", "target_datetime"]].duplicated().sum() == 0, (
        "Có row (zone, giờ) bị trùng!"
    )
    assert panel["demand"].min() >= 0, "Có demand âm, vô lý!"
    assert panel[[f"lag_{l}" for l in ALL_LAGS]].isna().sum().sum() == 0, (
        "Vẫn còn NaN trong cột lag sau khi đã cutoff warm-up!"
    )

    first_date = panel["target_datetime"].min()
    expected_cutoff = pd.Timestamp("2025-01-22 00:00:00")
    print(f"Ngày bắt đầu thực tế: {first_date} (kỳ vọng ~ {expected_cutoff}).")

    print(f"Tổng số row cuối cùng: {len(panel):,}")
    print(f"Số zone: {panel['pu_location_id'].nunique()}")
    print("Sanity check PASS.")


def save_variant_map(path: Path):
    """Lưu định nghĩa Variant A/B/C để bước train model đọc lại."""
    record = {
        "base_features": BASE_FEATURES,
        "variants": VARIANT_FEATURE_MAP,
        "note": (
            "Mỗi variant dùng chung base_features + weekly_features tương ứng. "
            "Ví dụ Variant A: base_features + [lag_168, lag_336, lag_504]."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu variant feature map -> {path}")


def main():
    zone_ids = load_frozen_zones(FROZEN_ZONES_PATH)
    agg_all = load_full_year_agg(AGG_DIR)

    grid = build_hourly_grid(zone_ids)
    panel = merge_demand(grid, agg_all, zone_ids)
    del grid, agg_all

    panel = add_calendar_features(panel)
    panel = add_lag_features(panel)
    panel = add_variant_features(panel)
    panel = apply_warmup_cutoff(panel)

    sanity_check(panel, zone_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(FEATURE_TABLE_PATH, index=False)
    print(f"Đã lưu feature table cuối cùng -> {FEATURE_TABLE_PATH}")

    save_variant_map(VARIANT_MAP_PATH)


if __name__ == "__main__":
    main()