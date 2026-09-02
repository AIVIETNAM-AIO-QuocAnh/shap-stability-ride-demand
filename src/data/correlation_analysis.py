"""
1. QA checks trên feature_table.csv: missing, duplicate, zero-demand.
2. Pearson correlation giữa 3 weekly lag (lag_168, lag_336, lag_504), tính
   riêng từng zone, trên giai đoạn 22/01-31/07/2025.
3. Với mỗi cặp lag, báo cáo mean +- std của hệ số correlation trên 50 zone.
4. Sinh data dictionary cho feature_table.csv.
"""

from pathlib import Path
import pandas as pd

# ============ CONFIG ============
REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_TABLE_PATH = REPO_ROOT / "data" / "processed" / "feature_table.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"

CORR_BY_ZONE_PATH = OUTPUT_DIR / "correlation_by_zone.csv"
CORR_SUMMARY_PATH = OUTPUT_DIR / "correlation_summary.csv"
DATA_DICT_PATH = OUTPUT_DIR / "data_dictionary.md"

# Giai đoạn đo correlation
CORR_PERIOD_START = "2025-01-22"
CORR_PERIOD_END_EXCLUSIVE = "2025-08-01"

WEEKLY_LAG_COLS = ["lag_168", "lag_336", "lag_504"]
LAG_PAIRS = [
    ("lag_168", "lag_336"),
    ("lag_168", "lag_504"),
    ("lag_336", "lag_504"),
]

ALL_EXPECTED_COLS = [
    "pu_location_id", "target_datetime", "demand", "hour", "dayofweek",
    "is_holiday", "lag_1", "lag_24", "lag_168", "lag_336", "lag_504",
    "median_lag_3w",
]
# ==================================


def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["target_datetime"])
    print(f"Đã load feature table: {len(df):,} row, {df['pu_location_id'].nunique()} zone.")
    return df


def qa_check_missing(df: pd.DataFrame):
    print("\n=== QA: MISSING VALUES ===")
    missing_cols = [c for c in ALL_EXPECTED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature table thiếu cột bắt buộc: {missing_cols}")

    na_counts = df[ALL_EXPECTED_COLS].isna().sum()
    na_counts = na_counts[na_counts > 0]

    if len(na_counts) > 0:
        print("CẢNH BÁO: phát hiện missing value:")
        print(na_counts.to_string())
        raise ValueError(
            "Feature table có missing value ở cột bắt buộc, cần kiểm tra lại bước build_panel."
        )
    print("Không có missing value ở bất kỳ cột bắt buộc nào.")


def qa_check_duplicate(df: pd.DataFrame):
    print("\n=== QA: DUPLICATE (zone, giờ) ===")
    n_dup = df.duplicated(subset=["pu_location_id", "target_datetime"]).sum()
    if n_dup > 0:
        raise ValueError(
            f"Phát hiện {n_dup} row (zone, giờ) bị trùng trong feature table."
        )
    print("Không có row (zone, giờ) bị trùng.")


def qa_check_zero_demand(df: pd.DataFrame):
    """
    Báo cáo mức độ và cảnh báo nếu có zone gần như toàn giờ = 0
    (rủi ro correlation bị NaN do lag không có biến thiên).
    """
    print("\n=== QA: ZERO-DEMAND RATE ===")
    overall_zero_rate = (df["demand"] == 0).mean()
    print(f"Tỷ lệ giờ demand = 0 trên toàn bộ dữ liệu: {overall_zero_rate:.1%}")

    zone_zero_rate = df.groupby("pu_location_id")["demand"].apply(lambda s: (s == 0).mean())
    high_zero_zones = zone_zero_rate[zone_zero_rate > 0.95]

    if len(high_zero_zones) > 0:
        print(
            f"CẢNH BÁO: {len(high_zero_zones)} zone có >95% giờ demand=0 "
            f"-- rủi ro correlation NaN:"
        )
        print(high_zero_zones.round(3).to_string())
    else:
        print("Không có zone nào có tỷ lệ demand = 0 bất thường (>95%).")

    return zone_zero_rate


def compute_correlation_by_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation giữa từng cặp weekly lag.
    """
    print(f"\n=== TÍNH CORRELATION: {CORR_PERIOD_START} -> {CORR_PERIOD_END_EXCLUSIVE} (exclusive) ===")

    period_df = df[
        (df["target_datetime"] >= pd.Timestamp(CORR_PERIOD_START))
        & (df["target_datetime"] < pd.Timestamp(CORR_PERIOD_END_EXCLUSIVE))
    ]
    print(f"Số row trong giai đoạn đo correlation: {len(period_df):,}")

    records = []
    n_undefined = 0

    for zone, zone_df in period_df.groupby("pu_location_id"):
        for col_a, col_b in LAG_PAIRS:
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
    print(f"Đã tính correlation cho {n_zones} zone x {len(LAG_PAIRS)} cặp lag.")
    if n_undefined > 0:
        print(
            f"CẢNH BÁO: {n_undefined} cặp (zone, pair) có correlation UNDEFINED (NaN)."
        )

    return corr_df


def summarize_correlation(corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean +- std của Pearson correlation trên 50 zone, cho mỗi cặp lag.
    NaN (zone bị undefined) bị loại khi tính mean/std, nhưng số lượng bị
    loại được báo cáo rõ.
    """
    summary = (
        corr_df.groupby("pair")["pearson_r"]
        .agg(mean_r="mean", std_r="std", n_valid="count")
        .reset_index()
    )
    summary["n_zones_total"] = corr_df.groupby("pair")["pearson_r"].size().values
    summary["n_undefined"] = summary["n_zones_total"] - summary["n_valid"]

    print("\n=== SUMMARY: MEAN +- SD PEARSON CORRELATION TRÊN 50 ZONE ===")
    print(summary.round(4).to_string(index=False))

    return summary


def save_data_dictionary(path: Path):
    content = """# Data Dictionary - feature_table.csv & correlation outputs

Mỗi row = 1 (zone, giờ). Đơn vị thời gian: giờ (hourly).

## 1. `feature_table.csv`
| Cột | Kiểu | Mô tả | Ghi chú |
|---|---|---|---|
| `pu_location_id` | int | Mã zone (pickup location), 1 trong 50 zone đã freeze | Chưa one-hot |
| `target_datetime` | datetime | Mốc giờ t, giờ mà `demand` được đo | Chỉ để tham chiếu/debug, không phải feature |
| `demand` | int | **Target** — số request tại zone đó, trong giờ đó | >= 0, có thể = 0 (giờ không có request nào) |
| `hour` | int (0-23) | Giờ trong ngày, suy từ `target_datetime` | Feature nền |
| `dayofweek` | int (0-6) | Thứ trong tuần, 0 = Thứ Hai | Feature nền |
| `is_holiday` | int (0/1) | 1 nếu ngày đó là ngày lễ liên bang Mỹ | Feature nền |
| `lag_1` | int | Demand cách đây 1 giờ | Feature nền |
| `lag_24` | int | Demand cách đây 24 giờ (1 ngày) | Feature nền, dùng cho mọi Variant |
| `lag_168` | int | Demand cách đây 168 giờ (1 tuần) | Weekly lag, dùng ở Variant A/C |
| `lag_336` | int | Demand cách đây 336 giờ (2 tuần) | Weekly lag, dùng ở Variant A |
| `lag_504` | int | Demand cách đây 504 giờ (3 tuần) | Weekly lag, dùng ở Variant A |
| `median_lag_3w` | float | median(`lag_168`, `lag_336`, `lag_504`) | Weekly lag gộp, dùng ở Variant B/C |

### Ràng buộc đã đảm bảo

- Không có missing value ở bất kỳ cột bắt buộc nào.
- Không có row (zone, giờ) bị trùng.
- Dữ liệu bắt đầu từ **2025-01-22** (đã cắt warm-up để đảm bảo `lag_504` luôn có đủ lịch sử).
- `demand = 0` là giá trị hợp lệ (giờ không có request), không phải missing.

## 2. `correlation_by_zone.csv`
 
Pearson correlation giữa 3 weekly lag, tính riêng cho từng zone, chỉ trên giai đoạn 22/01–31/07/2025.
 
| Cột | Kiểu | Mô tả | Ghi chú |
|---|---|---|---|
| `pu_location_id` | int | Zone được tính correlation | 1 trong 50 zone đã freeze |
| `pair` | str | Cặp weekly lag, dạng `lag_A_vs_lag_B` | 1 trong 3 giá trị: `lag_168_vs_lag_336`, `lag_168_vs_lag_504`, `lag_336_vs_lag_504` |
| `pearson_r` | float | Hệ số Pearson correlation của riêng zone đó, cặp đó | Trong khoảng [-1, 1]; có thể là `NaN` nếu 1 trong 2 cột gần như hằng số trong giai đoạn đo |
| `n_obs` | int | Số giờ dữ liệu dùng để tính correlation cho zone đó | Dùng để đánh giá độ tin cậy của `pearson_r` |
 
## 3. `correlation_summary.csv`
 
Tổng hợp mean ± SD của `pearson_r` trên 50 zone, cho mỗi cặp lag.
 
| Cột | Kiểu | Mô tả | Ghi chú |
|---|---|---|---|
| `pair` | str | Cặp weekly lag | Cùng 3 giá trị như trên |
| `mean_r` | float | Trung bình `pearson_r` trên các zone **hợp lệ** (đã loại `NaN`) | |
| `std_r` | float | Độ lệch chuẩn `pearson_r` trên các zone hợp lệ | |
| `n_valid` | int | Số zone tính được correlation hợp lệ (không `NaN`) | |
| `n_zones_total` | int | Tổng số zone (kỳ vọng = 50) | |
| `n_undefined` | int | Số zone bị loại vì `pearson_r` là `NaN` |  |
 
## File liên quan
 
- `variant_feature_map.json` — định nghĩa cột feature theo Variant A/B/C.
- `frozen/top50_zones_frozen.json` — vocabulary cố định cho one-hot `pu_location_id`.

"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nĐã lưu data dictionary -> {path}")


def main():
    df = load_feature_table(FEATURE_TABLE_PATH)

    qa_check_missing(df)
    qa_check_duplicate(df)
    qa_check_zero_demand(df)

    corr_df = compute_correlation_by_zone(df)
    summary_df = summarize_correlation(corr_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(CORR_BY_ZONE_PATH, index=False)
    summary_df.to_csv(CORR_SUMMARY_PATH, index=False)
    print(f"\nĐã lưu correlation theo zone -> {CORR_BY_ZONE_PATH}")
    print(f"Đã lưu correlation summary -> {CORR_SUMMARY_PATH}")

    save_data_dictionary(DATA_DICT_PATH)

    print("\n=== BÀN GIAO ===")
    print(f"feature_table.csv đã pass QA, sẵn sàng bàn giao cùng:")
    print(f"  - {DATA_DICT_PATH.name}")
    print(f"  - {CORR_BY_ZONE_PATH.name}")
    print(f"  - {CORR_SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()