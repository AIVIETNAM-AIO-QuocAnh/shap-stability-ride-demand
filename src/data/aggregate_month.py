"""
Aggregate raw HVFHV theo tháng, theo zone x hour.
Chỉ đọc 2 cột cần thiết: request_datetime, pu_location_id (đúng protocol mục 2.1).
"""

import gc
from pathlib import Path

import pandas as pd

# Chỉ 2 cột này được đọc từ raw data
NEEDED_COLUMNS = ["request_datetime", "pu_location_id"]


def _read_columns_csv(filepath: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    """
    Chỉ đọc 2 cột cần thiết từ file CSV raw.
    Đọc theo chunk để tránh load nguyên file CSV vào RAM cùng lúc.
    """
    chunks = []
    for chunk in pd.read_csv(
        filepath,
        usecols=NEEDED_COLUMNS,
        parse_dates=["request_datetime"],
        chunksize=chunksize,
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    return df


def aggregate_month(
    filepath: Path,
    month_label: str,
    output_dir: Path,
) -> Path:
    """
    Aggregate 1 tháng raw trip data -> bảng (pu_location_id, hour, trip_count).

    trip_count = số request trong giờ đó tại zone đó, tính theo request_datetime
    (đúng định nghĩa target trong protocol: "Target là số request trong target hour").
    """
    print(f"[{month_label}] Đang đọc raw data (chỉ {NEEDED_COLUMNS}) từ {filepath.name}...")

    if filepath.suffix == ".csv":
        df = _read_columns_csv(filepath)
    elif filepath.suffix == ".parquet":
        df = _read_columns_parquet(filepath)
    else:
        raise ValueError(f"Không hỗ trợ định dạng file: {filepath.suffix}")

    n_raw_rows = len(df)
    print(f"[{month_label}] Đã đọc {n_raw_rows:,} row raw.")

    if not pd.api.types.is_datetime64_any_dtype(df["request_datetime"]):
        df["request_datetime"] = pd.to_datetime(df["request_datetime"])

    # Loại bỏ row thiếu dữ liệu cần thiết
    n_before = len(df)
    df = df.dropna(subset=["request_datetime", "pu_location_id"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[{month_label}] CẢNH BÁO: Loại {n_dropped:,} row do thiếu request_datetime/pu_location_id.")

    # Chuẩn hoá hour và pu_location_id
    df["hour"] = df["request_datetime"].dt.floor("h")
    df["pu_location_id"] = df["pu_location_id"].astype("int32")

    # Aggregate: đếm số request theo (zone, hour)
    agg = (
        df.groupby(["pu_location_id", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
    )
    agg["trip_count"] = agg["trip_count"].astype("int32")

    # Giải phóng raw dataframe
    del df
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"agg_{month_label}.csv"
    agg.to_csv(out_path, index=False)

    print(
        f"[{month_label}] Aggregate xong: {len(agg):,} zone-hour rows, "
        f"tổng demand = {agg['trip_count'].sum():,}. Đã lưu -> {out_path}"
    )

    del agg
    gc.collect()

    return out_path