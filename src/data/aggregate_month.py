"""Tổng hợp một file HVFHV Parquet chính thức theo pickup zone và hour."""

from pathlib import Path
import pandas as pd

def aggregate_month(
    filepath: Path,
    month_label: str,
    output_dir: Path,
    request_datetime_column: str,
    pickup_zone_column: str,
    normalized_zone_column: str,
) -> Path:
    """Ghi một monthly aggregate sau khi validate raw schema."""
    if filepath.suffix != ".parquet":
        raise ValueError(f"Input phải là file Parquet, nhưng nhận được: {filepath}")

    output_path = output_dir / f"agg_{month_label}.csv"
    if output_path.exists():
        raise FileExistsError(f"Từ chối ghi đè aggregate đã tồn tại: {output_path}")

    raw_columns = [request_datetime_column, pickup_zone_column]
    df = pd.read_parquet(filepath, columns=raw_columns)
    if df[raw_columns].isna().any().any():
        missing_counts = df[raw_columns].isna().sum()
        raise ValueError(
            f"{filepath} chứa giá trị bắt buộc bị thiếu: {missing_counts.to_dict()}"
        )

    df = df.rename(columns={pickup_zone_column: normalized_zone_column})
    df[request_datetime_column] = pd.to_datetime(df[request_datetime_column], errors="raise")
    df["hour"] = df[request_datetime_column].dt.floor("h")
    df[normalized_zone_column] = df[normalized_zone_column].astype("int32")

    agg = (
        df.groupby([normalized_zone_column, "hour"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
    )
    agg["trip_count"] = agg["trip_count"].astype("int32")

    output_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)

    print(
        f"[{month_label}] raw row={len(df):,}, zone-hour row={len(agg):,}, "
        f"demand={agg['trip_count'].sum():,}, output={output_path}"
    )
    return output_path
