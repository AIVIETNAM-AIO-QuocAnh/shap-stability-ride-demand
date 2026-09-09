"""Đọc và cache dữ liệu cho dashboard.

Mọi số liệu đều lấy lại từ artifact đã có trong `results/`; module này không train,
không sửa gì trong core pipeline, và dùng lại đúng các hàm metric của
`src/pipeline/metrics.py` để con số khớp với `results/stats/`.
"""

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.load_config import PROJECT_ROOT, load_data_config, load_model_config
from src.pipeline.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.utilities import load_frozen_zone_ids


GEOJSON_PATH = PROJECT_ROOT / "src" / "dashboard" / "zones_50.geojson"
VARIANTS = ("A", "B", "C")
MODELS = ("xgboost", "lightgbm")
HPO_SPLIT = "hpo"


@lru_cache(maxsize=1)
def evaluation_windows() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Cửa sổ evaluation của từng core fold, đọc từ `configs/data.yaml`.

    Bỏ split `hpo` vì nó chỉ dùng để tune, không nằm trong 30 core run.
    """
    splits = load_data_config()["splits"]
    windows = {}
    for name, split in splits.items():
        if name == HPO_SPLIT:
            continue
        windows[name] = (
            pd.Timestamp(split["eval_start"]),
            pd.Timestamp(split["eval_end_exclusive"]),
        )
    return dict(sorted(windows.items(), key=lambda item: item[1][0]))


@lru_cache(maxsize=1)
def selectable_date_range() -> tuple[date, date]:
    """Ngày đầu và ngày cuối người dùng được phép chọn."""
    windows = evaluation_windows()
    starts = [start for start, _ in windows.values()]
    ends = [end for _, end in windows.values()]
    return min(starts).date(), (max(ends) - pd.Timedelta(hours=1)).date()


def fold_for_date(selected_date: date) -> str:
    """Suy ra fold chứa ngày đang chọn.

    Mỗi fold đánh giá đúng một tháng nên ánh xạ này là 1-1; người dùng chọn ngày,
    không phải chọn fold.
    """
    timestamp = pd.Timestamp(selected_date)
    for fold, (start, end_exclusive) in evaluation_windows().items():
        if start <= timestamp < end_exclusive:
            return fold
    first, last = selectable_date_range()
    raise ValueError(f"Ngày {selected_date} nằm ngoài dải đánh giá {first} đến {last}")


@lru_cache(maxsize=1)
def holiday_dates() -> frozenset[date]:
    """Ngày lễ liên bang Mỹ, dùng đúng lịch mà `build_panel.py` đã dùng cho `is_holiday`."""
    panel = load_data_config()["panel"]
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=panel["start"], end=panel["end_exclusive"])
    return frozenset(holidays.date)


def holiday_name(selected_date: date) -> str | None:
    """Tên ngày lễ nếu ngày đang chọn là lễ liên bang, ngược lại trả về None."""
    if selected_date not in holiday_dates():
        return None
    panel = load_data_config()["panel"]
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(
        start=panel["start"], end=panel["end_exclusive"], return_name=True
    )
    matched = holidays[holidays.index.date == selected_date]
    return str(matched.iloc[0]) if len(matched) else "Ngày lễ liên bang"


@st.cache_data(show_spinner=False)
def load_zone_geojson() -> dict:
    """GeoJSON WGS84 của 50 frozen zone, sinh bởi `build_zone_geojson.py`."""
    if not GEOJSON_PATH.is_file():
        raise FileNotFoundError(
            f"Chưa có {GEOJSON_PATH.name}. Chạy trước: python -m src.dashboard.build_zone_geojson"
        )
    with GEOJSON_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


@st.cache_data(show_spinner=False)
def load_zone_catalog() -> pd.DataFrame:
    """Bảng tra cứu 50 zone: id, tên zone, borough."""
    features = load_zone_geojson()["features"]
    catalog = pd.DataFrame(
        [feature["properties"] for feature in features]
    ).sort_values("zone_name", ignore_index=True)
    expected = len(load_frozen_zone_ids(load_data_config()))
    if len(catalog) != expected:
        raise ValueError(f"Zone catalog có {len(catalog)} zone, cần đúng {expected}")
    return catalog


@lru_cache(maxsize=len(VARIANTS))
def load_variant_features(variant: str) -> tuple[str, ...]:
    """Weekly feature của một variant, đọc từ config đã khoá.

    Dashboard dùng để in chú thích variant; lấy động thay vì viết cứng nên chú thích
    không thể nói một đằng trong khi model chạy một nẻo.
    """
    variants = load_data_config()["panel"]["variants"]
    if variant not in variants:
        raise ValueError(f"Variant không xác định '{variant}'; kỳ vọng một trong {sorted(variants)}")
    return tuple(variants[variant]["weekly_features"])


@st.cache_data(show_spinner=False)
def load_stats_table(name: str) -> pd.DataFrame:
    """Đọc một bảng tổng hợp trong `results/stats/`.

    Đây là artifact do `run.py all` và `src/analysis/run_stats.py` sinh ra, trang so sánh
    thí nghiệm chỉ đọc lại chứ không tính lại, để số trên dashboard luôn khớp báo cáo.
    """
    path = load_model_config()["paths"]["results_stats"] / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có {path.name}. Chạy trước: python run.py all && python -m src.analysis.run_stats"
        )
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_correlation_summary() -> pd.DataFrame:
    """Pearson correlation giữa ba weekly lag, tổng hợp trên 50 zone (proposal mục 2.2)."""
    path = load_data_config()["paths"]["correlation_summary"]
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {path}")
    return pd.read_csv(path)


def stats_plot_path(name: str) -> Path:
    """Đường dẫn tới một hình trong `results/stats/plots/`."""
    return load_model_config()["paths"]["results_stats"] / "plots" / name


def prediction_path(variant: str, fold: str, model: str) -> Path:
    """Đường dẫn tới artifact prediction của một tổ hợp."""
    return load_model_config()["paths"]["results"] / variant / fold / model / "y_pred.csv"


@st.cache_data(show_spinner=False)
def load_predictions(variant: str, fold: str, model: str) -> pd.DataFrame:
    """Đọc `y_pred.csv` của một tổ hợp và bổ sung cột phái sinh cho dashboard.

    Cache theo (variant, fold, model) nên đổi ngày trong cùng tháng không phải đọc lại file.
    """
    path = prediction_path(variant, fold, model)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy prediction artifact: {path}")
    frame = pd.read_csv(path, parse_dates=["target_datetime"])
    expected_columns = ["pu_location_id", "target_datetime", "y_true", "y_pred"]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{path} phải có đúng các cột {expected_columns}")
    frame["error"] = frame["y_pred"] - frame["y_true"]
    frame["absolute_error"] = frame["error"].abs()
    frame["date"] = frame["target_datetime"].dt.date
    frame["hour"] = frame["target_datetime"].dt.hour
    return frame


@st.cache_data(show_spinner=False)
def load_weekly_features(fold: str, variant: str) -> pd.DataFrame:
    """Giá trị weekly lag của từng row đánh giá trong một fold.

    Đọc thẳng file fold mà Pipeline đã dùng để train, nên cột lag ở đây đúng bằng cái
    model nhìn thấy. Dùng để vẽ chồng lên biểu đồ 24 giờ.
    """
    data_config = load_data_config()
    fold_dir = data_config["paths"]["folds_dir"] / fold
    path = fold_dir / ("test.csv" if fold == "final_test" else "val.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy fold file: {path}")
    features = list(load_variant_features(variant))
    frame = pd.read_csv(path, parse_dates=["target_datetime"])
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} thiếu cột weekly feature: {missing}")
    frame = frame[["pu_location_id", "target_datetime", *features]].copy()
    frame["date"] = frame["target_datetime"].dt.date
    frame["hour"] = frame["target_datetime"].dt.hour
    return frame


def filter_predictions(
    frame: pd.DataFrame,
    selected_date: date | None = None,
    selected_hour: int | None = None,
    zone_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Lọc prediction theo ngày, giờ và tập zone đang chọn."""
    filtered = frame
    if selected_date is not None:
        filtered = filtered[filtered["date"] == selected_date]
    if selected_hour is not None:
        filtered = filtered[filtered["hour"] == selected_hour]
    if zone_ids:
        filtered = filtered[filtered["pu_location_id"].isin(zone_ids)]
    return filtered


def compute_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    """Tính MAE, RMSE, WAPE bằng đúng hàm của `src/pipeline/metrics.py`.

    Bộ metric theo proposal mục 2.5; WAPE trả về phần trăm. Trả về None cho WAPE khi
    tổng demand thực bằng 0, vì lúc đó WAPE không xác định.
    """
    if frame.empty:
        return {"mae": None, "rmse": None, "wape": None, "n_rows": 0}
    y_true = frame["y_true"]
    y_pred = frame["y_pred"]
    try:
        wape = weighted_absolute_percentage_error(y_true, y_pred)
    except ValueError:
        wape = None
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "wape": wape,
        "n_rows": int(len(frame)),
    }


def attach_zone_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Gắn tên zone và borough vào bảng prediction."""
    return frame.merge(load_zone_catalog(), on="pu_location_id", how="left")
