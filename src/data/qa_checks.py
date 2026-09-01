"""
QA/QC FINAL DATA
================

Chỉ kiểm tra FINAL DATA đã được tạo bởi data pipeline.

Không đọc:
    - data/raw/
    - raw HVFHV files
    - monthly_agg/

Kiểm tra:
    1. top50_zones_frozen.json
    2. feature_table.csv
    3. folds/*.csv
    4. temporal leakage giữa các folds

Lưu ý:
    - Lag alignment KHÔNG kiểm tra ở đây.
    - Lag alignment vẫn thuộc trách nhiệm của split_folds.py.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

# qa_checks.py nằm tại:
#   src/data/qa_checks.py
#
# parents[0] = src/data
# parents[1] = src
# parents[2] = project root

REPO_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FROZEN_DIR = PROCESSED_DIR / "frozen"
FOLDS_DIR = REPO_ROOT / "data" / "folds"

TOP50_PATH = FROZEN_DIR / "top50_zones_frozen.json"
FEATURE_TABLE_PATH = PROCESSED_DIR / "feature_table.csv"
VARIANT_MAP_PATH = PROCESSED_DIR / "variant_feature_map.json"


# ============================================================
# EXPECTED CONFIGURATION
# ============================================================

N_TOP_ZONES = 50

EXPECTED_START = pd.Timestamp("2025-01-22 00:00:00")
EXPECTED_END = pd.Timestamp("2025-12-31 23:00:00")

ALL_LAGS = [
    "lag_1",
    "lag_24",
    "lag_168",
    "lag_336",
    "lag_504",
]

WEEKLY_LAGS = [
    "lag_168",
    "lag_336",
    "lag_504",
]

BASE_FEATURES = [
    "hour",
    "dayofweek",
    "is_holiday",
    "lag_1",
    "lag_24",
]

EXPECTED_VARIANTS = {
    "A": {
        "weekly_features": [
            "lag_168",
            "lag_336",
            "lag_504",
        ],
    },
    "B": {
        "weekly_features": [
            "median_lag_3w",
        ],
    },
    "C": {
        "weekly_features": [
            "lag_168",
            "median_lag_3w",
        ],
    },
}


# ============================================================
# EXPECTED TEMPORAL SPLITS
# ============================================================

# Khoảng [start, end)
#
# Ví dụ:
# ("2025-01-22", "2025-07-01")
# nghĩa là từ 22/01 00:00 đến 30/06 23:00.

SPLITS = {
    "hpo": {
        "train": (
            pd.Timestamp("2025-01-22"),
            pd.Timestamp("2025-07-01"),
        ),
        "eval": (
            pd.Timestamp("2025-07-01"),
            pd.Timestamp("2025-08-01"),
        ),
        "eval_kind": "val",
    },

    "fold1": {
        "train": (
            pd.Timestamp("2025-01-22"),
            pd.Timestamp("2025-08-01"),
        ),
        "eval": (
            pd.Timestamp("2025-08-01"),
            pd.Timestamp("2025-09-01"),
        ),
        "eval_kind": "val",
    },

    "fold2": {
        "train": (
            pd.Timestamp("2025-01-22"),
            pd.Timestamp("2025-09-01"),
        ),
        "eval": (
            pd.Timestamp("2025-09-01"),
            pd.Timestamp("2025-10-01"),
        ),
        "eval_kind": "val",
    },

    "fold3": {
        "train": (
            pd.Timestamp("2025-01-22"),
            pd.Timestamp("2025-10-01"),
        ),
        "eval": (
            pd.Timestamp("2025-10-01"),
            pd.Timestamp("2025-11-01"),
        ),
        "eval_kind": "val",
    },

    "fold4": {
        "train": (
            pd.Timestamp("2025-01-22"),
            pd.Timestamp("2025-11-01"),
        ),
        "eval": (
            pd.Timestamp("2025-11-01"),
            pd.Timestamp("2025-12-01"),
        ),
        "eval_kind": "val",
    },

    "final_test": {
        "train": (
            pd.Timestamp("2025-01-22"),
            pd.Timestamp("2025-12-01"),
        ),
        "eval": (
            pd.Timestamp("2025-12-01"),
            pd.Timestamp("2026-01-01"),
        ),
        "eval_kind": "test",
    },
}


# ============================================================
# HELPERS
# ============================================================

def fail(message: str):
    """Dừng QA ngay lập tức."""
    raise AssertionError(message)


def check_exists(path: Path, description: str):
    """Kiểm tra file tồn tại."""
    if not path.exists():
        fail(
            f"THIẾU {description}:\n"
            f"  {path}"
        )


def print_header(title: str):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# 1. TOP-50 FROZEN ZONES
# ============================================================

def check_top50_frozen():
    """
    Kiểm tra top50_zones_frozen.json.

    Kiểm tra:
        - file tồn tại
        - zone_ids tồn tại
        - đúng 50 zone
        - không duplicate
        - zone IDs là integer
        - selection period đúng Jan-Jun 2025
        - selection rule tồn tại
        - zone_total_demand có đủ 50 zone
        - zone_ids khớp với keys của zone_total_demand
    """

    print_header("1. TOP-50 FROZEN ZONES QA")

    check_exists(
        TOP50_PATH,
        "top50_zones_frozen.json",
    )

    with open(
        TOP50_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        record = json.load(f)

    # --------------------------------------------------------
    # zone_ids
    # --------------------------------------------------------

    if "zone_ids" not in record:
        fail(
            "top50_zones_frozen.json không có key 'zone_ids'."
        )

    zone_ids = record["zone_ids"]

    if not isinstance(zone_ids, list):
        fail("'zone_ids' phải là list.")

    if len(zone_ids) != N_TOP_ZONES:
        fail(
            f"Top zone có {len(zone_ids)} zone, "
            f"kỳ vọng {N_TOP_ZONES}."
        )

    if len(set(zone_ids)) != N_TOP_ZONES:
        fail(
            "zone_ids chứa zone bị duplicate."
        )

    # --------------------------------------------------------
    # Zone IDs
    # --------------------------------------------------------

    try:
        zone_ids_int = [int(z) for z in zone_ids]
    except Exception:
        fail(
            "Có zone ID không thể convert sang integer."
        )

    if any(z <= 0 for z in zone_ids_int):
        fail(
            "Có PULocationID <= 0 trong frozen zone list."
        )

    # --------------------------------------------------------
    # Selection period
    # --------------------------------------------------------

    selection_period = record.get(
        "selection_period"
    )

    expected_period = (
        "2025-01-01 to 2025-06-30"
    )

    if selection_period != expected_period:
        fail(
            "selection_period không đúng.\n"
            f"Observed : {selection_period}\n"
            f"Expected : {expected_period}"
        )

    # --------------------------------------------------------
    # Number of zones
    # --------------------------------------------------------

    if record.get("n_zones") != N_TOP_ZONES:
        fail(
            f"n_zones = {record.get('n_zones')}, "
            f"kỳ vọng {N_TOP_ZONES}."
        )

    # --------------------------------------------------------
    # Selection rule
    # --------------------------------------------------------

    selection_rule = record.get(
        "selection_rule",
        ""
    )

    if "Jan-Jun" not in selection_rule:
        fail(
            "selection_rule không xác nhận selection "
            "dựa trên Jan-Jun."
        )

    if "demand" not in selection_rule.lower():
        fail(
            "selection_rule không đề cập demand."
        )

    # --------------------------------------------------------
    # zone_total_demand
    # --------------------------------------------------------

    if "zone_total_demand" not in record:
        fail(
            "Không có 'zone_total_demand'."
        )

    zone_total_demand = record[
        "zone_total_demand"
    ]

    if len(zone_total_demand) != N_TOP_ZONES:
        fail(
            "zone_total_demand không có đúng "
            f"{N_TOP_ZONES} zone."
        )

    demand_zone_ids = {
        int(z)
        for z in zone_total_demand.keys()
    }

    if set(zone_ids_int) != demand_zone_ids:
        fail(
            "zone_ids và zone_total_demand "
            "không chứa cùng một tập zone."
        )

    # --------------------------------------------------------
    # Demand values
    # --------------------------------------------------------

    for zone, demand in zone_total_demand.items():

        try:
            demand = int(demand)
        except Exception:
            fail(
                f"Demand của zone {zone} không phải số."
            )

        if demand < 0:
            fail(
                f"Demand âm tại zone {zone}: {demand}"
            )

    print(
        f"Frozen zones : {len(zone_ids_int)}"
    )

    print(
        f"Selection period : {selection_period}"
    )

    print(
        "zone_ids unique : PASS"
    )

    print(
        "zone_total_demand consistency : PASS"
    )

    print(
        "TOP-50 FROZEN QA PASS"
    )

    return zone_ids_int


# ============================================================
# 2. FEATURE TABLE
# ============================================================

def load_feature_table():
    """
    Load feature_table.csv.

    Chỉ đọc feature table cuối cùng.
    Không đọc raw hoặc monthly aggregation.
    """

    check_exists(
        FEATURE_TABLE_PATH,
        "feature_table.csv",
    )

    df = pd.read_csv(
        FEATURE_TABLE_PATH,
        parse_dates=["target_datetime"],
    )

    return df


def check_feature_table(zone_ids):
    """
    Kiểm tra feature_table.csv.

    Kiểm tra:
        - đúng 50 zone
        - zone đúng frozen Top-50
        - target_datetime
        - hourly frequency
        - duplicate zone-hour
        - demand
        - calendar features
        - lag columns
        - no NaN
        - median_lag_3w
        - không có feature ngoài expected core scope
    """

    print_header("2. FEATURE TABLE QA")

    df = load_feature_table()

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    required_columns = [
        "PULocationID",
        "target_datetime",
        "demand",
        "hour",
        "dayofweek",
        "is_holiday",
        "lag_1",
        "lag_24",
        "lag_168",
        "lag_336",
        "lag_504",
        "median_lag_3w",
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:
        fail(
            "Feature table thiếu columns:\n"
            + "\n".join(
                f"  - {c}" for c in missing
            )
        )

    # --------------------------------------------------------
    # Zone check
    # --------------------------------------------------------

    observed_zones = sorted(
        df["PULocationID"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    expected_zones = sorted(
        zone_ids
    )

    if observed_zones != expected_zones:
        fail(
            "Zone trong feature_table không khớp "
            "frozen Top-50.\n"
            f"Expected: {expected_zones}\n"
            f"Observed : {observed_zones}"
        )

    print(
        f"Zones : {len(observed_zones)}"
    )

    print(
        "Zone vocabulary = frozen Top-50 : PASS"
    )

    # --------------------------------------------------------
    # Duplicate zone-hour
    # --------------------------------------------------------

    duplicate_count = df[
        ["PULocationID", "target_datetime"]
    ].duplicated().sum()

    if duplicate_count > 0:
        fail(
            f"Feature table có {duplicate_count:,} "
            "duplicate (zone, target_datetime)."
        )

    print(
        "Duplicate zone-hour : PASS"
    )

    # --------------------------------------------------------
    # Target timestamp range
    # --------------------------------------------------------

    actual_start = df[
        "target_datetime"
    ].min()

    actual_end = df[
        "target_datetime"
    ].max()

    if actual_start != EXPECTED_START:
        fail(
            "Feature table bắt đầu sai.\n"
            f"Observed : {actual_start}\n"
            f"Expected : {EXPECTED_START}"
        )

    if actual_end != EXPECTED_END:
        fail(
            "Feature table kết thúc sai.\n"
            f"Observed : {actual_end}\n"
            f"Expected : {EXPECTED_END}"
        )

    print(
        f"Target range : {actual_start} → {actual_end}"
    )

    # --------------------------------------------------------
    # Hour alignment
    # --------------------------------------------------------

    if (
        df["target_datetime"].dt.minute != 0
    ).any():
        fail(
            "Có target_datetime không nằm đúng đầu giờ."
        )

    if (
        df["target_datetime"].dt.second != 0
    ).any():
        fail(
            "Có target_datetime có second != 0."
        )

    print(
        "Target timestamp hour alignment : PASS"
    )

    # --------------------------------------------------------
    # Number of rows per zone
    # --------------------------------------------------------

    rows_per_zone = (
        df.groupby("PULocationID")
        .size()
    )

    expected_hours = 8256 #do đã cắt bỏ 404 giờ đầu

    if not (
        rows_per_zone == expected_hours
    ).all():

        bad = rows_per_zone[
            rows_per_zone != expected_hours
        ]

        fail(
            "Số hourly observations không đồng đều "
            "giữa các zone:\n"
            f"{bad.to_string()}"
        )

    print(
        f"Rows per zone : {expected_hours:,} : PASS"
    )

    expected_total_rows = (
        N_TOP_ZONES * expected_hours
    )

    if len(df) != expected_total_rows:
        fail(
            f"Feature table có {len(df):,} rows, "
            f"kỳ vọng {expected_total_rows:,}."
        )

    print(
        f"Total rows : {len(df):,} : PASS"
    )

    # --------------------------------------------------------
    # Hourly continuity
    # --------------------------------------------------------

    print(
        "Checking hourly continuity by zone..."
    )

    for zone, g in df.groupby(
        "PULocationID"
    ):

        times = (
            g["target_datetime"]
            .sort_values()
        )

        diffs = times.diff().dropna()

        if not (
            diffs == pd.Timedelta(hours=1)
        ).all():

            bad = diffs[
                diffs != pd.Timedelta(hours=1)
            ]

            fail(
                f"Zone {zone} có hourly gap/break:\n"
                f"{bad.head(10).to_string()}"
            )

    print(
        "Hourly continuity : PASS"
    )

    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    if df["demand"].isna().any():
        fail(
            "Demand vẫn còn NaN."
        )

    if (df["demand"] < 0).any():
        fail(
            "Demand có giá trị âm."
        )

    print(
        "Demand non-negative / no NaN : PASS"
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    if df["hour"].isna().any():
        fail(
            "hour chứa NaN."
        )

    if not (
        df["hour"]
        .astype(int)
        .between(0, 23)
    ).all():
        fail(
            "hour chứa giá trị ngoài [0, 23]."
        )

    expected_hour = (
        df["target_datetime"].dt.hour
    )

    if not (
        df["hour"].astype(int)
        .eq(expected_hour)
    ).all():
        fail(
            "hour không khớp target_datetime."
        )

    expected_dow = (
        df["target_datetime"].dt.dayofweek
    )

    if not (
        df["dayofweek"].astype(int)
        .eq(expected_dow)
    ).all():
        fail(
            "dayofweek không khớp target_datetime."
        )

    if not (
        df["is_holiday"]
        .isin([0, 1])
    ).all():
        fail(
            "is_holiday phải chỉ chứa 0/1."
        )

    print(
        "Calendar features : PASS"
    )

    # --------------------------------------------------------
    # Lag columns
    # --------------------------------------------------------

    for col in ALL_LAGS:

        if df[col].isna().any():
            fail(
                f"{col} vẫn còn NaN."
            )

        if (
            df[col] < 0
        ).any():
            fail(
                f"{col} có giá trị âm."
            )

    print(
        "Required lag columns / no NaN : PASS"
    )

    # --------------------------------------------------------
    # Median weekly lag
    # --------------------------------------------------------

    expected_median = df[
        WEEKLY_LAGS
    ].median(
        axis=1,
        skipna=False,
    )

    actual_median = df[
        "median_lag_3w"
    ]

    if not np.allclose(
        actual_median.to_numpy(),
        expected_median.to_numpy(),
        rtol=0,
        atol=0,
    ):
        mismatch = (
            actual_median
            != expected_median
        )

        fail(
            "median_lag_3w không bằng "
            "median(lag_168, lag_336, lag_504).\n"
            f"Số mismatch: {mismatch.sum():,}"
        )

    print(
        "median_lag_3w correctness : PASS"
    )

    # --------------------------------------------------------
    # Feature scope
    # --------------------------------------------------------

    forbidden_keywords = [
        "weather",
        "temperature",
        "precip",
        "snow",
        "neighbor",
        "borough",
        "hour_of_week",
        "dashboard",
        "deployment",
    ]

    suspicious = []

    for col in df.columns:

        col_lower = col.lower()

        for keyword in forbidden_keywords:

            if keyword in col_lower:
                suspicious.append(col)

    if suspicious:
        fail(
            "Phát hiện columns ngoài core scope:\n"
            + "\n".join(
                f"  - {c}"
                for c in sorted(set(suspicious))
            )
        )

    print(
        "Core feature scope : PASS"
    )

    print(
        "FEATURE TABLE QA PASS"
    )

    return df


# ============================================================
# 3. VARIANT MAP
# ============================================================

def check_variant_map():
    """
    Kiểm tra variant_feature_map.json.

    Không bắt buộc phải dùng trong training QA,
    nhưng kiểm tra rằng A/B/C được lưu đúng protocol.
    """

    print_header("3. VARIANT FEATURE MAP QA")

    check_exists(
        VARIANT_MAP_PATH,
        "variant_feature_map.json",
    )

    with open(
        VARIANT_MAP_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        record = json.load(f)

    if "base_features" not in record:
        fail(
            "variant_feature_map.json thiếu "
            "'base_features'."
        )

    if (
        record["base_features"]
        != BASE_FEATURES
    ):
        fail(
            "base_features không đúng.\n"
            f"Observed: {record['base_features']}\n"
            f"Expected: {BASE_FEATURES}"
        )

    if "variants" not in record:
        fail(
            "variant_feature_map.json thiếu "
            "'variants'."
        )

    variants = record["variants"]

    for variant_name, expected in EXPECTED_VARIANTS.items():

        if variant_name not in variants:
            fail(
                f"Thiếu Variant {variant_name}."
            )

        actual_features = variants[
            variant_name
        ].get("weekly_features")

        if actual_features != expected[
            "weekly_features"
        ]:
            fail(
                f"Variant {variant_name} sai weekly features.\n"
                f"Observed: {actual_features}\n"
                f"Expected: {expected['weekly_features']}"
            )

    print(
        "Variant A : lag_168 + lag_336 + lag_504 : PASS"
    )

    print(
        "Variant B : median_lag_3w : PASS"
    )

    print(
        "Variant C : lag_168 + median_lag_3w : PASS"
    )

    print(
        "VARIANT FEATURE MAP QA PASS"
    )


# ============================================================
# 4. LOAD / CHECK ONE FOLD
# ============================================================

def get_fold_paths(
    split_name: str,
    eval_kind: str,
):
    """
    Trả về train/eval path của một fold.
    """

    split_dir = FOLDS_DIR / split_name

    train_path = (
        split_dir / "train.csv"
    )

    eval_path = (
        split_dir / f"{eval_kind}.csv"
    )

    return train_path, eval_path


def load_fold_data(
    split_name: str,
    split_cfg: dict,
):
    """
    Load train + validation/test của một fold.
    """

    train_path, eval_path = get_fold_paths(
        split_name,
        split_cfg["eval_kind"],
    )

    check_exists(
        train_path,
        f"{split_name}/train.csv",
    )

    check_exists(
        eval_path,
        f"{split_name}/{split_cfg['eval_kind']}.csv",
    )

    train = pd.read_csv(
        train_path,
        parse_dates=["target_datetime"],
    )

    eval_df = pd.read_csv(
        eval_path,
        parse_dates=["target_datetime"],
    )

    return train, eval_df


# ============================================================
# 5. FOLD STRUCTURE QA
# ============================================================

def check_fold_structure(
    zone_ids,
):
    """
    Kiểm tra từng HPO/Fold/Final test.

    Kiểm tra:
        - file tồn tại
        - đúng 50 zone
        - zone vocabulary giống frozen Top-50
        - target datetime
        - train/eval không overlap
        - train/eval đúng thời gian
        - target range đúng
        - feature columns nhất quán
    """

    print_header("4. FOLD DATA QA")

    fold_data = {}

    expected_feature_columns = None

    for split_name, split_cfg in SPLITS.items():

        print(
            f"\n--- {split_name.upper()} ---"
        )

        train, eval_df = load_fold_data(
            split_name,
            split_cfg,
        )

        # ----------------------------------------------------
        # Basic non-empty
        # ----------------------------------------------------

        if len(train) == 0:
            fail(
                f"[{split_name}] train.csv rỗng."
            )

        if len(eval_df) == 0:
            fail(
                f"[{split_name}] eval/test rỗng."
            )

        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required = {
            "PULocationID",
            "target_datetime",
            "demand",
        }

        for dataset_name, data in [
            ("train", train),
            ("eval/test", eval_df),
        ]:

            missing = required - set(
                data.columns
            )

            if missing:
                fail(
                    f"[{split_name}/{dataset_name}] "
                    f"thiếu columns: {sorted(missing)}"
                )

        # ----------------------------------------------------
        # Feature column consistency
        # ----------------------------------------------------

        train_columns = list(
            train.columns
        )

        eval_columns = list(
            eval_df.columns
        )

        if train_columns != eval_columns:
            fail(
                f"[{split_name}] train và eval/test "
                "không có cùng columns."
            )

        if expected_feature_columns is None:
            expected_feature_columns = (
                train_columns
            )

        elif train_columns != expected_feature_columns:
            fail(
                f"[{split_name}] columns không nhất quán "
                "với các fold khác."
            )

        # ----------------------------------------------------
        # Zone vocabulary
        # ----------------------------------------------------

        train_zones = sorted(
            train["PULocationID"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        eval_zones = sorted(
            eval_df["PULocationID"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        expected_zones = sorted(
            zone_ids
        )

        if train_zones != expected_zones:
            fail(
                f"[{split_name}] train zones "
                "không khớp frozen Top-50."
            )

        if eval_zones != expected_zones:
            fail(
                f"[{split_name}] eval/test zones "
                "không khớp frozen Top-50."
            )

        # ----------------------------------------------------
        # Datetime range
        # ----------------------------------------------------

        train_start_expected = split_cfg[
            "train"
        ][0]

        train_end_expected = split_cfg[
            "train"
        ][1]

        eval_start_expected = split_cfg[
            "eval"
        ][0]

        eval_end_expected = split_cfg[
            "eval"
        ][1]

        train_min = train[
            "target_datetime"
        ].min()

        train_max = train[
            "target_datetime"
        ].max()

        eval_min = eval_df[
            "target_datetime"
        ].min()

        eval_max = eval_df[
            "target_datetime"
        ].max()

        if train_min != train_start_expected:
            fail(
                f"[{split_name}] train bắt đầu sai.\n"
                f"Observed : {train_min}\n"
                f"Expected : {train_start_expected}"
            )

        expected_train_max = (
            train_end_expected
            - pd.Timedelta(hours=1)
        )

        if train_max != expected_train_max:
            fail(
                f"[{split_name}] train kết thúc sai.\n"
                f"Observed : {train_max}\n"
                f"Expected : {expected_train_max}"
            )

        if eval_min != eval_start_expected:
            fail(
                f"[{split_name}] eval/test bắt đầu sai.\n"
                f"Observed : {eval_min}\n"
                f"Expected : {eval_start_expected}"
            )

        expected_eval_max = (
            eval_end_expected
            - pd.Timedelta(hours=1)
        )

        if eval_max != expected_eval_max:
            fail(
                f"[{split_name}] eval/test kết thúc sai.\n"
                f"Observed : {eval_max}\n"
                f"Expected : {expected_eval_max}"
            )

        # ----------------------------------------------------
        # Train / eval temporal separation
        # ----------------------------------------------------

        if train_max >= eval_min:
            fail(
                f"[{split_name}] train và eval/test "
                "có temporal overlap."
            )

        gap = (
            eval_min - train_max
        )

        if gap != pd.Timedelta(hours=1):
            fail(
                f"[{split_name}] train/eval không liền kề.\n"
                f"Gap: {gap}"
            )

        # ----------------------------------------------------
        # Duplicate zone-hour within datasets
        # ----------------------------------------------------

        for dataset_name, data in [
            ("train", train),
            ("eval/test", eval_df),
        ]:

            duplicate_count = data[
                [
                    "PULocationID",
                    "target_datetime",
                ]
            ].duplicated().sum()

            if duplicate_count > 0:
                fail(
                    f"[{split_name}/{dataset_name}] "
                    f"có {duplicate_count:,} duplicate "
                    "(zone, target_datetime)."
                )

        # ----------------------------------------------------
        # Demand
        # ----------------------------------------------------

        for dataset_name, data in [
            ("train", train),
            ("eval/test", eval_df),
        ]:

            if data["demand"].isna().any():
                fail(
                    f"[{split_name}/{dataset_name}] "
                    "demand có NaN."
                )

            if (
                data["demand"] < 0
            ).any():
                fail(
                    f"[{split_name}/{dataset_name}] "
                    "demand có giá trị âm."
                )

        print(
            f"Train : {len(train):,} rows "
            f"({train_min} → {train_max})"
        )

        print(
            f"Eval  : {len(eval_df):,} rows "
            f"({eval_min} → {eval_max})"
        )

        print(
            "Zone vocabulary : PASS"
        )

        print(
            "Train/eval temporal separation : PASS"
        )

        fold_data[split_name] = {
            "train": train,
            "eval": eval_df,
        }

    print(
        "\nFOLD STRUCTURE QA PASS"
    )

    return fold_data


# ============================================================
# 6. TEMPORAL LEAKAGE BETWEEN FOLDS
# ============================================================

def check_temporal_leakage(
    fold_data,
):
    """
    Kiểm tra temporal leakage giữa HPO/Fold 1-4/Final test.

    QUAN TRỌNG:

    Expanding-window design có chủ đích:

        HPO val July
             ↓
        Fold 1 train includes July

    Điều này KHÔNG phải leakage.

    Ta chỉ coi là leakage nếu một fold sử dụng
    observation nằm SAU evaluation period của chính nó
    trong training.

    Ngoài ra:
        - validation windows phải theo thứ tự thời gian
        - validation windows không overlap
        - December final test không xuất hiện trong
          bất kỳ train/validation trước final test
    """

    print_header(
        "5. TEMPORAL LEAKAGE BETWEEN FOLDS QA"
    )

    ordered_splits = [
        "hpo",
        "fold1",
        "fold2",
        "fold3",
        "fold4",
        "final_test",
    ]

    # --------------------------------------------------------
    # 6.1 Check each fold individually
    # --------------------------------------------------------

    for split_name in ordered_splits:

        cfg = SPLITS[split_name]

        train = fold_data[
            split_name
        ]["train"]

        eval_df = fold_data[
            split_name
        ]["eval"]

        train_max = train[
            "target_datetime"
        ].max()

        eval_min = eval_df[
            "target_datetime"
        ].min()

        if train_max >= eval_min:
            fail(
                f"[{split_name}] "
                "Temporal leakage: train chứa "
                "observation >= eval start."
            )

        print(
            f"[{split_name}] "
            f"train max={train_max}, "
            f"eval min={eval_min} : PASS"
        )

    # --------------------------------------------------------
    # 6.2 Validation/test windows must not overlap
    # --------------------------------------------------------

    eval_intervals = []

    for split_name in ordered_splits:

        cfg = SPLITS[split_name]

        eval_start = cfg["eval"][0]
        eval_end = cfg["eval"][1]

        eval_intervals.append(
            (
                split_name,
                eval_start,
                eval_end,
            )
        )

    for i in range(
        len(eval_intervals)
    ):

        name_i, start_i, end_i = (
            eval_intervals[i]
        )

        for j in range(i + 1, len(
            eval_intervals
        )):

            name_j, start_j, end_j = (
                eval_intervals[j]
            )

            overlap = (
                start_i < end_j
                and start_j < end_i
            )

            if overlap:
                fail(
                    "Evaluation windows overlap:\n"
                    f"{name_i}: {start_i} → {end_i}\n"
                    f"{name_j}: {start_j} → {end_j}"
                )

    print(
        "Validation/test windows non-overlapping : PASS"
    )

    # --------------------------------------------------------
    # 6.3 Explicitly check final December isolation
    # --------------------------------------------------------

    final_test = fold_data[
        "final_test"
    ]["eval"]

    final_test_start = (
        final_test["target_datetime"].min()
    )

    final_test_end = (
        final_test["target_datetime"].max()
    )

    expected_final_start = pd.Timestamp(
        "2025-12-01 00:00:00"
    )

    expected_final_end = pd.Timestamp(
        "2025-12-31 23:00:00"
    )

    if final_test_start != expected_final_start:
        fail(
            "Final test không bắt đầu từ "
            "2025-12-01."
        )

    if final_test_end != expected_final_end:
        fail(
            "Final test không kết thúc "
            "2025-12-31 23:00."
        )

    # --------------------------------------------------------
    # Check December absent from all earlier datasets
    # --------------------------------------------------------

    december_start = pd.Timestamp(
        "2025-12-01 00:00:00"
    )

    earlier_splits = [
        "hpo",
        "fold1",
        "fold2",
        "fold3",
        "fold4",
    ]

    for split_name in earlier_splits:

        train = fold_data[
            split_name
        ]["train"]

        eval_df = fold_data[
            split_name
        ]["eval"]

        train_december = (
            train["target_datetime"]
            >= december_start
        ).sum()

        eval_december = (
            eval_df["target_datetime"]
            >= december_start
        ).sum()

        if (
            train_december > 0
            or eval_december > 0
        ):
            fail(
                f"[{split_name}] December observations "
                "xuất hiện trước Final Test.\n"
                f"Train December rows: {train_december:,}\n"
                f"Eval December rows : {eval_december:,}"
            )

    print(
        "Final December isolation : PASS"
    )

    # --------------------------------------------------------
    # 6.4 Check expanding-window design
    # --------------------------------------------------------

    previous_train_end = None

    for split_name in ordered_splits:

        train_start = SPLITS[
            split_name
        ]["train"][0]

        train_end = SPLITS[
            split_name
        ]["train"][1]

        if previous_train_end is not None:

            if train_end < previous_train_end:
                fail(
                    f"[{split_name}] train window "
                    "không mở rộng theo thời gian."
                )

        previous_train_end = train_end

    print(
        "Expanding training windows : PASS"
    )

    # --------------------------------------------------------
    # Important explanation
    # --------------------------------------------------------

    print(
        "\nNOTE:"
    )

    print(
        "HPO/Fold validation data xuất hiện trong "
        "training data của fold kế tiếp là EXPECTED "
        "under the expanding-window design."
    )

    print(
        "Ví dụ: July là HPO validation nhưng July "
        "được phép xuất hiện trong Fold 1 training."
    )

    print(
        "Điều này không được đánh dấu là leakage."
    )

    print(
        "\nTEMPORAL LEAKAGE QA PASS"
    )


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

def main():

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL DATA QA/QC"
    )

    print(
        "=" * 75
    )

    print(
        f"\nRepository:\n"
        f"{REPO_ROOT}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "- Raw data is NOT checked here."
    )

    print(
        "- Monthly aggregation is NOT checked here."
    )

    print(
        "- Lag alignment is intentionally NOT checked here."
    )

    print(
        "- Lag alignment remains the responsibility "
        "of split_folds.py."
    )

    # --------------------------------------------------------
    # 1. Top 50
    # --------------------------------------------------------

    zone_ids = check_top50_frozen()

    # --------------------------------------------------------
    # 2. Feature table
    # --------------------------------------------------------

    feature_table = check_feature_table(
        zone_ids
    )

    del feature_table

    # --------------------------------------------------------
    # 3. Variant map
    # --------------------------------------------------------

    check_variant_map()

    # --------------------------------------------------------
    # 4. Folds
    # --------------------------------------------------------

    fold_data = check_fold_structure(
        zone_ids
    )

    # --------------------------------------------------------
    # 5. Temporal leakage
    # --------------------------------------------------------

    check_temporal_leakage(
        fold_data
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ALL FINAL DATA QA/QC CHECKS PASS"
    )

    print(
        "=" * 75
    )

    print(
        "\nChecked:"
    )

    print(
        "1. top50_zones_frozen.json"
    )

    print(
        "2. feature_table.csv"
    )

    print(
        "3. variant_feature_map.json"
    )

    print(
        "4. hpo/fold1/fold2/fold3/fold4/final_test"
    )

    print(
        "5. Temporal leakage / final-test isolation"
    )

    print(
        "\nLag alignment: NOT checked here; "
        "run split_folds.py for that QA."
    )


if __name__ == "__main__":

    try:
        main()

    except AssertionError as exc:

        print(
            "\n"
            + "=" * 75
        )

        print(
            "QA/QC FAILED"
        )

        print(
            "=" * 75
        )

        print(
            f"\n{exc}\n"
        )

        sys.exit(1)