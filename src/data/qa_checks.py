"""QA/QC checks for the final processed data.

This module validates artifacts produced by the data pipeline:
    1. top50_zones_frozen.json
    2. feature_table.csv
    3. variant_feature_map.json
    4. folds/*.csv (hpo, fold1-4, final_test)
    5. Temporal leakage across folds

It does not read raw data, HVFHV files, or monthly_agg/.

Lag alignment is already covered in src/data/split_folds.py by
verify_lag_alignment(). It checks the configured lag features
against demand at the corresponding t-N timestamps.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# Paths
# qa_checks.py lives at src/data/qa_checks.py, so parents[2] is the
# project root.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FROZEN_DIR = PROCESSED_DIR / "frozen"
FOLDS_DIR = REPO_ROOT / "data" / "folds"

TOP50_PATH = FROZEN_DIR / "top50_zones_frozen.json"
FEATURE_TABLE_PATH = PROCESSED_DIR / "feature_table.csv"
VARIANT_MAP_PATH = PROCESSED_DIR / "variant_feature_map.json"


# Expected configuration
N_TOP_ZONES = 50
EXPECTED_START = pd.Timestamp("2025-01-22 00:00:00")
EXPECTED_END = pd.Timestamp("2025-12-31 23:00:00")

ALL_LAGS = ["lag_1", "lag_24", "lag_168", "lag_336", "lag_504"]
WEEKLY_LAGS = ["lag_168", "lag_336", "lag_504"]
BASE_FEATURES = ["hour", "dayofweek", "is_holiday", "lag_1", "lag_24"]

EXPECTED_VARIANTS = {
    "A": {"weekly_features": ["lag_168", "lag_336", "lag_504"]},
    "B": {"weekly_features": ["median_lag_3w"]},
    "C": {"weekly_features": ["lag_168", "median_lag_3w"]},
}

# Expected temporal splits, half-open interval [start, end).
SPLITS = {
    "hpo": {
        "train": (pd.Timestamp("2025-01-22"), pd.Timestamp("2025-07-01")),
        "eval": (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")),
        "eval_kind": "val",
    },
    "fold1": {
        "train": (pd.Timestamp("2025-01-22"), pd.Timestamp("2025-08-01")),
        "eval": (pd.Timestamp("2025-08-01"), pd.Timestamp("2025-09-01")),
        "eval_kind": "val",
    },
    "fold2": {
        "train": (pd.Timestamp("2025-01-22"), pd.Timestamp("2025-09-01")),
        "eval": (pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")),
        "eval_kind": "val",
    },
    "fold3": {
        "train": (pd.Timestamp("2025-01-22"), pd.Timestamp("2025-10-01")),
        "eval": (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-01")),
        "eval_kind": "val",
    },
    "fold4": {
        "train": (pd.Timestamp("2025-01-22"), pd.Timestamp("2025-11-01")),
        "eval": (pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-01")),
        "eval_kind": "val",
    },
    "final_test": {
        "train": (pd.Timestamp("2025-01-22"), pd.Timestamp("2025-12-01")),
        "eval": (pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")),
        "eval_kind": "test",
    },
}


def fail(message):
    """Raise an AssertionError to stop the QA run immediately."""
    raise AssertionError(message)


def check_exists(path, description):
    """Assert that a required file exists on disk."""
    if not path.exists():
        fail(f"MISSING {description}:\n  {path}")


def print_header(title):
    """Print a section banner to stdout."""
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def check_top50_frozen():
    """Validate top50_zones_frozen.json.

    Checks the file exists, contains exactly 50 unique positive
    integer zone IDs, the selection period/rule metadata are correct,
    and zone_total_demand matches zone_ids with non-negative values.
    """
    print_header("1. TOP-50 FROZEN ZONES QA")
    check_exists(TOP50_PATH, "top50_zones_frozen.json")

    with open(TOP50_PATH, "r", encoding="utf-8") as f:
        record = json.load(f)

    if "zone_ids" not in record:
        fail("top50_zones_frozen.json is missing 'zone_ids'.")

    zone_ids = record["zone_ids"]
    if not isinstance(zone_ids, list):
        fail("'zone_ids' must be a list.")
    if len(zone_ids) != N_TOP_ZONES:
        fail(f"Got {len(zone_ids)} zones, expected {N_TOP_ZONES}.")
    if len(set(zone_ids)) != N_TOP_ZONES:
        fail("zone_ids contains duplicate zones.")

    try:
        zone_ids_int = [int(z) for z in zone_ids]
    except (TypeError, ValueError):
        fail("Some zone ID cannot be converted to int.")

    if any(z <= 0 for z in zone_ids_int):
        fail("Frozen zone list contains a PULocationID <= 0.")

    expected_period = "2025-01-01 to 2025-06-30"
    selection_period = record.get("selection_period")
    if selection_period != expected_period:
        fail(
            "selection_period is incorrect.\n"
            f"Observed : {selection_period}\n"
            f"Expected : {expected_period}"
        )

    if record.get("n_zones") != N_TOP_ZONES:
        fail(f"n_zones = {record.get('n_zones')}, expected {N_TOP_ZONES}.")

    selection_rule = record.get("selection_rule", "")
    if "Jan-Jun" not in selection_rule:
        fail("selection_rule does not reference Jan-Jun.")
    if "demand" not in selection_rule.lower():
        fail("selection_rule does not mention demand.")

    if "zone_total_demand" not in record:
        fail("Missing 'zone_total_demand'.")

    zone_total_demand = record["zone_total_demand"]
    if len(zone_total_demand) != N_TOP_ZONES:
        fail(f"zone_total_demand does not have {N_TOP_ZONES} zones.")

    demand_zone_ids = {int(z) for z in zone_total_demand.keys()}
    if set(zone_ids_int) != demand_zone_ids:
        fail("zone_ids and zone_total_demand cover different zones.")

    for zone, demand in zone_total_demand.items():
        try:
            demand = int(demand)
        except (TypeError, ValueError):
            fail(f"Demand for zone {zone} is not numeric.")
        if demand < 0:
            fail(f"Negative demand at zone {zone}: {demand}")

    print(f"Frozen zones: {len(zone_ids_int)}")
    print(f"Selection period: {selection_period}")
    print("zone_ids unique: PASS")
    print("zone_total_demand consistency: PASS")
    print("TOP-50 FROZEN QA PASS")
    return zone_ids_int


def load_feature_table():
    """Load feature_table.csv from the processed data directory."""
    check_exists(FEATURE_TABLE_PATH, "feature_table.csv")
    return pd.read_csv(FEATURE_TABLE_PATH, parse_dates=["target_datetime"])


def check_feature_table(zone_ids):
    """Validate feature_table.csv.

    Checks required columns, zone vocabulary against the frozen
    Top-50, timestamp range and hourly alignment/continuity, demand
    and lag values, correctness of median_lag_3w, and that no
    out-of-scope feature columns (weather, borough, etc.) are
    present.
    """
    print_header("2. FEATURE TABLE QA")
    df = load_feature_table()

    required_columns = [
        "PULocationID", "target_datetime", "demand", "hour",
        "dayofweek", "is_holiday", "lag_1", "lag_24", "lag_168",
        "lag_336", "lag_504", "median_lag_3w",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        fail(
            "Feature table is missing columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    observed_zones = sorted(
        df["PULocationID"].dropna().astype(int).unique().tolist()
    )
    expected_zones = sorted(zone_ids)
    if observed_zones != expected_zones:
        fail(
            "feature_table zones do not match the frozen Top-50.\n"
            f"Expected: {expected_zones}\n"
            f"Observed: {observed_zones}"
        )
    print(f"Zones: {len(observed_zones)}")
    print("Zone vocabulary = frozen Top-50: PASS")

    duplicate_count = (
        df[["PULocationID", "target_datetime"]].duplicated().sum()
    )
    if duplicate_count > 0:
        fail(f"Feature table has {duplicate_count:,} duplicate rows.")
    print("Duplicate zone-hour: PASS")

    actual_start = df["target_datetime"].min()
    actual_end = df["target_datetime"].max()
    if actual_start != EXPECTED_START:
        fail(
            "Feature table starts at the wrong time.\n"
            f"Observed : {actual_start}\n"
            f"Expected : {EXPECTED_START}"
        )
    if actual_end != EXPECTED_END:
        fail(
            "Feature table ends at the wrong time.\n"
            f"Observed : {actual_end}\n"
            f"Expected : {EXPECTED_END}"
        )
    print(f"Target range: {actual_start} -> {actual_end}")

    if (df["target_datetime"].dt.minute != 0).any():
        fail("Some target_datetime values are not on the hour.")
    if (df["target_datetime"].dt.second != 0).any():
        fail("Some target_datetime values have second != 0.")
    print("Target timestamp hour alignment: PASS")

    rows_per_zone = df.groupby("PULocationID").size()
    expected_hours = 8256  # first 404 hours were trimmed
    if not (rows_per_zone == expected_hours).all():
        bad = rows_per_zone[rows_per_zone != expected_hours]
        fail(f"Uneven hourly observation counts per zone:\n{bad}")
    print(f"Rows per zone: {expected_hours:,}: PASS")

    expected_total_rows = N_TOP_ZONES * expected_hours
    if len(df) != expected_total_rows:
        fail(
            f"Feature table has {len(df):,} rows, "
            f"expected {expected_total_rows:,}."
        )
    print(f"Total rows: {len(df):,}: PASS")

    print("Checking hourly continuity by zone...")
    for zone, group in df.groupby("PULocationID"):
        times = group["target_datetime"].sort_values()
        diffs = times.diff().dropna()
        if not (diffs == pd.Timedelta(hours=1)).all():
            bad = diffs[diffs != pd.Timedelta(hours=1)]
            fail(f"Zone {zone} has an hourly gap:\n{bad.head(10)}")
    print("Hourly continuity: PASS")

    if df["demand"].isna().any():
        fail("Demand contains NaN.")
    if (df["demand"] < 0).any():
        fail("Demand contains negative values.")
    print("Demand non-negative / no NaN: PASS")

    if df["hour"].isna().any():
        fail("hour contains NaN.")
    if not df["hour"].astype(int).between(0, 23).all():
        fail("hour contains values outside [0, 23].")
    if not df["hour"].astype(int).eq(df["target_datetime"].dt.hour).all():
        fail("hour does not match target_datetime.")
    dow = df["target_datetime"].dt.dayofweek
    if not df["dayofweek"].astype(int).eq(dow).all():
        fail("dayofweek does not match target_datetime.")
    if not df["is_holiday"].isin([0, 1]).all():
        fail("is_holiday must only contain 0/1.")
    print("Calendar features: PASS")

    for col in ALL_LAGS:
        if df[col].isna().any():
            fail(f"{col} contains NaN.")
        if (df[col] < 0).any():
            fail(f"{col} contains negative values.")
    print("Required lag columns / no NaN: PASS")

    expected_median = df[WEEKLY_LAGS].median(axis=1, skipna=False)
    actual_median = df["median_lag_3w"]
    if not np.allclose(
        actual_median.to_numpy(),
        expected_median.to_numpy(),
        rtol=0,
        atol=0,
    ):
        mismatch = actual_median != expected_median
        fail(
            "median_lag_3w != median(lag_168, lag_336, lag_504).\n"
            f"Mismatches: {mismatch.sum():,}"
        )
    print("median_lag_3w correctness: PASS")

    forbidden_keywords = [
        "weather", "temperature", "precip", "snow", "neighbor",
        "borough", "hour_of_week", "dashboard", "deployment",
    ]
    suspicious = [
        col for col in df.columns
        if any(k in col.lower() for k in forbidden_keywords)
    ]
    if suspicious:
        fail(
            "Found columns outside core scope:\n"
            + "\n".join(f"  - {c}" for c in sorted(set(suspicious)))
        )
    print("Core feature scope: PASS")
    print("FEATURE TABLE QA PASS")
    return df


def check_variant_map():
    """Validate variant_feature_map.json.

    Confirms base_features and the weekly_features for variants
    A, B, and C match the documented protocol. Not required for
    training itself, but guards against silent protocol drift.
    """
    print_header("3. VARIANT FEATURE MAP QA")
    check_exists(VARIANT_MAP_PATH, "variant_feature_map.json")

    with open(VARIANT_MAP_PATH, "r", encoding="utf-8") as f:
        record = json.load(f)

    if "base_features" not in record:
        fail("variant_feature_map.json is missing 'base_features'.")
    if record["base_features"] != BASE_FEATURES:
        fail(
            "base_features is incorrect.\n"
            f"Observed: {record['base_features']}\n"
            f"Expected: {BASE_FEATURES}"
        )

    if "variants" not in record:
        fail("variant_feature_map.json is missing 'variants'.")

    variants = record["variants"]
    for variant_name, expected in EXPECTED_VARIANTS.items():
        if variant_name not in variants:
            fail(f"Missing Variant {variant_name}.")
        actual_features = variants[variant_name].get("weekly_features")
        if actual_features != expected["weekly_features"]:
            fail(
                f"Variant {variant_name} has wrong weekly features.\n"
                f"Observed: {actual_features}\n"
                f"Expected: {expected['weekly_features']}"
            )

    print("Variant A: lag_168 + lag_336 + lag_504: PASS")
    print("Variant B: median_lag_3w: PASS")
    print("Variant C: lag_168 + median_lag_3w: PASS")
    print("VARIANT FEATURE MAP QA PASS")


def get_fold_paths(split_name, eval_kind):
    """Return the (train, eval) CSV paths for a fold."""
    split_dir = FOLDS_DIR / split_name
    return split_dir / "train.csv", split_dir / f"{eval_kind}.csv"


def load_fold_data(split_name, split_cfg):
    """Load the train and eval/test CSVs for a single fold."""
    train_path, eval_path = get_fold_paths(
        split_name, split_cfg["eval_kind"]
    )
    check_exists(train_path, f"{split_name}/train.csv")
    check_exists(eval_path, f"{split_name}/{split_cfg['eval_kind']}.csv")

    train = pd.read_csv(train_path, parse_dates=["target_datetime"])
    eval_df = pd.read_csv(eval_path, parse_dates=["target_datetime"])
    return train, eval_df


def check_fold_structure(zone_ids):
    """Validate every fold: hpo, fold1-4, final_test.

    For each fold, checks that files exist and are non-empty, both
    splits contain the required columns and the same column set,
    zone vocabulary matches the frozen Top-50, train/eval timestamp
    ranges match the configured SPLITS window, train and eval are
    temporally adjacent with no overlap, and there is no duplicate
    (zone, timestamp) row or invalid demand.
    """
    print_header("4. FOLD DATA QA")
    fold_data = {}
    expected_feature_columns = None

    for split_name, split_cfg in SPLITS.items():
        print(f"\n--- {split_name.upper()} ---")
        train, eval_df = load_fold_data(split_name, split_cfg)

        if len(train) == 0:
            fail(f"[{split_name}] train.csv is empty.")
        if len(eval_df) == 0:
            fail(f"[{split_name}] eval/test is empty.")

        required = {"PULocationID", "target_datetime", "demand"}
        for name, data in [("train", train), ("eval/test", eval_df)]:
            missing = required - set(data.columns)
            if missing:
                fail(
                    f"[{split_name}/{name}] missing columns: "
                    f"{sorted(missing)}"
                )

        train_columns = list(train.columns)
        eval_columns = list(eval_df.columns)
        if train_columns != eval_columns:
            fail(f"[{split_name}] train and eval/test columns differ.")

        if expected_feature_columns is None:
            expected_feature_columns = train_columns
        elif train_columns != expected_feature_columns:
            fail(f"[{split_name}] columns differ from other folds.")

        train_zones = sorted(
            train["PULocationID"].dropna().astype(int).unique().tolist()
        )
        eval_zones = sorted(
            eval_df["PULocationID"].dropna().astype(int).unique().tolist()
        )
        expected_zones = sorted(zone_ids)
        if train_zones != expected_zones:
            fail(f"[{split_name}] train zones != frozen Top-50.")
        if eval_zones != expected_zones:
            fail(f"[{split_name}] eval/test zones != frozen Top-50.")

        train_start_expected, train_end_expected = split_cfg["train"]
        eval_start_expected, eval_end_expected = split_cfg["eval"]

        train_min = train["target_datetime"].min()
        train_max = train["target_datetime"].max()
        eval_min = eval_df["target_datetime"].min()
        eval_max = eval_df["target_datetime"].max()

        if train_min != train_start_expected:
            fail(
                f"[{split_name}] train starts at the wrong time.\n"
                f"Observed : {train_min}\n"
                f"Expected : {train_start_expected}"
            )
        expected_train_max = train_end_expected - pd.Timedelta(hours=1)
        if train_max != expected_train_max:
            fail(
                f"[{split_name}] train ends at the wrong time.\n"
                f"Observed : {train_max}\n"
                f"Expected : {expected_train_max}"
            )
        if eval_min != eval_start_expected:
            fail(
                f"[{split_name}] eval/test starts at the wrong time.\n"
                f"Observed : {eval_min}\n"
                f"Expected : {eval_start_expected}"
            )
        expected_eval_max = eval_end_expected - pd.Timedelta(hours=1)
        if eval_max != expected_eval_max:
            fail(
                f"[{split_name}] eval/test ends at the wrong time.\n"
                f"Observed : {eval_max}\n"
                f"Expected : {expected_eval_max}"
            )

        if train_max >= eval_min:
            fail(f"[{split_name}] train and eval/test overlap.")
        gap = eval_min - train_max
        if gap != pd.Timedelta(hours=1):
            fail(f"[{split_name}] train/eval are not adjacent. Gap: {gap}")

        for name, data in [("train", train), ("eval/test", eval_df)]:
            duplicate_count = (
                data[["PULocationID", "target_datetime"]]
                .duplicated()
                .sum()
            )
            if duplicate_count > 0:
                fail(
                    f"[{split_name}/{name}] has {duplicate_count:,} "
                    "duplicate (zone, target_datetime) rows."
                )
            if data["demand"].isna().any():
                fail(f"[{split_name}/{name}] demand has NaN.")
            if (data["demand"] < 0).any():
                fail(f"[{split_name}/{name}] demand has negative values.")

        print(f"Train : {len(train):,} rows ({train_min} -> {train_max})")
        print(f"Eval  : {len(eval_df):,} rows ({eval_min} -> {eval_max})")
        print("Zone vocabulary: PASS")
        print("Train/eval temporal separation: PASS")

        fold_data[split_name] = {"train": train, "eval": eval_df}

    print("\nFOLD STRUCTURE QA PASS")
    return fold_data


def check_temporal_leakage(fold_data):
    """Validate that folds follow the expanding-window design.

    Checks, per fold, that train never contains an observation at or
    after its own eval start; that eval/test windows across folds do
    not overlap; that December data only appears in final_test; and
    that training windows expand monotonically over the fold order.

    Note: it is expected, not leakage, for a fold's eval data to
    reappear inside the training window of a later fold (e.g. July
    is the hpo eval period but is included in fold1 training).
    """
    print_header("5. TEMPORAL LEAKAGE BETWEEN FOLDS QA")
    ordered_splits = ["hpo", "fold1", "fold2", "fold3", "fold4", "final_test"]

    for split_name in ordered_splits:
        train = fold_data[split_name]["train"]
        eval_df = fold_data[split_name]["eval"]
        train_max = train["target_datetime"].max()
        eval_min = eval_df["target_datetime"].min()

        if train_max >= eval_min:
            fail(
                f"[{split_name}] Temporal leakage: train contains an "
                "observation >= eval start."
            )
        print(f"[{split_name}] train max={train_max}, eval min={eval_min}: PASS")

    eval_intervals = [
        (name, SPLITS[name]["eval"][0], SPLITS[name]["eval"][1])
        for name in ordered_splits
    ]
    for i, (name_i, start_i, end_i) in enumerate(eval_intervals):
        for name_j, start_j, end_j in eval_intervals[i + 1:]:
            if start_i < end_j and start_j < end_i:
                fail(
                    "Evaluation windows overlap:\n"
                    f"{name_i}: {start_i} -> {end_i}\n"
                    f"{name_j}: {start_j} -> {end_j}"
                )
    print("Validation/test windows non-overlapping: PASS")

    final_test = fold_data["final_test"]["eval"]
    final_test_start = final_test["target_datetime"].min()
    final_test_end = final_test["target_datetime"].max()
    expected_final_start = pd.Timestamp("2025-12-01 00:00:00")
    expected_final_end = pd.Timestamp("2025-12-31 23:00:00")

    if final_test_start != expected_final_start:
        fail("Final test does not start at 2025-12-01.")
    if final_test_end != expected_final_end:
        fail("Final test does not end at 2025-12-31 23:00.")

    december_start = pd.Timestamp("2025-12-01 00:00:00")
    for split_name in ["hpo", "fold1", "fold2", "fold3", "fold4"]:
        train = fold_data[split_name]["train"]
        eval_df = fold_data[split_name]["eval"]
        train_december = (train["target_datetime"] >= december_start).sum()
        eval_december = (eval_df["target_datetime"] >= december_start).sum()
        if train_december > 0 or eval_december > 0:
            fail(
                f"[{split_name}] December data appears before "
                "final_test.\n"
                f"Train December rows: {train_december:,}\n"
                f"Eval December rows : {eval_december:,}"
            )
    print("Final December isolation: PASS")

    previous_train_end = None
    for split_name in ordered_splits:
        train_end = SPLITS[split_name]["train"][1]
        if previous_train_end is not None and train_end < previous_train_end:
            fail(f"[{split_name}] train window does not expand over time.")
        previous_train_end = train_end
    print("Expanding training windows: PASS")

    print("\nTEMPORAL LEAKAGE QA PASS")


def main():
    """Run the full QA/QC suite and exit non-zero on failure."""
    print("\n" + "=" * 75)
    print("FINAL DATA QA/QC")
    print("=" * 75)
    print(f"\nRepository:\n{REPO_ROOT}")
    print("\nNOTE:")
    print("- Raw data is not checked here.")
    print("- Monthly aggregation is not checked here.")
    print("- Lag alignment is not checked here; see split_folds.py.")

    zone_ids = check_top50_frozen()
    check_feature_table(zone_ids)
    check_variant_map()
    fold_data = check_fold_structure(zone_ids)
    check_temporal_leakage(fold_data)

    print("\n" + "=" * 75)
    print("ALL FINAL DATA QA/QC CHECKS PASS")
    print("=" * 75)
    print("\nChecked:")
    print("1. top50_zones_frozen.json")
    print("2. feature_table.csv")
    print("3. variant_feature_map.json")
    print("4. hpo/fold1/fold2/fold3/fold4/final_test")
    print("5. Temporal leakage / final-test isolation")
    print("\nLag alignment: not checked here; run split_folds.py for that.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("\n" + "=" * 75)
        print("QA/QC FAILED")
        print("=" * 75)
        print(f"\n{exc}\n")
        sys.exit(1)
