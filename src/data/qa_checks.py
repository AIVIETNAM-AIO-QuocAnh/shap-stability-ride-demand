"""Run the final data QA checks for the Data handoff."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

from src.configuration import PROJECT_ROOT, DataConfig, SplitConfig, load_data_config


EXPECTED_TOP50_KEYS = {
    "selection_period",
    "selection_rule",
    "n_zones",
    "zone_ids",
    "zone_total_demand",
}


def fail(message):
    """Raise AssertionError so QA stops immediately."""
    raise AssertionError(message)


def check_exists(path: Path, description: str) -> None:
    """Require a mandatory file to exist on disk."""
    if not path.exists():
        fail(f"MISSING {description}:\n  {path}")


def print_header(title: str) -> None:
    """Print a section heading to stdout."""
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def check_raw_inputs(config: DataConfig) -> None:
    """Require all raw Parquet files and the two protocol columns."""
    print_header("1. RAW INPUT QA")
    source = config["source"]
    raw_dir = config["paths"]["raw_dir"]
    required_columns = set(source["required_columns"])
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("The project environment is missing pyarrow for Parquet QA") from exc

    for month_label in config["months"]:
        path = raw_dir / source["filename_pattern"].format(
            year=config["year"], month=int(month_label[-2:])
        )
        check_exists(path, f"raw month {month_label}")
        missing = required_columns - set(parquet.read_schema(path).names)
        if missing:
            fail(f"{path} is missing raw columns: {sorted(missing)}")
    print("All 12 raw Parquet files and required columns: PASS")


def check_monthly_aggregates(config: DataConfig) -> pd.DataFrame:
    """Validate every monthly aggregate and return the combined table."""
    print_header("2. MONTHLY AGGREGATE QA")
    frames: list[pd.DataFrame] = []
    required_columns = {"pu_location_id", "hour", "trip_count"}
    aggregate_dir = config["paths"]["monthly_aggregates_dir"]
    for month_label in config["months"]:
        path = aggregate_dir / f"agg_{month_label}.csv"
        check_exists(path, f"monthly aggregate {month_label}")
        frame = pd.read_csv(path, parse_dates=["hour"])
        missing = required_columns - set(frame.columns)
        if missing:
            fail(f"{path} is missing columns: {sorted(missing)}")
        duplicate_count = frame[["pu_location_id", "hour"]].duplicated().sum()
        if duplicate_count > 0:
            fail(f"{path} contains {duplicate_count:,} duplicate zone-hour rows")
        if frame["trip_count"].isna().any() or (frame["trip_count"] < 0).any():
            fail(f"{path} contains invalid trip_count values")
        frames.append(frame)
    aggregate = pd.concat(frames, ignore_index=True)
    print(f"Validated 12 monthly aggregates: {len(aggregate):,} zone-hour rows.")
    return aggregate


def check_top50_frozen(aggregate: pd.DataFrame, config: DataConfig) -> list[int]:
    """Validate the frozen-zone artifact against an independent recomputation."""
    print_header("3. TOP-50 FROZEN ZONES QA")
    top50_path = config["paths"]["frozen_zones"]
    selection = config["selection"]
    check_exists(top50_path, "top50_zones_frozen.json")

    with open(top50_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    if set(record) != EXPECTED_TOP50_KEYS:
        fail(
            "top50_zones_frozen.json has an unexpected schema. "
            f"Observed keys: {sorted(record)}; expected keys: {sorted(EXPECTED_TOP50_KEYS)}"
        )

    if "zone_ids" not in record:
        fail("top50_zones_frozen.json is missing 'zone_ids'.")

    zone_ids = record["zone_ids"]
    if not isinstance(zone_ids, list):
        fail("'zone_ids' must be a list.")
    if len(zone_ids) != selection["top_zones"]:
        fail(f"Found {len(zone_ids)} zones; expected {selection['top_zones']}.")
    if len(set(zone_ids)) != selection["top_zones"]:
        fail("zone_ids contains duplicate IDs.")

    try:
        zone_ids_int = [int(z) for z in zone_ids]
    except (TypeError, ValueError):
        fail("At least one zone ID cannot be converted to int.")

    if any(z <= 0 for z in zone_ids_int):
        fail("Frozen-zone list contains pu_location_id <= 0.")

    selection_period = record.get("selection_period")
    if selection_period != selection["period_label"]:
        fail(
            "selection_period is incorrect.\n"
            f"Observed : {selection_period}\n"
            f"Expected : {selection['period_label']}"
        )

    if record.get("n_zones") != selection["top_zones"]:
        fail(f"n_zones = {record.get('n_zones')}; expected {selection['top_zones']}.")

    if record.get("selection_rule") != selection["selection_rule"]:
        fail(
            "selection_rule is incorrect.\n"
            f"Observed : {record.get('selection_rule')}\n"
            f"Expected : {selection['selection_rule']}"
        )

    if "zone_total_demand" not in record:
        fail("Missing 'zone_total_demand'.")

    zone_total_demand = record["zone_total_demand"]
    if len(zone_total_demand) != selection["top_zones"]:
        fail(f"zone_total_demand must contain exactly {selection['top_zones']} zones.")

    demand_zone_ids = {int(z) for z in zone_total_demand.keys()}
    if set(zone_ids_int) != demand_zone_ids:
        fail("zone_ids and zone_total_demand cover different zones.")

    for zone, demand in zone_total_demand.items():
        try:
            demand = int(demand)
        except (TypeError, ValueError):
            fail(f"Demand for zone {zone} is not numeric.")
        if demand < 0:
            fail(f"Negative demand for zone {zone}: {demand}")

    jan_jun = aggregate[
        (aggregate["hour"] >= selection["start"])
        & (aggregate["hour"] < selection["end_exclusive"])
    ]
    totals = jan_jun.groupby("pu_location_id")["trip_count"].sum()
    ranking = totals.rename("total_demand").reset_index().sort_values(
        ["total_demand", "pu_location_id"], ascending=[False, True]
    )
    recomputed_ids = ranking.head(selection["top_zones"])["pu_location_id"].astype(int).tolist()
    if recomputed_ids != zone_ids_int:
        fail("Frozen zone IDs do not match the Jan-Jun aggregate ranking.")
    recomputed_totals = {
        str(int(row.pu_location_id)): int(row.total_demand)
        for row in ranking.head(selection["top_zones"]).itertuples(index=False)
    }
    stored_totals = {str(key): int(value) for key, value in zone_total_demand.items()}
    if recomputed_totals != stored_totals:
        differences = [
            abs(recomputed_totals[zone] - stored_totals.get(zone, 0))
            for zone in recomputed_totals
        ]
        fail(
            "Frozen zone_total_demand does not match the Jan-Jun recomputation: "
            f"{sum(delta != 0 for delta in differences)} zones differ; "
            f"maximum absolute difference is {max(differences):,}."
        )

    print(f"Frozen zones: {len(zone_ids_int)}")
    print(f"Selection period: {selection_period}")
    print("Unique zone_ids: PASS")
    print("zone_total_demand exact match: PASS")
    print("Independent Jan-Jun selection recomputation: PASS")
    print("QA TOP-50 FROZEN PASS")
    return zone_ids_int


def load_feature_table(path: Path) -> pd.DataFrame:
    """Read feature_table.csv from the processed-data directory."""
    check_exists(path, "feature_table.csv")
    return pd.read_csv(path, parse_dates=["target_datetime"])


def check_feature_table(zone_ids: list[int], config: DataConfig) -> pd.DataFrame:
    """Validate required columns, zones, timestamps, demand, lags, and scope."""
    print_header("4. FEATURE TABLE QA")
    panel = config["panel"]
    selection = config["selection"]
    df = load_feature_table(config["paths"]["feature_table"])

    required_columns = [
        "pu_location_id", "target_datetime", "demand", *panel["base_features"],
        *(f"lag_{lag}" for lag in panel["weekly_lags_hours"]), panel["median_feature"],
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        fail(
            "Feature table is missing columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    observed_zones = sorted(
        df["pu_location_id"].dropna().astype(int).unique().tolist()
    )
    expected_zones = sorted(zone_ids)
    if observed_zones != expected_zones:
        fail(
            "Zones in feature_table do not match the frozen Top-50.\n"
            f"Expected: {expected_zones}\n"
            f"Observed: {observed_zones}"
        )
    print(f"Zones: {len(observed_zones)}")
    print("Zone vocabulary = frozen Top-50: PASS")

    duplicate_count = (
        df[["pu_location_id", "target_datetime"]].duplicated().sum()
    )
    if duplicate_count > 0:
        fail(f"Feature table contains {duplicate_count:,} duplicate rows.")
    print("Unique zone-hour keys: PASS")

    actual_start = df["target_datetime"].min()
    actual_end = df["target_datetime"].max()
    warmup_start = panel["start"] + pd.Timedelta(hours=max(panel["all_lags_hours"]))
    expected_end = panel["end_exclusive"] - pd.Timedelta(hours=1)
    if actual_start != warmup_start:
        fail(
            "Feature table starts at the wrong timestamp.\n"
            f"Observed : {actual_start}\n"
            f"Expected : {warmup_start}"
        )
    if actual_end != expected_end:
        fail(
            "Feature table ends at the wrong timestamp.\n"
            f"Observed : {actual_end}\n"
            f"Expected : {expected_end}"
        )
    print(f"Target range: {actual_start} -> {actual_end}")

    if (df["target_datetime"].dt.minute != 0).any():
        fail("Some target_datetime values are not on the hour.")
    if (df["target_datetime"].dt.second != 0).any():
        fail("Some target_datetime values have second != 0.")
    print("Target timestamps are aligned to the hour: PASS")

    rows_per_zone = df.groupby("pu_location_id").size()
    expected_hours = int((panel["end_exclusive"] - warmup_start) / pd.Timedelta(hours=1))
    if not (rows_per_zone == expected_hours).all():
        bad = rows_per_zone[rows_per_zone != expected_hours]
        fail(f"Uneven hourly observation counts per zone:\n{bad}")
    print(f"Rows per zone: {expected_hours:,}: PASS")

    expected_total_rows = selection["top_zones"] * expected_hours
    if len(df) != expected_total_rows:
        fail(
            f"Feature table has {len(df):,} rows; "
            f"expected {expected_total_rows:,}."
        )
    print(f"Total rows: {len(df):,}: PASS")

    print("Checking hourly continuity by zone...")
    for zone, group in df.groupby("pu_location_id"):
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

    lag_columns = tuple(f"lag_{lag}" for lag in panel["all_lags_hours"])
    for col in lag_columns:
        if df[col].isna().any():
            fail(f"{col} contains NaN.")
        if (df[col] < 0).any():
            fail(f"{col} contains negative values.")
    print("Required lag columns / no NaN: PASS")

    weekly_columns = [f"lag_{lag}" for lag in panel["weekly_lags_hours"]]
    expected_median = df[weekly_columns].median(axis=1, skipna=False)
    actual_median = df[panel["median_feature"]]
    if not np.allclose(
        actual_median.to_numpy(),
        expected_median.to_numpy(),
        rtol=0,
        atol=0,
    ):
        mismatch = actual_median != expected_median
        fail(
            f"{panel['median_feature']} does not equal the median weekly lag.\n"
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


def check_variant_map(config: DataConfig) -> None:
    """Validate the A/B/C feature contract in variant_feature_map.json."""
    print_header("5. VARIANT FEATURE MAP QA")
    panel = config["panel"]
    variant_map_path = config["paths"]["variant_map"]
    check_exists(variant_map_path, "variant_feature_map.json")

    with open(variant_map_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    if "base_features" not in record:
        fail("variant_feature_map.json is missing 'base_features'.")
    if record["base_features"] != list(panel["base_features"]):
        fail(
            "base_features is incorrect.\n"
            f"Observed: {record['base_features']}\n"
            f"Expected: {list(panel['base_features'])}"
        )

    if "variants" not in record:
        fail("variant_feature_map.json is missing 'variants'.")

    variants = record["variants"]
    for variant_name, expected in panel["variants"].items():
        if variant_name not in variants:
            fail(f"Missing Variant {variant_name}.")
        actual_features = variants[variant_name].get("weekly_features")
        if actual_features != list(expected["weekly_features"]):
            fail(
                f"Variant {variant_name} has wrong weekly features.\n"
                f"Observed: {actual_features}\n"
                f"Expected: {list(expected['weekly_features'])}"
            )

    print("Variant A: lag_168 + lag_336 + lag_504: PASS")
    print("Variant B: median_lag_3w: PASS")
    print("Variant C: lag_168 + median_lag_3w: PASS")
    print("VARIANT FEATURE MAP QA PASS")


def check_lag_alignment_examples(config: DataConfig) -> None:
    """Validate durable lag-alignment evidence from the full panel."""
    print_header("6. LAG ALIGNMENT QA")
    lag_examples_path = config["paths"]["lag_examples"]
    check_exists(lag_examples_path, "lag_alignment_examples.csv")
    examples = pd.read_csv(
        lag_examples_path,
        parse_dates=["target_datetime", "source_datetime"],
    )
    expected_columns = {
        "pu_location_id",
        "target_datetime",
        "lag",
        "source_datetime",
        "source_demand",
        "feature_value",
        "matches",
    }
    missing = expected_columns - set(examples.columns)
    if missing:
        fail(f"Lag examples are missing columns: {sorted(missing)}")
    if len(examples) != 75:
        fail(f"Expected 75 lag examples, found {len(examples)}")
    if not examples["matches"].eq(True).all():
        fail("Lag examples contain mismatches")
    source_delta = examples["target_datetime"] - examples["source_datetime"]
    expected_delta = pd.to_timedelta(examples["lag"], unit="h")
    if not source_delta.eq(expected_delta).all():
        fail("Lag examples contain incorrect source timestamps")
    print("75 full-panel lag examples and source timestamps: PASS")


def get_fold_paths(split_name: str, eval_kind: str, folds_dir: Path) -> tuple[Path, Path]:
    """Return the train and evaluation CSV paths for one split."""
    split_dir = folds_dir / split_name
    return split_dir / "train.csv", split_dir / f"{eval_kind}.csv"


def load_fold_data(
    split_name: str, split_cfg: SplitConfig, folds_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the train and evaluation/test CSV files for one split."""
    train_path, eval_path = get_fold_paths(
        split_name, split_cfg["eval_kind"], folds_dir
    )
    check_exists(train_path, f"{split_name}/train.csv")
    check_exists(eval_path, f"{split_name}/{split_cfg['eval_kind']}.csv")

    train = pd.read_csv(train_path, parse_dates=["target_datetime"])
    eval_df = pd.read_csv(eval_path, parse_dates=["target_datetime"])
    return train, eval_df


def check_fold_structure(zone_ids: list[int], config: DataConfig) -> dict[str, dict[str, pd.DataFrame]]:
    """Validate files, columns, zones, windows, and values for every split."""
    print_header("7. FOLD DATA QA")
    fold_data = {}
    expected_feature_columns = None

    for split_name, split_cfg in config["splits"].items():
        print(f"\n--- {split_name.upper()} ---")
        train, eval_df = load_fold_data(split_name, split_cfg, config["paths"]["folds_dir"])

        if len(train) == 0:
            fail(f"[{split_name}] train.csv is empty.")
        if len(eval_df) == 0:
            fail(f"[{split_name}] eval/test is empty.")

        required = {"pu_location_id", "target_datetime", "demand"}
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
            train["pu_location_id"].dropna().astype(int).unique().tolist()
        )
        eval_zones = sorted(
            eval_df["pu_location_id"].dropna().astype(int).unique().tolist()
        )
        expected_zones = sorted(zone_ids)
        if train_zones != expected_zones:
            fail(f"[{split_name}] train zones != frozen Top-50.")
        if eval_zones != expected_zones:
            fail(f"[{split_name}] eval/test zones != frozen Top-50.")

        train_start_expected = split_cfg["train_start"]
        train_end_expected = split_cfg["train_end_exclusive"]
        eval_start_expected = split_cfg["eval_start"]
        eval_end_expected = split_cfg["eval_end_exclusive"]

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
                data[["pu_location_id", "target_datetime"]]
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


def check_temporal_leakage(
    fold_data: dict[str, dict[str, pd.DataFrame]], config: DataConfig
) -> None:
    """Validate cross-fold windows, final-test isolation, and expansion.

    An earlier fold's evaluation rows reappearing in a later training window
    is required by the expanding-window design and is not leakage.
    """
    print_header("8. TEMPORAL LEAKAGE BETWEEN FOLDS QA")
    ordered_splits = tuple(config["splits"])

    eval_intervals = [
        (
            name,
            config["splits"][name]["eval_start"],
            config["splits"][name]["eval_end_exclusive"],
        )
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
    expected_final_start = config["splits"]["final_test"]["eval_start"]
    expected_final_end = config["splits"]["final_test"]["eval_end_exclusive"] - pd.Timedelta(hours=1)

    if final_test_start != expected_final_start:
        fail("Final test does not start at 2025-12-01.")
    if final_test_end != expected_final_end:
        fail("Final test does not end at 2025-12-31 23:00.")

    final_test_start = config["splits"]["final_test"]["eval_start"]
    for split_name in ordered_splits[:-1]:
        train = fold_data[split_name]["train"]
        eval_df = fold_data[split_name]["eval"]
        train_final_period = (train["target_datetime"] >= final_test_start).sum()
        eval_final_period = (eval_df["target_datetime"] >= final_test_start).sum()
        if train_final_period > 0 or eval_final_period > 0:
            fail(
                f"[{split_name}] December data appears before "
                "final_test.\n"
                f"Train final-test-period rows: {train_final_period:,}\n"
                f"Eval final-test-period rows : {eval_final_period:,}"
            )
    print("Final December isolation: PASS")

    previous_train_end = None
    for split_name in ordered_splits:
        train_end = config["splits"][split_name]["train_end_exclusive"]
        if previous_train_end is not None and train_end < previous_train_end:
            fail(f"[{split_name}] train window does not expand over time.")
        previous_train_end = train_end
    print("Expanding training windows: PASS")

    print("\nTEMPORAL LEAKAGE QA PASS")


def main():
    """Run all QA/QC checks and exit non-zero on failure."""
    config = load_data_config()
    print("\n" + "=" * 75)
    print("FINAL DATA QA/QC")
    print("=" * 75)
    print(f"\nRepository:\n{PROJECT_ROOT}")
    print("\nNOTE:")
    print("- Raw Parquet schema, monthly aggregates, and full-panel lag examples are checked here.")

    check_raw_inputs(config)
    aggregate = check_monthly_aggregates(config)
    zone_ids = check_top50_frozen(aggregate, config)
    check_feature_table(zone_ids, config)
    check_variant_map(config)
    check_lag_alignment_examples(config)
    fold_data = check_fold_structure(zone_ids, config)
    check_temporal_leakage(fold_data, config)

    print("\n" + "=" * 75)
    print("ALL FINAL DATA QA/QC CHECKS PASS")
    print("=" * 75)
    print("\nChecked:")
    print("1. Raw Parquet schemas")
    print("2. Monthly aggregate tables")
    print("3. top50_zones_frozen.json")
    print("4. feature_table.csv")
    print("5. variant_feature_map.json")
    print("6. Full-panel lag-alignment examples")
    print("7. hpo/fold1/fold2/fold3/fold4/final_test")
    print("8. Temporal leakage / final-test isolation")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("\n" + "=" * 75)
        print("QA/QC FAILED")
        print("=" * 75)
        print(f"\n{exc}\n")
        sys.exit(1)
