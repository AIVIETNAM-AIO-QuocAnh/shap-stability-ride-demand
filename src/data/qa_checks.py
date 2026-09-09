"""Chạy các kiểm tra Data QA cuối cho Data handoff."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

from src.load_config import PROJECT_ROOT, DataConfig, SplitConfig, load_data_config


EXPECTED_TOP50_KEYS = {
    "selection_period",
    "selection_rule",
    "n_zones",
    "zone_ids",
    "zone_total_demand",
}


def fail(message):
    """Raise AssertionError để QA dừng ngay lập tức."""
    raise AssertionError(message)


def check_exists(path: Path, description: str) -> None:
    """Yêu cầu file bắt buộc tồn tại trên disk."""
    if not path.exists():
        fail(f"THIẾU {description}:\n  {path}")


def print_header(title: str) -> None:
    """In section heading ra stdout."""
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def check_raw_inputs(config: DataConfig) -> None:
    """Yêu cầu đủ raw Parquet file và hai protocol column."""
    print_header("1. QA RAW INPUT")
    source = config["source"]
    raw_dir = config["paths"]["raw_dir"]
    required_columns = set(source["required_columns"])
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Project environment thiếu pyarrow để thực hiện Parquet QA") from exc

    for month_label in config["months"]:
        path = raw_dir / source["filename_pattern"].format(
            year=config["year"], month=int(month_label[-2:])
        )
        check_exists(path, f"raw month {month_label}")
        missing = required_columns - set(parquet.read_schema(path).names)
        if missing:
            fail(f"{path} thiếu raw column: {sorted(missing)}")
    print("Đủ 12 raw Parquet file và column bắt buộc: PASS")


def check_monthly_aggregates(config: DataConfig) -> pd.DataFrame:
    """Validate mọi monthly aggregate và trả về table đã gộp."""
    print_header("2. QA MONTHLY AGGREGATE")
    frames: list[pd.DataFrame] = []
    required_columns = {"pu_location_id", "hour", "trip_count"}
    aggregate_dir = config["paths"]["monthly_aggregates_dir"]
    for month_label in config["months"]:
        path = aggregate_dir / f"agg_{month_label}.csv"
        check_exists(path, f"monthly aggregate {month_label}")
        frame = pd.read_csv(path, parse_dates=["hour"])
        missing = required_columns - set(frame.columns)
        if missing:
            fail(f"{path} thiếu column: {sorted(missing)}")
        duplicate_count = frame[["pu_location_id", "hour"]].duplicated().sum()
        if duplicate_count > 0:
            fail(f"{path} chứa {duplicate_count:,} zone-hour row duplicate")
        if frame["trip_count"].isna().any() or (frame["trip_count"] < 0).any():
            fail(f"{path} chứa giá trị trip_count không hợp lệ")
        frames.append(frame)
    aggregate = pd.concat(frames, ignore_index=True)
    print(f"Đã validate 12 monthly aggregate: {len(aggregate):,} zone-hour row.")
    return aggregate


def check_top50_frozen(aggregate: pd.DataFrame, config: DataConfig) -> list[int]:
    """Validate frozen-zone artifact bằng một lần recompute độc lập."""
    print_header("3. QA TOP-50 FROZEN ZONE")
    top50_path = config["paths"]["frozen_zones"]
    selection = config["selection"]
    check_exists(top50_path, "top50_zones_frozen.json")

    with open(top50_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    if set(record) != EXPECTED_TOP50_KEYS:
        fail(
            "top50_zones_frozen.json có schema không mong đợi. "
            f"Key quan sát: {sorted(record)}; key kỳ vọng: {sorted(EXPECTED_TOP50_KEYS)}"
        )

    if "zone_ids" not in record:
        fail("top50_zones_frozen.json thiếu 'zone_ids'.")

    zone_ids = record["zone_ids"]
    if not isinstance(zone_ids, list):
        fail("'zone_ids' phải là list.")
    if len(zone_ids) != selection["top_zones"]:
        fail(f"Phát hiện {len(zone_ids)} zone; kỳ vọng {selection['top_zones']}.")
    if len(set(zone_ids)) != selection["top_zones"]:
        fail("zone_ids chứa ID duplicate.")

    try:
        zone_ids_int = [int(z) for z in zone_ids]
    except (TypeError, ValueError):
        fail("Có ít nhất một zone ID không thể chuyển thành int.")

    if any(z <= 0 for z in zone_ids_int):
        fail("Danh sách frozen zone chứa pu_location_id <= 0.")

    selection_period = record.get("selection_period")
    if selection_period != selection["period_label"]:
        fail(
            "selection_period không đúng.\n"
            f"Quan sát : {selection_period}\n"
            f"Kỳ vọng : {selection['period_label']}"
        )

    if record.get("n_zones") != selection["top_zones"]:
        fail(f"n_zones = {record.get('n_zones')}; kỳ vọng {selection['top_zones']}.")

    if record.get("selection_rule") != selection["selection_rule"]:
        fail(
            "selection_rule không đúng.\n"
            f"Quan sát : {record.get('selection_rule')}\n"
            f"Kỳ vọng : {selection['selection_rule']}"
        )

    if "zone_total_demand" not in record:
        fail("Thiếu 'zone_total_demand'.")

    zone_total_demand = record["zone_total_demand"]
    if len(zone_total_demand) != selection["top_zones"]:
        fail(f"zone_total_demand phải chứa đúng {selection['top_zones']} zone.")

    demand_zone_ids = {int(z) for z in zone_total_demand.keys()}
    if set(zone_ids_int) != demand_zone_ids:
        fail("zone_ids và zone_total_demand bao phủ các zone khác nhau.")

    for zone, demand in zone_total_demand.items():
        try:
            demand = int(demand)
        except (TypeError, ValueError):
            fail(f"Demand của zone {zone} không phải numeric.")
        if demand < 0:
            fail(f"Demand âm cho zone {zone}: {demand}")

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
        fail("Frozen zone ID không khớp ranking aggregate Jan-Jun.")
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
            "Frozen zone_total_demand không khớp recompute Jan-Jun: "
            f"{sum(delta != 0 for delta in differences)} zone khác nhau; "
            f"độ lệch tuyệt đối lớn nhất là {max(differences):,}."
        )

    print(f"Số frozen zone: {len(zone_ids_int)}")
    print(f"Khoảng selection: {selection_period}")
    print("zone_ids unique: PASS")
    print("zone_total_demand khớp tuyệt đối: PASS")
    print("Recompute selection Jan-Jun độc lập: PASS")
    print("QA TOP-50 FROZEN PASS")
    return zone_ids_int


def load_feature_table(path: Path) -> pd.DataFrame:
    """Đọc feature_table.csv từ processed-data directory."""
    check_exists(path, "feature_table.csv")
    return pd.read_csv(path, parse_dates=["target_datetime"])


def check_feature_table(zone_ids: list[int], config: DataConfig) -> pd.DataFrame:
    """Validate column bắt buộc, zone, timestamp, demand, lag và scope."""
    print_header("4. QA FEATURE TABLE")
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
            "Feature table thiếu column:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    observed_zones = sorted(
        df["pu_location_id"].dropna().astype(int).unique().tolist()
    )
    expected_zones = sorted(zone_ids)
    if observed_zones != expected_zones:
        fail(
            "Zone trong feature_table không khớp frozen Top-50.\n"
            f"Kỳ vọng: {expected_zones}\n"
            f"Quan sát: {observed_zones}"
        )
    print(f"Số zone: {len(observed_zones)}")
    print("Zone vocabulary = frozen Top-50: PASS")

    duplicate_count = (
        df[["pu_location_id", "target_datetime"]].duplicated().sum()
    )
    if duplicate_count > 0:
        fail(f"Feature table chứa {duplicate_count:,} row duplicate.")
    print("Zone-hour key unique: PASS")

    actual_start = df["target_datetime"].min()
    actual_end = df["target_datetime"].max()
    warmup_start = panel["start"] + pd.Timedelta(hours=max(panel["all_lags_hours"]))
    expected_end = panel["end_exclusive"] - pd.Timedelta(hours=1)
    if actual_start != warmup_start:
        fail(
            "Feature table bắt đầu ở timestamp sai.\n"
            f"Quan sát : {actual_start}\n"
            f"Kỳ vọng : {warmup_start}"
        )
    if actual_end != expected_end:
        fail(
            "Feature table kết thúc ở timestamp sai.\n"
            f"Quan sát : {actual_end}\n"
            f"Kỳ vọng : {expected_end}"
        )
    print(f"Dải target: {actual_start} -> {actual_end}")

    if (df["target_datetime"].dt.minute != 0).any():
        fail("Một số target_datetime không nằm đúng đầu hour.")
    if (df["target_datetime"].dt.second != 0).any():
        fail("Một số target_datetime có second != 0.")
    print("Target timestamp được căn đúng đầu hour: PASS")

    rows_per_zone = df.groupby("pu_location_id").size()
    expected_hours = int((panel["end_exclusive"] - warmup_start) / pd.Timedelta(hours=1))
    if not (rows_per_zone == expected_hours).all():
        bad = rows_per_zone[rows_per_zone != expected_hours]
        fail(f"Số hourly observation theo zone không đều:\n{bad}")
    print(f"Số row mỗi zone: {expected_hours:,}: PASS")

    expected_total_rows = selection["top_zones"] * expected_hours
    if len(df) != expected_total_rows:
        fail(
            f"Feature table có {len(df):,} row; "
            f"kỳ vọng {expected_total_rows:,}."
        )
    print(f"Tổng số row: {len(df):,}: PASS")

    print("Đang kiểm tra hourly continuity theo zone...")
    for zone, group in df.groupby("pu_location_id"):
        times = group["target_datetime"].sort_values()
        diffs = times.diff().dropna()
        if not (diffs == pd.Timedelta(hours=1)).all():
            bad = diffs[diffs != pd.Timedelta(hours=1)]
            fail(f"Zone {zone} có hourly gap:\n{bad.head(10)}")
    print("Hourly continuity: PASS")

    if df["demand"].isna().any():
        fail("Demand chứa NaN.")
    if (df["demand"] < 0).any():
        fail("Demand chứa giá trị âm.")
    print("Demand không âm / không có NaN: PASS")

    if df["hour"].isna().any():
        fail("hour chứa NaN.")
    if not df["hour"].astype(int).between(0, 23).all():
        fail("hour chứa giá trị ngoài [0, 23].")
    if not df["hour"].astype(int).eq(df["target_datetime"].dt.hour).all():
        fail("hour không khớp target_datetime.")
    dow = df["target_datetime"].dt.dayofweek
    if not df["dayofweek"].astype(int).eq(dow).all():
        fail("dayofweek không khớp target_datetime.")
    if not df["is_holiday"].isin([0, 1]).all():
        fail("is_holiday chỉ được chứa 0/1.")
    print("Calendar feature: PASS")

    lag_columns = tuple(f"lag_{lag}" for lag in panel["all_lags_hours"])
    for col in lag_columns:
        if df[col].isna().any():
            fail(f"{col} chứa NaN.")
        if (df[col] < 0).any():
            fail(f"{col} chứa giá trị âm.")
    print("Lag column bắt buộc / không có NaN: PASS")

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
            f"{panel['median_feature']} không bằng median weekly lag.\n"
            f"Số mismatch: {mismatch.sum():,}"
        )
    print("median_lag_3w chính xác: PASS")

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
            "Phát hiện column ngoài core scope:\n"
            + "\n".join(f"  - {c}" for c in sorted(set(suspicious)))
        )
    print("Core feature scope: PASS")
    print("FEATURE TABLE QA PASS")
    return df


def check_variant_map(config: DataConfig) -> None:
    """Validate feature contract A/B/C trong variant_feature_map.json."""
    print_header("5. QA VARIANT FEATURE MAP")
    panel = config["panel"]
    variant_map_path = config["paths"]["variant_map"]
    check_exists(variant_map_path, "variant_feature_map.json")

    with open(variant_map_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    if "base_features" not in record:
        fail("variant_feature_map.json thiếu 'base_features'.")
    if record["base_features"] != list(panel["base_features"]):
        fail(
            "base_features không đúng.\n"
            f"Quan sát: {record['base_features']}\n"
            f"Kỳ vọng: {list(panel['base_features'])}"
        )

    if "variants" not in record:
        fail("variant_feature_map.json thiếu 'variants'.")

    variants = record["variants"]
    for variant_name, expected in panel["variants"].items():
        if variant_name not in variants:
            fail(f"Thiếu Variant {variant_name}.")
        actual_features = variants[variant_name].get("weekly_features")
        if actual_features != list(expected["weekly_features"]):
            fail(
                f"Variant {variant_name} có weekly feature sai.\n"
                f"Quan sát: {actual_features}\n"
                f"Kỳ vọng: {list(expected['weekly_features'])}"
            )

    print("Variant A: lag_168 + lag_336 + lag_504: PASS")
    print("Variant B: median_lag_3w: PASS")
    print("Variant C: lag_168 + median_lag_3w: PASS")
    print("VARIANT FEATURE MAP QA PASS")


def check_lag_alignment_examples(config: DataConfig) -> None:
    """Validate evidence lag-alignment bền vững từ full panel."""
    print_header("6. QA LAG ALIGNMENT")
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
        fail(f"Lag example thiếu column: {sorted(missing)}")
    if len(examples) != 75:
        fail(f"Kỳ vọng 75 lag example, nhưng có {len(examples)}")
    if not examples["matches"].eq(True).all():
        fail("Lag example chứa mismatch")
    source_delta = examples["target_datetime"] - examples["source_datetime"]
    expected_delta = pd.to_timedelta(examples["lag"], unit="h")
    if not source_delta.eq(expected_delta).all():
        fail("Lag example chứa source timestamp không đúng")
    print("75 full-panel lag example và source timestamp: PASS")


def get_fold_paths(split_name: str, eval_kind: str, folds_dir: Path) -> tuple[Path, Path]:
    """Trả về path CSV train và evaluation của một split."""
    split_dir = folds_dir / split_name
    return split_dir / "train.csv", split_dir / f"{eval_kind}.csv"


def load_fold_data(
    split_name: str, split_cfg: SplitConfig, folds_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc file CSV train và evaluation/test của một split."""
    train_path, eval_path = get_fold_paths(
        split_name, split_cfg["eval_kind"], folds_dir
    )
    check_exists(train_path, f"{split_name}/train.csv")
    check_exists(eval_path, f"{split_name}/{split_cfg['eval_kind']}.csv")

    train = pd.read_csv(train_path, parse_dates=["target_datetime"])
    eval_df = pd.read_csv(eval_path, parse_dates=["target_datetime"])
    return train, eval_df


def check_fold_structure(zone_ids: list[int], config: DataConfig) -> dict[str, dict[str, pd.DataFrame]]:
    """Validate file, column, zone, window và value của mọi split."""
    print_header("7. QA FOLD DATA")
    fold_data = {}
    expected_feature_columns = None

    for split_name, split_cfg in config["splits"].items():
        print(f"\n--- {split_name.upper()} ---")
        train, eval_df = load_fold_data(split_name, split_cfg, config["paths"]["folds_dir"])

        if len(train) == 0:
            fail(f"[{split_name}] train.csv rỗng.")
        if len(eval_df) == 0:
            fail(f"[{split_name}] eval/test rỗng.")

        required = {"pu_location_id", "target_datetime", "demand"}
        for name, data in [("train", train), ("eval/test", eval_df)]:
            missing = required - set(data.columns)
            if missing:
                fail(
                    f"[{split_name}/{name}] thiếu column: "
                    f"{sorted(missing)}"
                )

        train_columns = list(train.columns)
        eval_columns = list(eval_df.columns)
        if train_columns != eval_columns:
            fail(f"[{split_name}] column train và eval/test khác nhau.")

        if expected_feature_columns is None:
            expected_feature_columns = train_columns
        elif train_columns != expected_feature_columns:
            fail(f"[{split_name}] column khác với fold khác.")

        train_zones = sorted(
            train["pu_location_id"].dropna().astype(int).unique().tolist()
        )
        eval_zones = sorted(
            eval_df["pu_location_id"].dropna().astype(int).unique().tolist()
        )
        expected_zones = sorted(zone_ids)
        if train_zones != expected_zones:
            fail(f"[{split_name}] train zone != frozen Top-50.")
        if eval_zones != expected_zones:
            fail(f"[{split_name}] eval/test zone != frozen Top-50.")

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
                f"[{split_name}] train bắt đầu ở thời điểm sai.\n"
                f"Quan sát : {train_min}\n"
                f"Kỳ vọng : {train_start_expected}"
            )
        expected_train_max = train_end_expected - pd.Timedelta(hours=1)
        if train_max != expected_train_max:
            fail(
                f"[{split_name}] train kết thúc ở thời điểm sai.\n"
                f"Quan sát : {train_max}\n"
                f"Kỳ vọng : {expected_train_max}"
            )
        if eval_min != eval_start_expected:
            fail(
                f"[{split_name}] eval/test bắt đầu ở thời điểm sai.\n"
                f"Quan sát : {eval_min}\n"
                f"Kỳ vọng : {eval_start_expected}"
            )
        expected_eval_max = eval_end_expected - pd.Timedelta(hours=1)
        if eval_max != expected_eval_max:
            fail(
                f"[{split_name}] eval/test kết thúc ở thời điểm sai.\n"
                f"Quan sát : {eval_max}\n"
                f"Kỳ vọng : {expected_eval_max}"
            )

        if train_max >= eval_min:
            fail(f"[{split_name}] train và eval/test bị overlap.")
        gap = eval_min - train_max
        if gap != pd.Timedelta(hours=1):
            fail(f"[{split_name}] train/eval không liền kề. Gap: {gap}")

        for name, data in [("train", train), ("eval/test", eval_df)]:
            duplicate_count = (
                data[["pu_location_id", "target_datetime"]]
                .duplicated()
                .sum()
            )
            if duplicate_count > 0:
                fail(
                    f"[{split_name}/{name}] có {duplicate_count:,} "
                    "row (zone, target_datetime) duplicate."
                )
            if data["demand"].isna().any():
                fail(f"[{split_name}/{name}] demand có NaN.")
            if (data["demand"] < 0).any():
                fail(f"[{split_name}/{name}] demand có giá trị âm.")

        print(f"Train : {len(train):,} row ({train_min} -> {train_max})")
        print(f"Eval  : {len(eval_df):,} row ({eval_min} -> {eval_max})")
        print("Zone vocabulary: PASS")
        print("Train/eval temporal separation: PASS")

        fold_data[split_name] = {"train": train, "eval": eval_df}

    print("\nFOLD STRUCTURE QA PASS")
    return fold_data


def check_temporal_leakage(
    fold_data: dict[str, dict[str, pd.DataFrame]], config: DataConfig
) -> None:
    """Validate cross-fold window, final-test isolation và expansion.

    Evaluation row của fold trước xuất hiện lại trong training window muộn hơn
    là yêu cầu của expanding-window design và không phải leakage.
    """
    print_header("8. QA TEMPORAL LEAKAGE GIỮA CÁC FOLD")
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
                    "Evaluation window bị overlap:\n"
                    f"{name_i}: {start_i} -> {end_i}\n"
                    f"{name_j}: {start_j} -> {end_j}"
                )
    print("Validation/test window không overlap: PASS")

    final_test = fold_data["final_test"]["eval"]
    final_test_start = final_test["target_datetime"].min()
    final_test_end = final_test["target_datetime"].max()
    expected_final_start = config["splits"]["final_test"]["eval_start"]
    expected_final_end = config["splits"]["final_test"]["eval_end_exclusive"] - pd.Timedelta(hours=1)

    if final_test_start != expected_final_start:
        fail("Final test không bắt đầu tại 2025-12-01.")
    if final_test_end != expected_final_end:
        fail("Final test không kết thúc tại 2025-12-31 23:00.")

    final_test_start = config["splits"]["final_test"]["eval_start"]
    for split_name in ordered_splits[:-1]:
        train = fold_data[split_name]["train"]
        eval_df = fold_data[split_name]["eval"]
        train_final_period = (train["target_datetime"] >= final_test_start).sum()
        eval_final_period = (eval_df["target_datetime"] >= final_test_start).sum()
        if train_final_period > 0 or eval_final_period > 0:
            fail(
                f"[{split_name}] dữ liệu December xuất hiện trước "
                "final_test.\n"
                f"Train row trong final-test period: {train_final_period:,}\n"
                f"Eval row trong final-test period : {eval_final_period:,}"
            )
    print("Tách riêng December final_test: PASS")

    previous_train_end = None
    for split_name in ordered_splits:
        train_end = config["splits"][split_name]["train_end_exclusive"]
        if previous_train_end is not None and train_end < previous_train_end:
            fail(f"[{split_name}] train window không mở rộng theo thời gian.")
        previous_train_end = train_end
    print("Expanding training windows: PASS")

    print("\nTEMPORAL LEAKAGE QA PASS")


def main():
    """Chạy toàn bộ QA/QC check và exit non-zero khi thất bại."""
    config = load_data_config()
    print("\n" + "=" * 75)
    print("FINAL DATA QA/QC")
    print("=" * 75)
    print(f"\nRepository:\n{PROJECT_ROOT}")
    print("\nGHI CHÚ:")
    print("- Raw Parquet schema, monthly aggregate và full-panel lag example được kiểm tra tại đây.")

    check_raw_inputs(config)
    aggregate = check_monthly_aggregates(config)
    zone_ids = check_top50_frozen(aggregate, config)
    check_feature_table(zone_ids, config)
    check_variant_map(config)
    check_lag_alignment_examples(config)
    fold_data = check_fold_structure(zone_ids, config)
    check_temporal_leakage(fold_data, config)

    print("\n" + "=" * 75)
    print("TOÀN BỘ FINAL DATA QA/QC CHECK ĐỀU PASS")
    print("=" * 75)
    print("\nĐã kiểm tra:")
    print("1. Raw Parquet schema")
    print("2. Monthly aggregate table")
    print("3. top50_zones_frozen.json")
    print("4. feature_table.csv")
    print("5. variant_feature_map.json")
    print("6. Full-panel lag-alignment example")
    print("7. hpo/fold1/fold2/fold3/fold4/final_test")
    print("8. Temporal leakage / tách riêng final_test")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("\n" + "=" * 75)
        print("QA/QC THẤT BẠI")
        print("=" * 75)
        print(f"\n{exc}\n")
        sys.exit(1)
