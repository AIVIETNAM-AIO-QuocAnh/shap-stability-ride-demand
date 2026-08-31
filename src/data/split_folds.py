"""
Split feature table theo thời gian
- Toàn bộ split cố định theo target_datetime, không dùng random K-fold.
- 6 giai đoạn: HPO, Fold 1-4, Final test -- train luôn bắt đầu từ 22/01/2025,
  mở rộng dần; validation/test là 1 tháng liền kề ngay sau train.
- QA/QC: xác minh lag alignment bằng sample cụ thể trước khi cho phép chạy
  toàn bộ experiment.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ============ CONFIG ============
REPO_ROOT = Path(__file__).resolve().parents[2]

FEATURE_TABLE_PATH = REPO_ROOT / "data" / "processed" / "feature_table.csv"
FOLDS_DIR = REPO_ROOT / "data" / "folds"

ALL_LAGS = [1, 24, 168, 336, 504]

# Khoảng nửa-mở [start, end) -- end là ngày đầu tiên của tháng kế tiếp,
# tránh phải tính số ngày cuối tháng thủ công.
# eval_kind: "val" dùng cho HPO/Fold 1-4, "test" dùng cho final_test
SPLITS = {
    "hpo": {
        "train": ("2025-01-22", "2025-07-01"),
        "eval": ("2025-07-01", "2025-08-01"),
        "eval_kind": "val",
    },
    "fold1": {
        "train": ("2025-01-22", "2025-08-01"),
        "eval": ("2025-08-01", "2025-09-01"),
        "eval_kind": "val",
    },
    "fold2": {
        "train": ("2025-01-22", "2025-09-01"),
        "eval": ("2025-09-01", "2025-10-01"),
        "eval_kind": "val",
    },
    "fold3": {
        "train": ("2025-01-22", "2025-10-01"),
        "eval": ("2025-10-01", "2025-11-01"),
        "eval_kind": "val",
    },
    "fold4": {
        "train": ("2025-01-22", "2025-11-01"),
        "eval": ("2025-11-01", "2025-12-01"),
        "eval_kind": "val",
    },
    "final_test": {
        "train": ("2025-01-22", "2025-12-01"),
        "eval": ("2025-12-01", "2026-01-01"),
        "eval_kind": "test",
    },
}
# ==================================


def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["target_datetime"])
    print(f"Đã load feature table: {len(df):,} row, {df['PULocationID'].nunique()} zone.")
    return df


def verify_lag_alignment(panel: pd.DataFrame, n_samples: int = 30, seed: int = 42):
    """
    QA/QC: với sample ngẫu nhiên, tra lại trong feature table xem giá trị lag_N tại 
    row (zone, t) có đúng bằng demand tại row (zone, t - N giờ) hay không
    """
    print("=== QA/QC: XÁC MINH LAG ALIGNMENT ===")

    lookup = panel.set_index(["PULocationID", "target_datetime"])["demand"]

    # Chỉ check từ các row đủ xa mốc bắt đầu bảng để đảm bảo mọi lag đều tra cứu được
    min_dt = panel["target_datetime"].min()
    safe_start = min_dt + pd.Timedelta(hours=max(ALL_LAGS))
    eligible = panel[panel["target_datetime"] >= safe_start]

    if len(eligible) == 0:
        raise ValueError(
            "Không có row nào đủ xa mốc đầu bảng để verify lag alignment -- "
            "feature table quá ngắn so với max(lag)."
        )

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(eligible), size=min(n_samples, len(eligible)), replace=False)
    samples = eligible.iloc[sample_idx]

    n_checked = 0
    n_mismatch = 0
    mismatches = []

    for _, row in samples.iterrows():
        zone = row["PULocationID"]
        t = row["target_datetime"]

        for lag in ALL_LAGS:
            expected_time = t - pd.Timedelta(hours=lag)
            key = (zone, expected_time)

            if key not in lookup.index:
                mismatches.append(
                    f"  [MISSING] zone={zone}, t={t}, lag_{lag}: "
                    f"không tìm thấy row (zone, {expected_time}) trong panel."
                )
                n_mismatch += 1
                n_checked += 1
                continue

            expected_demand = lookup.loc[key]
            actual_lag_value = row[f"lag_{lag}"]

            n_checked += 1
            if expected_demand != actual_lag_value:
                mismatches.append(
                    f"  [MISMATCH] zone={zone}, t={t}, lag_{lag}: "
                    f"cột lag_{lag}={actual_lag_value} nhưng demand thực tế tại "
                    f"(zone, {expected_time}) = {expected_demand}."
                )
                n_mismatch += 1

    print(f"Đã kiểm tra {n_checked} phép so khớp (lag) trên {len(samples)} sample row.")

    if n_mismatch > 0:
        print(f"THẤT BẠI: phát hiện {n_mismatch} chỗ lag_alignment sai:")
        for m in mismatches[:20]:
            print(m)
        raise ValueError(
            "QA/QC lag alignment THẤT BẠI -- dừng lại, không chạy split/experiment "
            "cho tới khi sửa xong build_panel."
        )

    print("QA/QC lag alignment PASS -- toàn bộ sample khớp đúng.")


def split_one(panel: pd.DataFrame, split_name: str, split_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_start, train_end = pd.Timestamp(split_cfg["train"][0]), pd.Timestamp(split_cfg["train"][1])
    eval_start, eval_end = pd.Timestamp(split_cfg["eval"][0]), pd.Timestamp(split_cfg["eval"][1])

    train_df = panel[
        (panel["target_datetime"] >= train_start) & (panel["target_datetime"] < train_end)
    ].copy()
    eval_df = panel[
        (panel["target_datetime"] >= eval_start) & (panel["target_datetime"] < eval_end)
    ].copy()

    # Sanity: train và eval không được chồng lấn thời gian.
    assert train_df["target_datetime"].max() < eval_df["target_datetime"].min(), (
        f"[{split_name}] Train và eval bị chồng lấn thời gian!"
    )
    # Sanity: eval phải liền kề ngay sau train (không có gap, đúng thiết kế).
    gap = eval_df["target_datetime"].min() - train_df["target_datetime"].max()
    assert gap == pd.Timedelta(hours=1), (
        f"[{split_name}] Khoảng cách giữa train và eval = {gap}, kỳ vọng đúng 1 giờ (liền kề)."
    )

    print(
        f"[{split_name}] train: {len(train_df):,} row "
        f"({train_df['target_datetime'].min()} -> {train_df['target_datetime'].max()}), "
        f"eval: {len(eval_df):,} row "
        f"({eval_df['target_datetime'].min()} -> {eval_df['target_datetime'].max()})"
    )

    return train_df, eval_df


def save_split(train_df: pd.DataFrame, eval_df: pd.DataFrame, split_name: str, eval_kind: str):
    split_dir = FOLDS_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.csv"
    eval_path = split_dir / f"{eval_kind}.csv"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f"  Đã lưu -> {train_path.relative_to(REPO_ROOT)}, {eval_path.relative_to(REPO_ROOT)}")


def main():
    panel = load_feature_table(FEATURE_TABLE_PATH)

    # QA/QC bắt buộc TRƯỚC khi chạy toàn bộ split
    verify_lag_alignment(panel, n_samples=30, seed=42)

    print("\n=== SPLIT THEO 6 GIAI ĐOẠN ===")
    for split_name, split_cfg in SPLITS.items():
        train_df, eval_df = split_one(panel, split_name, split_cfg)
        save_split(train_df, eval_df, split_name, split_cfg["eval_kind"])

    print("\nHoàn tất. Toàn bộ split đã lưu tại:", FOLDS_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()