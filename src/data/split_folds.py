"""Split the final feature table into the locked temporal windows."""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.configuration import PROJECT_ROOT, DataConfig, SplitConfig, load_data_config


def load_feature_table(path: Path) -> pd.DataFrame:
    """Read the final feature table."""
    df = pd.read_csv(path, parse_dates=["target_datetime"])
    print(f"Loaded feature table: {len(df):,} rows, {df['pu_location_id'].nunique()} zones.")
    return df


def verify_lag_alignment(
    panel: pd.DataFrame, examples_path: Path, all_lags: tuple[int, ...]
) -> None:
    """Check retained lags and the full-panel pre-warm-up examples."""
    print("=== QA/QC: LAG ALIGNMENT ===")
    for lag in all_lags:
        expected = panel.groupby("pu_location_id", sort=False)["demand"].shift(lag)
        eligible = expected.notna()
        actual = panel.loc[eligible, f"lag_{lag}"]
        if not actual.equals(expected.loc[eligible].astype(actual.dtype)):
            raise ValueError(f"Alignment for lag_{lag} is incorrect in retained rows")

    if not examples_path.exists():
        raise FileNotFoundError(f"Missing full-panel lag examples: {examples_path}")
    examples = pd.read_csv(examples_path, parse_dates=["target_datetime", "source_datetime"])
    if len(examples) != 75 or not examples["matches"].eq(True).all():
        raise ValueError("Lag-alignment examples are incomplete or contain mismatches")
    print("Lag alignment PASS for retained rows and 75 full-panel examples.")


def split_one(
    panel: pd.DataFrame, split_name: str, split_cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_start = split_cfg["train_start"]
    train_end = split_cfg["train_end_exclusive"]
    eval_start = split_cfg["eval_start"]
    eval_end = split_cfg["eval_end_exclusive"]

    train_df = panel[
        (panel["target_datetime"] >= train_start) & (panel["target_datetime"] < train_end)
    ].copy()
    eval_df = panel[
        (panel["target_datetime"] >= eval_start) & (panel["target_datetime"] < eval_end)
    ].copy()

    # Training and evaluation windows must not overlap.
    if train_df["target_datetime"].max() >= eval_df["target_datetime"].min():
        raise ValueError(f"[{split_name}] training and evaluation windows overlap")
    # Evaluation must start at the immediately following hour.
    gap = eval_df["target_datetime"].min() - train_df["target_datetime"].max()
    if gap != pd.Timedelta(hours=1):
        raise ValueError(f"[{split_name}] training/evaluation gap is {gap}; expected 1 hour")

    tqdm.write(
        f"[{split_name}] train: {len(train_df):,} rows "
        f"({train_df['target_datetime'].min()} -> {train_df['target_datetime'].max()}), "
        f"eval: {len(eval_df):,} rows "
        f"({eval_df['target_datetime'].min()} -> {eval_df['target_datetime'].max()})"
    )

    return train_df, eval_df


def save_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    split_name: str,
    eval_kind: str,
    folds_dir: Path,
    repo_root: Path,
) -> None:
    split_dir = folds_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.csv"
    eval_path = split_dir / f"{eval_kind}.csv"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    tqdm.write(f"  Saved -> {train_path.relative_to(repo_root)}, {eval_path.relative_to(repo_root)}")


def main():
    config: DataConfig = load_data_config()
    paths = config["paths"]
    panel_config = config["panel"]
    panel = load_feature_table(paths["feature_table"])

    # Full-panel lag evidence must pass before temporal splits are written.
    verify_lag_alignment(panel, paths["lag_examples"], panel_config["all_lags_hours"])

    print("\n=== SIX LOCKED TEMPORAL SPLITS ===")
    for split_name, split_cfg in tqdm(
        config["splits"].items(),
        total=len(config["splits"]),
        desc="Writing temporal splits",
        unit="split",
    ):
        train_df, eval_df = split_one(panel, split_name, split_cfg)
        save_split(
            train_df,
            eval_df,
            split_name,
            split_cfg["eval_kind"],
            paths["folds_dir"],
            PROJECT_ROOT,
        )

    print("\nComplete. Splits saved under:", paths["folds_dir"].relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
