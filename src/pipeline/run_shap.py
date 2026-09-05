"""Generate stable semantic SHAP samples and per-run SHAP artifacts."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.load_config import load_data_config, load_model_config
from src.pipeline.artifacts import (
    load_pickle_model,
    package_snapshot,
    save_shap_values,
    write_json,
)
from src.utilities import ModelData, load_frozen_zone_ids


logger = logging.getLogger(__name__)


def generate_sample_keys(fold: str, data: ModelData) -> pd.DataFrame:
    """Select exactly 100 deterministic semantic rows for every frozen zone."""
    data_config = load_data_config()
    model_config = load_model_config()
    sample_size = model_config["shap"]["zone_sample_size"]
    keys = data["evaluation_keys"].reset_index(drop=True)
    if list(keys.columns) != ["pu_location_id", "target_datetime"]:
        raise ValueError("Evaluation keys have an unexpected schema")
    rng = np.random.default_rng(model_config["seed"])
    selected: list[pd.DataFrame] = []
    zone_ids = load_frozen_zone_ids(data_config)
    for zone_id in zone_ids:
        zone_rows = keys[keys["pu_location_id"] == zone_id].sort_values("target_datetime")
        if len(zone_rows) < sample_size:
            raise ValueError(
                f"Fold {fold} zone {zone_id} has {len(zone_rows)} rows; "
                f"exactly {sample_size} are required"
            )
        positions = np.sort(rng.choice(len(zone_rows), size=sample_size, replace=False))
        selected.append(zone_rows.iloc[positions])
    sample_keys = pd.concat(selected, ignore_index=True)
    sample_keys.insert(0, "sample_order", np.arange(len(sample_keys), dtype=int))
    if len(sample_keys) != len(zone_ids) * sample_size:
        raise RuntimeError(f"Fold {fold} did not produce the configured SHAP sample size")
    if sample_keys[["pu_location_id", "target_datetime"]].duplicated().any():
        raise RuntimeError(f"Fold {fold} produced duplicate semantic SHAP keys")
    return sample_keys


def save_sample_keys(fold: str, data: ModelData) -> Path:
    """Write one fold's semantic SHAP sample keys without replacement."""
    config = load_data_config()
    path = config["paths"]["folds_dir"] / fold / "shap_sample_keys.csv"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    sample_keys = generate_sample_keys(fold, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_keys.to_csv(path, index=False)
    logger.info("shap_sample_saved", extra={"fold": fold, "rows": len(sample_keys), "path": str(path)})
    return path


def load_sample_keys(fold: str) -> pd.DataFrame:
    """Read and validate one semantic SHAP sample-key artifact."""
    config = load_data_config()
    path = config["paths"]["folds_dir"] / fold / "shap_sample_keys.csv"
    if not path.is_file():
        raise FileNotFoundError(f"SHAP sample keys not found: {path}")
    sample_keys = pd.read_csv(path, parse_dates=["target_datetime"])
    expected_columns = ["sample_order", "pu_location_id", "target_datetime"]
    if list(sample_keys.columns) != expected_columns:
        raise ValueError(f"SHAP sample keys must contain exactly {expected_columns}")
    if not sample_keys["sample_order"].eq(np.arange(len(sample_keys))).all():
        raise ValueError(f"SHAP sample order is not contiguous: {path}")
    return sample_keys


def _sample_frame(data: ModelData, sample_keys: pd.DataFrame) -> pd.DataFrame:
    """Resolve semantic keys to the ordered model feature rows."""
    evaluation_keys = data["evaluation_keys"].reset_index(drop=True).reset_index()
    evaluation_keys = evaluation_keys.rename(columns={"index": "row_position"})
    joined = sample_keys.merge(
        evaluation_keys,
        on=["pu_location_id", "target_datetime"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if joined["row_position"].isna().any():
        raise KeyError("SHAP sample contains keys absent from the evaluation fold")
    positions = joined["row_position"].astype(int).tolist()
    return data["X_evaluation"].iloc[positions].reset_index(drop=True)


def _save_bar_plot(path: Path, shap_values: shap.Explanation) -> None:
    """Save one SHAP bar plot without replacing an existing artifact."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def _save_beeswarm_plot(path: Path, shap_values: shap.Explanation) -> None:
    """Save one SHAP beeswarm plot without replacing an existing artifact."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def run_shap(fold: str, variant: str, model_key: str, data: ModelData) -> None:
    """Compute and persist SHAP artifacts for one trained model run."""
    data_config = load_data_config()
    model_config = load_model_config()
    if variant not in data_config["panel"]["variants"]:
        raise ValueError(f"Unknown variant '{variant}'")
    model_dir = model_config["paths"]["results"] / variant / fold / model_key
    model = load_pickle_model(model_dir / "model.pkl")
    sample_keys = load_sample_keys(fold)
    X_sample = _sample_frame(data, sample_keys)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X_sample)
    if not isinstance(shap_values, shap.Explanation):
        raise TypeError("TreeExplainer did not return a shap.Explanation")
    if shap_values.values.ndim != 2 or shap_values.values.shape[1] != len(X_sample.columns):
        raise ValueError("SHAP output shape does not match the sampled feature matrix")
    save_shap_values(model_dir / "shap_values.pkl", shap_values)
    _save_bar_plot(model_dir / "shap_bar.png", shap_values)
    _save_beeswarm_plot(model_dir / "shap_beeswarm.png", shap_values)

    importance = np.abs(shap_values.values).mean(axis=0)
    importance_frame = pd.DataFrame(
        {"feature": X_sample.columns, "importance": importance}
    ).sort_values("importance", ascending=False)
    importance_path = model_dir / "shap_importance.csv"
    if importance_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {importance_path}")
    importance_frame.to_csv(importance_path, index=False)
    weekly_features = data_config["panel"]["variants"][variant]["weekly_features"]
    weekly_importance = float(
        importance_frame[importance_frame["feature"].isin(weekly_features)]["importance"].sum()
    )
    write_json(
        model_dir / "shap_weekly_group.json",
        {"weekly_group_importance": weekly_importance},
    )
    write_json(
        model_dir / "shap_config.json",
        {
            "stage": "shap",
            "model": model_key,
            "variant": variant,
            "fold": fold,
            "seed": model_config["seed"],
            "zone_sample_size": model_config["shap"]["zone_sample_size"],
            "feature_perturbation": "tree_path_dependent",
            "package_versions": package_snapshot(),
        },
    )
    logger.info("shap_completed", extra={"model": model_key, "variant": variant, "fold": fold})
