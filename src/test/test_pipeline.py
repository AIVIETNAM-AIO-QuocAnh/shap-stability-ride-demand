"""Focused real-data and protocol checks for the Pipeline contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.artifacts import save_prediction_artifacts
from src.pipeline.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_absolute_percentage_error,
)
from src.pipeline.run_shap import generate_sample_keys
from src.pipeline.summarize import _feature_stability, _group_stability, _performance_summary
from src.utilities import load_data


class PipelineContractTests(unittest.TestCase):
    """Exercise the semantic data, metric, sampling, and artifact contracts."""

    def test_model_features_exclude_raw_zone_identifier(self) -> None:
        """Ensure raw zone identifiers stay semantic keys, not model features."""
        data = load_data("hpo", "A")
        self.assertNotIn("pu_location_id", data["X_train"].columns)
        self.assertEqual(len(data["X_train"].columns), 58)
        self.assertTrue(pd.api.types.is_integer_dtype(data["evaluation_keys"]["pu_location_id"]))

    def test_locked_metrics_use_percent_wape(self) -> None:
        """Ensure WAPE follows the locked percentage definition."""
        actual = np.array([100.0, 200.0])
        predicted = np.array([90.0, 220.0])
        self.assertAlmostEqual(mean_absolute_error(actual, predicted), 15.0)
        self.assertAlmostEqual(root_mean_squared_error(actual, predicted), np.sqrt(250.0))
        self.assertAlmostEqual(weighted_absolute_percentage_error(actual, predicted), 10.0)

    def test_semantic_sample_keys_are_deterministic(self) -> None:
        """Ensure one seeded fold produces stable, unique zone samples."""
        data = load_data("fold1", "A")
        first = generate_sample_keys("fold1", data)
        second = generate_sample_keys("fold1", data)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 5000)
        self.assertTrue(first["pu_location_id"].value_counts().eq(100).all())
        self.assertFalse(first[["pu_location_id", "target_datetime"]].duplicated().any())

    def test_prediction_artifacts_are_keyed_and_non_overwriting(self) -> None:
        """Ensure keyed prediction files allow existing directories but not overwrites."""
        keys = pd.DataFrame(
            {
                "pu_location_id": [1, 2],
                "target_datetime": pd.to_datetime(["2025-07-01 00:00", "2025-07-01 01:00"]),
            }
        )
        actual = pd.Series([100.0, 200.0])
        predicted = np.array([90.0, 220.0])
        metrics = {
            "mae": 15.0,
            "rmse": float(np.sqrt(250.0)),
            "wape": 10.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / ".gitkeep").touch()
            save_prediction_artifacts(directory, keys, actual, predicted, metrics)
            written = pd.read_csv(directory / "y_pred.csv", parse_dates=["target_datetime"])
            self.assertEqual(
                list(written.columns),
                ["pu_location_id", "target_datetime", "y_true", "y_pred"],
            )
            with self.assertRaises(FileExistsError):
                save_prediction_artifacts(directory, keys, actual, predicted, metrics)

    def test_core_summaries_use_sample_std_and_exclude_final_test(self) -> None:
        """Ensure fold summaries use sample std and keep final_test separate."""
        folds = ["fold1", "fold2", "fold3", "fold4", "final_test"]
        metrics = pd.DataFrame(
            {
                "variant": ["A"] * 5,
                "model": ["xgboost"] * 5,
                "fold": folds,
                "mae": [1.0, 2.0, 3.0, 4.0, 10.0],
                "rmse": [2.0, 3.0, 4.0, 5.0, 11.0],
                "wape": [10.0, 20.0, 30.0, 40.0, 100.0],
            }
        )
        performance = _performance_summary(metrics).iloc[0]
        self.assertAlmostEqual(float(performance["mae_mean"]), 2.5)
        self.assertAlmostEqual(float(performance["mae_std"]), np.std([1, 2, 3, 4], ddof=1))
        self.assertAlmostEqual(float(performance["mae_final_test"]), 10.0)

        feature_rows = pd.DataFrame(
            {
                "variant": ["A"] * 5,
                "model": ["xgboost"] * 5,
                "fold": folds,
                "feature": ["lag_168"] * 5,
                "importance": [1.0, 2.0, 3.0, 4.0, 10.0],
            }
        )
        feature = _feature_stability(feature_rows).iloc[0]
        self.assertAlmostEqual(float(feature["mean_importance"]), 2.5)
        self.assertAlmostEqual(float(feature["std_importance"]), np.std([1, 2, 3, 4], ddof=1))
        self.assertAlmostEqual(float(feature["importance_final_test"]), 10.0)

        group_rows = feature_rows.rename(columns={"feature": "unused"}).drop(columns=["unused"])
        group_rows = group_rows.rename(columns={"importance": "weekly_group_importance"})
        group = _group_stability(group_rows).iloc[0]
        self.assertAlmostEqual(float(group["mean_group_importance"]), 2.5)
        self.assertAlmostEqual(
            float(group["std_group_importance"]), np.std([1, 2, 3, 4], ddof=1)
        )
        self.assertAlmostEqual(float(group["group_importance_final_test"]), 10.0)


if __name__ == "__main__":
    unittest.main()
