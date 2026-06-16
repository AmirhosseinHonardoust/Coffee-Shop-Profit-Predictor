from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from coffee_shop_predictor.config import DEFAULT_SQL_PATH
from coffee_shop_predictor.create_db import load_csvs_to_db
from coffee_shop_predictor.generate_data import write_datasets
from coffee_shop_predictor.train_regression import run_training


class TrainingSmokeTests(unittest.TestCase):
    def test_training_writes_artifacts_and_beats_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_datasets(tmp_path, n_train=160, n_candidates=20, seed=123)
            db_path = tmp_path / "coffee.db"
            load_csvs_to_db(
                tmp_path / "locations_train.csv",
                tmp_path / "locations_candidates.csv",
                db_path,
            )
            outdir = tmp_path / "outputs"

            start = time.perf_counter()
            run_training(db_path, DEFAULT_SQL_PATH, outdir)
            elapsed = time.perf_counter() - start

            expected_files = [
                "metrics.json",
                "model.joblib",
                "model_metadata.json",
                "model_comparison.csv",
                "feature_importance.csv",
                "predictions_test.csv",
                "predictions_all.csv",
            ]
            for name in expected_files:
                self.assertTrue((outdir / name).exists(), f"missing {name}")

            for chart in [
                "actual_vs_predicted.png",
                "residuals_hist.png",
                "feature_importance.png",
                "model_comparison_mae.png",
            ]:
                self.assertTrue((outdir / "charts" / chart).exists(), f"missing chart {chart}")

            metrics = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
            holdout = metrics["holdout"]
            # The selected model must improve over a mean baseline.
            self.assertLess(holdout["mae"], holdout["baseline_mae"])
            self.assertGreater(holdout["r2"], holdout["baseline_r2"])
            # Guard against the smoke test silently becoming slow.
            self.assertLess(elapsed, 60.0)


if __name__ == "__main__":
    unittest.main()
