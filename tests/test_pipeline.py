from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from coffee_shop_predictor.config import DEFAULT_SQL_PATH
from coffee_shop_predictor.run_pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


class PipelineIntegrationTests(unittest.TestCase):
    def test_end_to_end_produces_scored_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outdir = tmp_path / "outputs"
            run_pipeline(
                train_csv=DATA_DIR / "locations_train.csv",
                candidates_csv=DATA_DIR / "locations_candidates.csv",
                db_path=tmp_path / "coffee.db",
                sql_path=DEFAULT_SQL_PATH,
                outdir=outdir,
            )

            scored_path = outdir / "scored_candidates.csv"
            self.assertTrue(scored_path.exists())

            scored = pd.read_csv(scored_path)
            self.assertGreater(len(scored), 0)
            for column in [
                "rank",
                "predicted_profit",
                "prediction_low_eur",
                "prediction_high_eur",
                "risk_band",
            ]:
                self.assertIn(column, scored.columns)

            # Lower bound must sit below the upper bound for every candidate.
            self.assertTrue((scored["prediction_low_eur"] < scored["prediction_high_eur"]).all())


if __name__ == "__main__":
    unittest.main()
