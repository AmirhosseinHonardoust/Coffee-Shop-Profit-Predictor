from __future__ import annotations

import py_compile
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
FEATURES = [
    "foot_traffic",
    "rent_per_sqm",
    "competition",
    "median_income",
    "office_density",
    "weekend_activity",
    "events_per_month",
    "coffee_price",
    "promo_spend",
    "demand_adj",
    "wknd_traffic",
    "price_income",
    "promo_comp_adj",
]


def _run_command(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, timeout=30)


class WorkflowTests(unittest.TestCase):
    def test_source_files_compile(self) -> None:
        for path in SRC_DIR.glob("*.py"):
            py_compile.compile(str(path), doraise=True)

    def test_create_db_and_sql_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "coffee.db"

            _run_command(
                [
                    sys.executable,
                    "src/create_db.py",
                    "--train",
                    "data/locations_train.csv",
                    "--candidates",
                    "data/locations_candidates.csv",
                    "--db",
                    str(db_path),
                ],
                PROJECT_ROOT,
            )

            with closing(sqlite3.connect(db_path)) as con:
                table_names = {
                    row[0]
                    for row in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table';"
                    ).fetchall()
                }
                self.assertIn("locations_train", table_names)
                self.assertIn("locations_candidates", table_names)

                con.executescript((SRC_DIR / "queries.sql").read_text(encoding="utf-8"))
                train_features = pd.read_sql_query("SELECT * FROM features_train;", con)
                candidate_features = pd.read_sql_query("SELECT * FROM features_candidates;", con)

            for feature in FEATURES:
                self.assertIn(feature, train_features.columns)
                self.assertIn(feature, candidate_features.columns)
            self.assertEqual(len(train_features), 220)
            self.assertEqual(len(candidate_features), 60)
            self.assertFalse(train_features[FEATURES + ["profit"]].isna().any().any())
            self.assertFalse(candidate_features[FEATURES].isna().any().any())

    def test_validation_rejects_impossible_values(self) -> None:
        sys.path.insert(0, str(SRC_DIR))
        from config import REQUIRED_TRAIN_COLUMNS
        from utils import validate_dataframe

        df = pd.read_csv(PROJECT_ROOT / "data" / "locations_train.csv").head(3).copy()
        df.loc[0, "foot_traffic"] = -1
        with self.assertRaises(ValueError):
            validate_dataframe(df, REQUIRED_TRAIN_COLUMNS, "bad_train")


if __name__ == "__main__":
    unittest.main()
