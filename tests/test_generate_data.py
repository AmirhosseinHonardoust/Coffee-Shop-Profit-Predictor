from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from coffee_shop_predictor.config import REQUIRED_TRAIN_COLUMNS
from coffee_shop_predictor.generate_data import (
    RAW_COLUMN_ORDER,
    generate_candidate_data,
    generate_training_data,
)
from coffee_shop_predictor.utils import validate_dataframe


class GenerateDataTests(unittest.TestCase):
    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        a = generate_training_data(50, np.random.default_rng(42))
        b = generate_training_data(50, np.random.default_rng(42))
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = generate_training_data(50, np.random.default_rng(1))
        b = generate_training_data(50, np.random.default_rng(2))
        self.assertFalse(a.equals(b))

    def test_training_schema_and_validation(self) -> None:
        df = generate_training_data(100, np.random.default_rng(0))
        self.assertEqual(list(df.columns), RAW_COLUMN_ORDER + ["profit"])
        self.assertEqual(len(df), 100)
        # Must pass the same validation used when loading into SQLite.
        validate_dataframe(df, REQUIRED_TRAIN_COLUMNS, "generated_train")

    def test_candidates_have_no_target(self) -> None:
        df = generate_candidate_data(30, np.random.default_rng(0))
        self.assertEqual(list(df.columns), RAW_COLUMN_ORDER)
        self.assertNotIn("profit", df.columns)
        self.assertEqual(len(df), 30)

    def test_weekend_activity_within_unit_interval(self) -> None:
        df = generate_training_data(200, np.random.default_rng(7))
        self.assertTrue(((df["weekend_activity"] >= 0.0) & (df["weekend_activity"] <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
