from __future__ import annotations

import unittest

import pandas as pd

from coffee_shop_predictor.config import FEATURES
from coffee_shop_predictor.score_new_sites import (
    _candidate_drivers,
    _feature_zscores,
    _interval_half_width,
    _risk_band,
)


def _standard_metadata() -> dict:
    """Metadata with zero mean and unit std for every feature.

    With this metadata a feature's z-score equals its raw value, which makes
    driver/risk behaviour easy to control in tests.
    """
    return {
        "feature_means": {f: 0.0 for f in FEATURES},
        "feature_stds": {f: 1.0 for f in FEATURES},
        "risk_distance_q75": 0.8,
        "risk_distance_q95": 1.2,
        "expected_mae_eur": 400.0,
    }


def _row(**overrides: float) -> pd.Series:
    values = {f: 0.0 for f in FEATURES}
    values.update(overrides)
    return pd.Series(values)


class FeatureZScoreTests(unittest.TestCase):
    def test_uses_metadata_mean_and_std(self) -> None:
        metadata = {
            "feature_means": {f: 10.0 for f in FEATURES},
            "feature_stds": {f: 2.0 for f in FEATURES},
        }
        zscores = _feature_zscores(_row(**{f: 14.0 for f in FEATURES}), metadata)
        for value in zscores.values():
            self.assertAlmostEqual(value, 2.0)

    def test_zero_std_falls_back_to_one(self) -> None:
        metadata = {
            "feature_means": {f: 5.0 for f in FEATURES},
            "feature_stds": {f: 0.0 for f in FEATURES},
        }
        zscores = _feature_zscores(_row(**{f: 8.0 for f in FEATURES}), metadata)
        for value in zscores.values():
            self.assertAlmostEqual(value, 3.0)


class RiskBandTests(unittest.TestCase):
    def test_low_band_below_q75(self) -> None:
        band, distance = _risk_band({"a": 0.5}, _standard_metadata())
        self.assertEqual(band, "low")
        self.assertAlmostEqual(distance, 0.5)

    def test_medium_band_between_thresholds(self) -> None:
        band, _ = _risk_band({"a": 1.0}, _standard_metadata())
        self.assertEqual(band, "medium")

    def test_high_band_above_q95(self) -> None:
        band, _ = _risk_band({"a": 1.5}, _standard_metadata())
        self.assertEqual(band, "high")

    def test_distance_is_mean_absolute_zscore(self) -> None:
        _, distance = _risk_band({"a": 1.0, "b": -3.0}, _standard_metadata())
        self.assertAlmostEqual(distance, 2.0)


class IntervalTests(unittest.TestCase):
    def test_uses_metadata_half_width(self) -> None:
        metadata = _standard_metadata()
        metadata["interval_half_width_eur"] = 500.0
        self.assertAlmostEqual(_interval_half_width(metadata), 500.0)

    def test_falls_back_to_scaled_mae(self) -> None:
        metadata = {"expected_mae_eur": 400.0}
        self.assertAlmostEqual(_interval_half_width(metadata), 640.0)


class CandidateDriverTests(unittest.TestCase):
    def test_high_positive_feature_is_a_positive_driver(self) -> None:
        positive, _ = _candidate_drivers(_row(foot_traffic=1.5), _standard_metadata())
        self.assertIn("foot traffic", positive)

    def test_high_competition_is_a_negative_driver(self) -> None:
        positive, negative = _candidate_drivers(_row(competition=1.5), _standard_metadata())
        self.assertIn("local competition", negative)
        self.assertEqual(positive, "balanced profile")

    def test_low_rent_is_a_positive_driver(self) -> None:
        positive, _ = _candidate_drivers(_row(rent_per_sqm=-1.5), _standard_metadata())
        self.assertIn("rent cost", positive)

    def test_balanced_profile_when_all_near_mean(self) -> None:
        positive, negative = _candidate_drivers(_row(), _standard_metadata())
        self.assertEqual(positive, "balanced profile")
        self.assertEqual(negative, "no major red flags")

    def test_drivers_capped_at_top_n(self) -> None:
        row = _row(foot_traffic=3.0, median_income=2.5, office_density=2.0, coffee_price=1.5)
        positive, _ = _candidate_drivers(row, _standard_metadata(), top_n=2)
        self.assertEqual(positive.count(";"), 1)  # two drivers => one separator


if __name__ == "__main__":
    unittest.main()
