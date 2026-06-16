#!/usr/bin/env python3
"""Reproducible synthetic data generator for the Coffee Shop Profit Predictor.

This module makes the project's dataset reproducible and parameterizable. Given
a seed it deterministically samples location features from documented
distributions and computes ``profit`` from a transparent linear formula plus
Gaussian noise. The distributions and coefficients are calibrated to match the
project's bundled reference sample, so a model trained on generated data behaves
similarly to one trained on the committed CSVs.

Feature distributions (per location)
------------------------------------
- ``lat``              : Normal(52.52, 0.055)   - Berlin latitude band
- ``lon``              : Normal(13.41, 0.090)   - Berlin longitude band
- ``foot_traffic``     : Uniform[800, 8000]     - integer
- ``rent_per_sqm``     : Normal(30, 8), clipped to [10, 55]
- ``competition``      : Uniform integer [0, 11]
- ``median_income``    : Normal(2400, 520), clipped to [1200, 4000]
- ``office_density``   : Uniform integer [10, 399]
- ``weekend_activity`` : Uniform[0.20, 1.00]
- ``events_per_month`` : Uniform integer [0, 11]
- ``coffee_price``     : Normal(3.35, 0.55), clipped to [2.0, 5.5]
- ``promo_spend``      : Uniform integer [0, 2500]

Profit formula
--------------
``profit`` is a transparent linear combination of the raw features plus noise::

    profit = INTERCEPT
           + 0.08  * foot_traffic
           - 35.0  * rent_per_sqm
           - 47.0  * competition
           + 0.52  * median_income
           + 1.20  * office_density
           + 380.0 * weekend_activity
           + 100.0 * events_per_month
           + 103.0 * coffee_price
           + 0.17  * promo_spend
           + Normal(0, NOISE_STD)

The signs match the project's documented business priors: rent and competition
reduce profit, every other feature increases it. The noise term means profit is
not perfectly predictable, which is what makes the modelling task meaningful.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from coffee_shop_predictor.config import RANDOM_STATE, REQUIRED_TRAIN_COLUMNS
from coffee_shop_predictor.utils import save_csv, setup_logging, validate_dataframe

LOGGER = logging.getLogger(__name__)

DEFAULT_N_TRAIN: Final[int] = 220
DEFAULT_N_CANDIDATES: Final[int] = 60

# Linear coefficients used to synthesize profit from the raw features.
PROFIT_INTERCEPT: Final[float] = -1340.0
PROFIT_COEFFICIENTS: Final[dict[str, float]] = {
    "foot_traffic": 0.08,
    "rent_per_sqm": -35.0,
    "competition": -47.0,
    "median_income": 0.52,
    "office_density": 1.20,
    "weekend_activity": 380.0,
    "events_per_month": 100.0,
    "coffee_price": 103.0,
    "promo_spend": 0.17,
}
PROFIT_NOISE_STD: Final[float] = 470.0

# Column order matches the bundled CSV schema.
RAW_COLUMN_ORDER: Final[list[str]] = [
    "lat",
    "lon",
    "foot_traffic",
    "rent_per_sqm",
    "competition",
    "median_income",
    "office_density",
    "weekend_activity",
    "events_per_month",
    "coffee_price",
    "promo_spend",
]


def _sample_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample ``n`` rows of raw location features from the documented distributions."""
    data = {
        "lat": rng.normal(52.52, 0.055, n),
        "lon": rng.normal(13.41, 0.090, n),
        "foot_traffic": rng.integers(800, 8001, n),
        "rent_per_sqm": np.clip(rng.normal(30.0, 8.0, n), 10.0, 55.0).round(4),
        "competition": rng.integers(0, 12, n),
        "median_income": np.clip(rng.normal(2400.0, 520.0, n), 1200.0, 4000.0).round(4),
        "office_density": rng.integers(10, 400, n),
        "weekend_activity": rng.uniform(0.20, 1.00, n).round(6),
        "events_per_month": rng.integers(0, 12, n),
        "coffee_price": np.clip(
            rng.normal(3.35, 0.55, n),
            2.0,
            5.5,
        ).round(4),
        "promo_spend": rng.integers(0, 2501, n),
    }
    return pd.DataFrame(data)[RAW_COLUMN_ORDER]


def _compute_profit(features: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Apply the transparent linear profit formula plus Gaussian noise."""
    profit = np.full(len(features), PROFIT_INTERCEPT, dtype=float)
    for column, coefficient in PROFIT_COEFFICIENTS.items():
        profit += coefficient * features[column].to_numpy(dtype=float)
    profit += rng.normal(0.0, PROFIT_NOISE_STD, len(features))
    return profit.round(6)


def generate_training_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a training DataFrame with features and a synthesized ``profit`` target."""
    features = _sample_features(n, rng)
    features["profit"] = _compute_profit(features, rng)
    return features


def generate_candidate_data(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a candidate DataFrame (features only, no target)."""
    return _sample_features(n, rng)


def write_datasets(out_dir: Path, n_train: int, n_candidates: int, seed: int) -> tuple[Path, Path]:
    """Generate and validate both datasets, then write them as CSV files."""
    train_rng = np.random.default_rng(seed)
    # Derive a distinct but deterministic stream for candidates.
    candidate_rng = np.random.default_rng(seed + 1)

    train_df = generate_training_data(n_train, train_rng)
    candidate_df = generate_candidate_data(n_candidates, candidate_rng)

    validate_dataframe(train_df, REQUIRED_TRAIN_COLUMNS, "generated_train")
    validate_dataframe(candidate_df, set(RAW_COLUMN_ORDER) | {"lat", "lon"}, "generated_candidates")

    train_path = save_csv(train_df, out_dir / "locations_train.csv")
    candidate_path = save_csv(candidate_df, out_dir / "locations_candidates.csv")
    LOGGER.info("Wrote %s training rows to %s", n_train, train_path)
    LOGGER.info("Wrote %s candidate rows to %s", n_candidates, candidate_path)
    return train_path, candidate_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible synthetic location data.")
    parser.add_argument(
        "--n-train", type=int, default=DEFAULT_N_TRAIN, help="Number of training rows."
    )
    parser.add_argument(
        "--n-candidates", type=int, default=DEFAULT_N_CANDIDATES, help="Number of candidate rows."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_STATE, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Output directory. WARNING: defaults to data/, which overwrites the bundled CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    write_datasets(Path(args.out_dir), args.n_train, args.n_candidates, args.seed)


if __name__ == "__main__":
    main()
