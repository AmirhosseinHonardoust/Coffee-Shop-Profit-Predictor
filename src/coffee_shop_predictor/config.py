#!/usr/bin/env python3
"""Central configuration for the Coffee Shop Profit Predictor project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Location of this installed package, used to locate bundled data files
# (e.g. queries.sql) independently of the current working directory.
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
DEFAULT_SQL_PATH: Final[Path] = PACKAGE_DIR / "queries.sql"

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.25
CV_SPLITS: Final[int] = 5
TARGET: Final[str] = "profit"

RAW_FEATURES: Final[list[str]] = [
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

ENGINEERED_FEATURES: Final[list[str]] = [
    "demand_adj",
    "wknd_traffic",
    "price_income",
    "promo_comp_adj",
]

FEATURES: Final[list[str]] = RAW_FEATURES + ENGINEERED_FEATURES
REQUIRED_TRAIN_COLUMNS: Final[set[str]] = set(RAW_FEATURES + ["lat", "lon", TARGET])
REQUIRED_CANDIDATE_COLUMNS: Final[set[str]] = set(RAW_FEATURES + ["lat", "lon"])

# Lightweight plausibility checks. These protect against impossible values while
# avoiding overly strict business assumptions for a portfolio/demo dataset.
COLUMN_RANGES: Final[dict[str, tuple[float | None, float | None]]] = {
    "lat": (-90, 90),
    "lon": (-180, 180),
    "foot_traffic": (0, None),
    "rent_per_sqm": (0, None),
    "competition": (0, None),
    "median_income": (0, None),
    "office_density": (0, None),
    "weekend_activity": (0, 1),
    "events_per_month": (0, None),
    "coffee_price": (0, None),
    "promo_spend": (0, None),
    TARGET: (None, None),
}

# Human-readable labels used in candidate explanations.
FEATURE_LABELS: Final[dict[str, str]] = {
    "foot_traffic": "foot traffic",
    "rent_per_sqm": "rent cost",
    "competition": "local competition",
    "median_income": "local income",
    "office_density": "office density",
    "weekend_activity": "weekend activity",
    "events_per_month": "event density",
    "coffee_price": "coffee price",
    "promo_spend": "promo spend",
    "demand_adj": "competition-adjusted demand",
    "wknd_traffic": "weekend traffic",
    "price_income": "price-income fit",
    "promo_comp_adj": "competition-adjusted promotion",
}

# Directional business priors for explanation text. They are not causal claims.
POSITIVE_DRIVER_FEATURES: Final[set[str]] = {
    "foot_traffic",
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
}
NEGATIVE_DRIVER_FEATURES: Final[set[str]] = {"rent_per_sqm", "competition"}


@dataclass(frozen=True)
class OutputFiles:
    metrics: str = "metrics.json"
    model: str = "model.joblib"
    metadata: str = "model_metadata.json"
    model_comparison: str = "model_comparison.csv"
    feature_importance: str = "feature_importance.csv"
    predictions_test: str = "predictions_test.csv"
    predictions_all: str = "predictions_all.csv"
    scored_candidates: str = "scored_candidates.csv"


OUTPUT_FILES: Final[OutputFiles] = OutputFiles()
