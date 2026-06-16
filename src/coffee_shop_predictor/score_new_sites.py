#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from coffee_shop_predictor.config import (
    DEFAULT_SQL_PATH,
    FEATURE_LABELS,
    FEATURES,
    NEGATIVE_DRIVER_FEATURES,
    OUTPUT_FILES,
    POSITIVE_DRIVER_FEATURES,
)
from coffee_shop_predictor.utils import (
    ensure_outdir,
    load_json,
    save_csv,
    setup_logging,
    validate_dataframe,
)

LOGGER = logging.getLogger(__name__)


def _load_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path.with_name(OUTPUT_FILES.metadata)
    if metadata_path.exists():
        return load_json(metadata_path)
    LOGGER.warning(
        "Model metadata not found at %s. Risk bands will use fallback values.", metadata_path
    )
    return {
        "features": FEATURES,
        "feature_means": {},
        "feature_stds": {},
        "risk_distance_q75": 0.8,
        "risk_distance_q95": 1.2,
        "expected_mae_eur": 400.0,
        "interval_coverage": 0.80,
        "interval_half_width_eur": 640.0,
    }


def _feature_zscores(row: pd.Series, metadata: dict[str, Any]) -> dict[str, float]:
    zscores: dict[str, float] = {}
    for feature in FEATURES:
        mean = float(metadata.get("feature_means", {}).get(feature, row.get(feature, 0.0)))
        std = float(metadata.get("feature_stds", {}).get(feature, 1.0)) or 1.0
        zscores[feature] = float((row[feature] - mean) / std)
    return zscores


def _risk_band(zscores: dict[str, float], metadata: dict[str, Any]) -> tuple[str, float]:
    """Classify how far a candidate's feature profile is from the training data.

    This is an extrapolation-risk signal (how unusual the candidate looks), not a
    prediction interval; uncertainty around the point estimate is reported
    separately via the conformal interval.
    """
    distance = sum(abs(v) for v in zscores.values()) / max(len(zscores), 1)
    q75 = float(metadata.get("risk_distance_q75", 0.8))
    q95 = float(metadata.get("risk_distance_q95", 1.2))

    if distance <= q75:
        return "low", float(distance)
    if distance <= q95:
        return "medium", float(distance)
    return "high", float(distance)


def _interval_half_width(metadata: dict[str, Any]) -> float:
    """Conformal prediction-interval half-width in euros, with a safe fallback."""
    return float(
        metadata.get("interval_half_width_eur", 1.6 * metadata.get("expected_mae_eur", 400.0))
    )


def _candidate_drivers(
    row: pd.Series, metadata: dict[str, Any], *, top_n: int = 3
) -> tuple[str, str]:
    zscores = _feature_zscores(row, metadata)

    positive: list[tuple[float, str]] = []
    negative: list[tuple[float, str]] = []

    for feature, z in zscores.items():
        label = FEATURE_LABELS.get(feature, feature)
        if feature in POSITIVE_DRIVER_FEATURES:
            if z >= 0.5:
                positive.append((abs(z), f"high {label}"))
            elif z <= -0.5:
                negative.append((abs(z), f"low {label}"))
        elif feature in NEGATIVE_DRIVER_FEATURES:
            if z >= 0.5:
                negative.append((abs(z), f"high {label}"))
            elif z <= -0.5:
                positive.append((abs(z), f"low {label}"))

    positive_text = "; ".join(text for _, text in sorted(positive, reverse=True)[:top_n])
    negative_text = "; ".join(text for _, text in sorted(negative, reverse=True)[:top_n])
    return positive_text or "balanced profile", negative_text or "no major red flags"


def run_scoring(db_path: Path, sql_path: Path, model_path: Path, outdir: Path) -> None:
    """Score candidate locations using a trained pipeline."""
    outdir = ensure_outdir(outdir)

    sql_text = sql_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as con:
        con.executescript(sql_text)
        X_cand = pd.read_sql_query("SELECT * FROM features_candidates;", con)

    validate_dataframe(X_cand, set(FEATURES + ["lat", "lon"]), "features_candidates")

    pipe = joblib.load(model_path)
    metadata = _load_metadata(model_path)
    preds = pipe.predict(X_cand[FEATURES])

    scored = X_cand.copy()
    scored["predicted_profit"] = preds
    scored = scored.sort_values("predicted_profit", ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", range(1, len(scored) + 1))

    half_width = _interval_half_width(metadata)
    coverage = float(metadata.get("interval_coverage", 0.80))

    risk_bands: list[str] = []
    profile_distances: list[float] = []
    positive_drivers: list[str] = []
    negative_drivers: list[str] = []

    for _, row in scored.iterrows():
        zscores = _feature_zscores(row, metadata)
        risk, distance = _risk_band(zscores, metadata)
        pos, neg = _candidate_drivers(row, metadata)
        risk_bands.append(risk)
        profile_distances.append(distance)
        positive_drivers.append(pos)
        negative_drivers.append(neg)

    scored["prediction_low_eur"] = scored["predicted_profit"] - half_width
    scored["prediction_high_eur"] = scored["predicted_profit"] + half_width
    scored["interval_half_width_eur"] = half_width
    scored["interval_coverage"] = coverage
    scored["risk_band"] = risk_bands
    scored["profile_distance"] = profile_distances
    scored["main_positive_drivers"] = positive_drivers
    scored["main_negative_drivers"] = negative_drivers

    # Keep high-level decision columns first, then all raw and engineered features.
    leading_cols = [
        "rank",
        "lat",
        "lon",
        "predicted_profit",
        "prediction_low_eur",
        "prediction_high_eur",
        "interval_coverage",
        "risk_band",
        "profile_distance",
        "interval_half_width_eur",
        "main_positive_drivers",
        "main_negative_drivers",
    ]
    remaining_cols = [c for c in scored.columns if c not in leading_cols]
    scored = scored[leading_cols + remaining_cols]

    save_csv(scored, outdir / OUTPUT_FILES.scored_candidates)
    LOGGER.info("Scored %s candidates. Saved to: %s", len(scored), outdir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate coffee shop locations.")
    parser.add_argument("--db", default="coffee.db", help="SQLite DB path.")
    parser.add_argument(
        "--sql",
        default=str(DEFAULT_SQL_PATH),
        help="Path to queries.sql (defaults to packaged SQL).",
    )
    parser.add_argument("--model", default="outputs/model.joblib", help="Path to fitted model.")
    parser.add_argument("--outdir", default="outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_scoring(Path(args.db), Path(args.sql), Path(args.model), Path(args.outdir))


if __name__ == "__main__":
    main()
