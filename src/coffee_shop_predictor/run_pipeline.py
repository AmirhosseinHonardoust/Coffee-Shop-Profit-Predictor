#!/usr/bin/env python3
"""One-command pipeline: build the database, train the model, score candidates.

This chains the three stages in a single process so the whole workflow can be
run with one command (`coffee-pipeline`) instead of three.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coffee_shop_predictor.config import DEFAULT_SQL_PATH, OUTPUT_FILES
from coffee_shop_predictor.create_db import load_csvs_to_db
from coffee_shop_predictor.score_new_sites import run_scoring
from coffee_shop_predictor.train_regression import run_training
from coffee_shop_predictor.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    train_csv: Path,
    candidates_csv: Path,
    db_path: Path,
    sql_path: Path,
    outdir: Path,
) -> None:
    """Run build-db -> train -> score end to end."""
    LOGGER.info("Step 1/3: building database")
    load_csvs_to_db(train_csv, candidates_csv, db_path)

    LOGGER.info("Step 2/3: training model")
    run_training(db_path, sql_path, outdir)

    LOGGER.info("Step 3/3: scoring candidates")
    run_scoring(db_path, sql_path, outdir / OUTPUT_FILES.model, outdir)

    LOGGER.info("Pipeline complete. Artifacts in: %s", outdir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full coffee-shop pipeline end to end.")
    parser.add_argument("--train", default="data/locations_train.csv", help="Training CSV path.")
    parser.add_argument(
        "--candidates", default="data/locations_candidates.csv", help="Candidate CSV path."
    )
    parser.add_argument("--db", default="coffee.db", help="SQLite DB path.")
    parser.add_argument(
        "--sql",
        default=str(DEFAULT_SQL_PATH),
        help="Path to queries.sql (defaults to packaged SQL).",
    )
    parser.add_argument("--outdir", default="outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_pipeline(
        train_csv=Path(args.train),
        candidates_csv=Path(args.candidates),
        db_path=Path(args.db),
        sql_path=Path(args.sql),
        outdir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
