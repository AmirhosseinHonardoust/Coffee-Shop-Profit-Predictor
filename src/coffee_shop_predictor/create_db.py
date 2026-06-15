#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Final

import pandas as pd

from coffee_shop_predictor.config import REQUIRED_CANDIDATE_COLUMNS, REQUIRED_TRAIN_COLUMNS
from coffee_shop_predictor.utils import setup_logging, validate_dataframe

LOGGER = logging.getLogger(__name__)

SCHEMA_TRAIN: Final[str] = """
CREATE TABLE IF NOT EXISTS locations_train (
    lat REAL,
    lon REAL,
    foot_traffic INTEGER,
    rent_per_sqm REAL,
    competition INTEGER,
    median_income REAL,
    office_density INTEGER,
    weekend_activity REAL,
    events_per_month INTEGER,
    coffee_price REAL,
    promo_spend REAL,
    profit REAL
);
"""

SCHEMA_CANDIDATES: Final[str] = """
CREATE TABLE IF NOT EXISTS locations_candidates (
    lat REAL,
    lon REAL,
    foot_traffic INTEGER,
    rent_per_sqm REAL,
    competition INTEGER,
    median_income REAL,
    office_density INTEGER,
    weekend_activity REAL,
    events_per_month INTEGER,
    coffee_price REAL,
    promo_spend REAL
);
"""


def _load_csv_to_db(csv_path: Path, table: str, con: sqlite3.Connection) -> None:
    df = pd.read_csv(csv_path)
    required = REQUIRED_TRAIN_COLUMNS if table == "locations_train" else REQUIRED_CANDIDATE_COLUMNS
    validate_dataframe(df, required, table)
    df.to_sql(table, con, if_exists="replace", index=False)
    LOGGER.info("Loaded %s rows into %s", len(df), table)


def load_csvs_to_db(train_csv: Path, candidates_csv: Path, db_path: Path) -> None:
    """Load and validate source CSV files into SQLite tables."""
    with sqlite3.connect(db_path) as con:
        con.execute(SCHEMA_TRAIN)
        con.execute(SCHEMA_CANDIDATES)
        _load_csv_to_db(train_csv, "locations_train", con)
        _load_csv_to_db(candidates_csv, "locations_candidates", con)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load location CSVs into SQLite.")
    parser.add_argument("--train", required=True, help="Path to locations_train.csv")
    parser.add_argument("--candidates", required=True, help="Path to locations_candidates.csv")
    parser.add_argument("--db", default="coffee.db", help="SQLite DB path")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    db = Path(args.db)
    load_csvs_to_db(Path(args.train), Path(args.candidates), db)
    LOGGER.info("Loaded CSVs into %s: tables [locations_train, locations_candidates]", db)


if __name__ == "__main__":
    main()
