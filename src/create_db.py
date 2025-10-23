#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Final

import pandas as pd

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

SCHEMA_CAND: Final[str] = """
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

REQUIRED_TRAIN = {
    "lat", "lon", "foot_traffic", "rent_per_sqm", "competition",
    "median_income", "office_density", "weekend_activity",
    "events_per_month", "coffee_price", "promo_spend", "profit",
}

REQUIRED_CAND = REQUIRED_TRAIN - {"profit"}


def _validate_columns(df: pd.DataFrame, required: set[str], table: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns for {table}: {sorted(missing)}. "
            "Please check your input CSV schema."
        )


def _load_csv_to_db(csv_path: Path, table: str, con: sqlite3.Connection) -> None:
    df = pd.read_csv(csv_path)
    required = REQUIRED_TRAIN if table == "locations_train" else REQUIRED_CAND
    _validate_columns(df, required, table)
    df.to_sql(table, con, if_exists="replace", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load location CSVs into SQLite.")
    parser.add_argument("--train", required=True, help="Path to locations_train.csv")
    parser.add_argument("--candidates", required=True, help="Path to locations_candidates.csv")
    parser.add_argument("--db", default="coffee.db", help="SQLite DB path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db = Path(args.db)

    with sqlite3.connect(db) as con:
        con.execute(SCHEMA_TRAIN)
        con.execute(SCHEMA_CAND)
        _load_csv_to_db(Path(args.train), "locations_train", con)
        _load_csv_to_db(Path(args.candidates), "locations_candidates", con)

    print(f"Loaded CSVs into {db}: tables [locations_train, locations_candidates]")


if __name__ == "__main__":
    main()
