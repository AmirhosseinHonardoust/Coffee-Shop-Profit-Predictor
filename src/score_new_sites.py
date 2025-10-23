#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Final

import joblib
import pandas as pd

from utils import ensure_outdir, save_csv

FEATURES: Final[list[str]] = [
    "foot_traffic", "rent_per_sqm", "competition", "median_income", "office_density",
    "weekend_activity", "events_per_month", "coffee_price", "promo_spend",
    "demand_adj", "wknd_traffic", "price_income", "promo_comp_adj",
]


def run_scoring(db_path: Path, sql_path: Path, model_path: Path, outdir: Path) -> None:
    """
    Score candidate locations using a trained pipeline.
    """
    outdir = ensure_outdir(outdir)

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_text = f.read()

    with sqlite3.connect(db_path) as con:
        con.executescript(sql_text)
        X_cand = pd.read_sql_query("SELECT * FROM features_candidates;", con)

    pipe = joblib.load(model_path)
    preds = pipe.predict(X_cand[FEATURES])

    scored = X_cand[["lat", "lon"]].copy()
    scored["predicted_profit"] = preds
    save_csv(scored, outdir / "scored_candidates.csv")

    print(f"Scored {len(scored)} candidates. Saved to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate coffee shop locations.")
    parser.add_argument("--db", default="coffee.db", help="SQLite DB path.")
    parser.add_argument("--sql", default="src/queries.sql", help="Path to queries.sql.")
    parser.add_argument("--model", default="outputs/model.joblib", help="Path to fitted model.")
    parser.add_argument("--outdir", default="outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_scoring(Path(args.db), Path(args.sql), Path(args.model), Path(args.outdir))


if __name__ == "__main__":
    main()
