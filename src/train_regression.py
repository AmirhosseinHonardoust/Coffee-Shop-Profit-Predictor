#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    ensure_outdir,
    plot_bar,
    plot_hist,
    plot_scatter_actual_vs_pred,
    save_csv,
    save_json,
)

FEATURES: Final[list[str]] = [
    "foot_traffic", "rent_per_sqm", "competition", "median_income", "office_density",
    "weekend_activity", "events_per_month", "coffee_price", "promo_spend",
    "demand_adj", "wknd_traffic", "price_income", "promo_comp_adj",
]
TARGET: Final[str] = "profit"


def run_training(db_path: Path, sql_path: Path, outdir: Path) -> None:
    """
    Train an ElasticNet regression model on engineered features.
    Saves metrics, diagnostics, and the fitted model.
    """
    outdir = ensure_outdir(outdir)
    charts_dir = ensure_outdir(outdir / "charts")

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_text = f.read()

    with sqlite3.connect(db_path) as con:
        # Create views
        con.executescript(sql_text)
        train_df = pd.read_sql_query("SELECT * FROM features_train;", con)

    X = train_df[FEATURES].copy()
    y = train_df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURES)],
        remainder="drop",
    )

    model = ElasticNet(alpha=0.05, l1_ratio=0.20, random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    pipe.fit(X_train, y_train)
    y_pred_test = pipe.predict(X_test)

    # Metrics
    metrics = {
        "r2": float(r2_score(y_test, y_pred_test)),
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
    }
    save_json(metrics, outdir / "metrics.json")

    # Diagnostics on full training set
    y_pred_all = pipe.predict(X)
    preds = train_df[["lat", "lon"]].copy()
    preds["actual_profit"] = y.values
    preds["predicted_profit"] = y_pred_all
    save_csv(preds, outdir / "predictions_train.csv")

    plot_scatter_actual_vs_pred(
        y_true=y, y_pred=y_pred_all,
        out_path=charts_dir / "actual_vs_predicted.png",
        title="Actual vs Predicted Profit",
    )
    residuals = y - y_pred_all
    plot_hist(residuals, charts_dir / "residuals_hist.png", title="Residuals (Actual − Predicted)")

    # Feature importance (std. coefficients after scaling)
    coefs = pipe.named_steps["model"].coef_
    importance = pd.DataFrame({"feature": FEATURES, "importance": coefs})
    importance = importance.reindex(importance["importance"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    save_csv(importance, outdir / "feature_importance.csv")
    plot_bar(
        names=importance["feature"][:15],
        values=importance["importance"][:15],
        out_path=charts_dir / "feature_importance.png",
        title="Feature Importance (Std. Coefficients)",
    )

    # Persist model
    joblib.dump(pipe, outdir / "model.joblib")
    print(f"Artifacts saved to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train coffee shop profit regression model.")
    parser.add_argument("--db", default="coffee.db", help="Path to SQLite DB.")
    parser.add_argument("--sql", default="src/queries.sql", help="Path to queries.sql.")
    parser.add_argument("--outdir", default="outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(Path(args.db), Path(args.sql), Path(args.outdir))


if __name__ == "__main__":
    main()
