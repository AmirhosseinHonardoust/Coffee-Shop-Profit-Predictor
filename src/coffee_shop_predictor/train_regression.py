#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from coffee_shop_predictor.config import (
    CV_SPLITS,
    DEFAULT_SQL_PATH,
    FEATURES,
    OUTPUT_FILES,
    PREDICTION_INTERVAL_COVERAGE,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from coffee_shop_predictor.utils import (
    ensure_outdir,
    plot_bar,
    plot_hist,
    plot_scatter_actual_vs_pred,
    save_csv,
    save_json,
    setup_logging,
    validate_dataframe,
)

LOGGER = logging.getLogger(__name__)


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURES)],
        remainder="drop",
    )


def _pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(steps=[("pre", _preprocessor()), ("model", model)])


def _model_candidates() -> dict[str, tuple[Pipeline, dict[str, list[Any]] | None]]:
    """Return lightweight model candidates and tuning grids."""
    return {
        "MeanBaseline": (
            Pipeline(steps=[("model", DummyRegressor(strategy="mean"))]),
            None,
        ),
        "LinearRegression": (_pipeline(LinearRegression()), None),
        "Ridge": (
            _pipeline(Ridge(random_state=RANDOM_STATE)),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "ElasticNet": (
            _pipeline(ElasticNet(max_iter=20_000, random_state=RANDOM_STATE)),
            {
                "model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                "model__l1_ratio": [0.1, 0.2, 0.5, 0.8],
            },
        ),
        "RandomForest": (
            _pipeline(
                RandomForestRegressor(
                    n_estimators=50,
                    max_depth=4,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            None,
        ),
        "GradientBoosting": (
            _pipeline(
                GradientBoostingRegressor(
                    n_estimators=80,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=RANDOM_STATE,
                )
            ),
            None,
        ),
    }


def _evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: KFold,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Tune/evaluate model candidates and return comparison table plus fitted estimators."""
    rows: list[dict[str, Any]] = []
    fitted_estimators: dict[str, Pipeline] = {}

    for name, (estimator, grid) in _model_candidates().items():
        LOGGER.info("Evaluating %s", name)
        if grid:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                cv=cv,
                scoring="neg_mean_absolute_error",
                refit=True,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = search.best_params_
            cv_mae_mean = float(-search.best_score_)

            cv_scores = cross_validate(
                fitted,
                X_train,
                y_train,
                cv=cv,
                scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
                n_jobs=1,
                error_score="raise",
            )
            cv_r2_mean = float(np.mean(cv_scores["test_r2"]))
            cv_r2_std = float(np.std(cv_scores["test_r2"]))
            cv_mae_std = float(np.std(-cv_scores["test_neg_mae"]))
        else:
            cv_scores = cross_validate(
                estimator,
                X_train,
                y_train,
                cv=cv,
                scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
                n_jobs=1,
                error_score="raise",
            )
            fitted = clone(estimator).fit(X_train, y_train)
            best_params = {}
            cv_r2_mean = float(np.mean(cv_scores["test_r2"]))
            cv_r2_std = float(np.std(cv_scores["test_r2"]))
            cv_mae_mean = float(np.mean(-cv_scores["test_neg_mae"]))
            cv_mae_std = float(np.std(-cv_scores["test_neg_mae"]))

        y_pred_test = fitted.predict(X_test)
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        test_r2 = float(r2_score(y_test, y_pred_test))

        fitted_estimators[name] = fitted
        rows.append(
            {
                "model": name,
                "cv_mae_mean": cv_mae_mean,
                "cv_mae_std": cv_mae_std,
                "cv_r2_mean": cv_r2_mean,
                "cv_r2_std": cv_r2_std,
                "holdout_mae": test_mae,
                "holdout_r2": test_r2,
                "best_params": best_params,
            }
        )

    comparison = (
        pd.DataFrame(rows).sort_values("cv_mae_mean", ascending=True).reset_index(drop=True)
    )
    return comparison, fitted_estimators


def _selected_model_cv(
    estimator: Pipeline, X: pd.DataFrame, y: pd.Series, cv: KFold
) -> dict[str, float]:
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
        n_jobs=1,
        error_score="raise",
    )
    mae = -scores["test_neg_mae"]
    r2 = scores["test_r2"]
    return {
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
        "mae_mean": float(np.mean(mae)),
        "mae_std": float(np.std(mae)),
    }


def _feature_importance(
    estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Estimate feature importance for the selected model.

    Linear models use standardized coefficients for speed and interpretability.
    Non-linear models fall back to permutation importance on the holdout set.
    """
    model = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else None
    if model is not None and hasattr(model, "coef_"):
        signed_effect = np.asarray(model.coef_, dtype=float)
        importance = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": np.abs(signed_effect),
                "signed_effect": signed_effect,
                "importance_type": "standardized_coefficient",
            }
        )
        return importance.sort_values("importance", ascending=False).reset_index(drop=True)

    result = permutation_importance(
        estimator,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
            "signed_effect": np.nan,
            "importance_type": "permutation_mae_increase_holdout",
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def _conformal_half_width(residuals: pd.Series, coverage: float) -> tuple[float, float]:
    """Split-conformal symmetric prediction-interval half-width from holdout residuals.

    Returns the additive half-width ``q`` such that ``[pred - q, pred + q]`` covers
    a new target with approximately ``coverage`` probability under exchangeability,
    along with the empirical coverage of that half-width on the holdout residuals.
    """
    abs_residuals = np.abs(residuals.to_numpy(dtype=float))
    n = abs_residuals.size
    # Finite-sample conformal quantile level: ceil((n + 1) * coverage) / n, capped at 1.
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    half_width = float(np.quantile(abs_residuals, level))
    empirical = float(np.mean(abs_residuals <= half_width))
    return half_width, empirical


def _make_metadata(
    selected_model: str,
    X: pd.DataFrame,
    holdout_mae: float,
    cv_mae_mean: float,
    cv_mae_std: float,
    interval_coverage: float,
    interval_half_width: float,
) -> dict[str, Any]:
    means = X[FEATURES].mean()
    stds = X[FEATURES].std(ddof=0).replace(0, 1)
    average_abs_z = ((X[FEATURES] - means) / stds).abs().mean(axis=1)
    return {
        "selected_model": selected_model,
        "features": FEATURES,
        "feature_means": {k: float(v) for k, v in means.items()},
        "feature_stds": {k: float(v) for k, v in stds.items()},
        "risk_distance_q75": float(average_abs_z.quantile(0.75)),
        "risk_distance_q95": float(average_abs_z.quantile(0.95)),
        "expected_mae_eur": float(max(holdout_mae, cv_mae_mean)),
        "interval_coverage": float(interval_coverage),
        "interval_half_width_eur": float(interval_half_width),
        "cv_mae_mean": float(cv_mae_mean),
        "cv_mae_std": float(cv_mae_std),
    }


def run_training(db_path: Path, sql_path: Path, outdir: Path) -> None:
    """Train, compare, evaluate, and persist a regression model."""
    outdir = ensure_outdir(outdir)
    charts_dir = ensure_outdir(outdir / "charts")

    sql_text = sql_path.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as con:
        con.executescript(sql_text)
        train_df = pd.read_sql_query("SELECT * FROM features_train;", con)

    validate_dataframe(train_df, set(FEATURES + [TARGET, "lat", "lon"]), "features_train")

    X = train_df[FEATURES].copy()
    y = train_df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    comparison, fitted_estimators = _evaluate_candidates(X_train, y_train, X_test, y_test, cv)
    save_csv(comparison, outdir / OUTPUT_FILES.model_comparison)
    LOGGER.info("Saved model comparison table")

    plot_bar(
        names=comparison["model"],
        values=comparison["cv_mae_mean"],
        out_path=charts_dir / "model_comparison_mae.png",
        title="Model Comparison (5-fold CV MAE on Training Split)",
        ylabel="MAE (€), lower is better",
    )
    LOGGER.info("Saved model comparison chart")

    non_baseline = comparison[comparison["model"] != "MeanBaseline"]
    selected_model = str(
        non_baseline.iloc[0]["model"] if not non_baseline.empty else comparison.iloc[0]["model"]
    )
    holdout_estimator = fitted_estimators[selected_model]
    baseline_row = comparison[comparison["model"] == "MeanBaseline"].iloc[0]

    y_pred_test = holdout_estimator.predict(X_test)
    residuals = y_test - y_pred_test

    predictions_test = train_df.loc[X_test.index, ["lat", "lon"]].copy()
    predictions_test["actual_profit"] = y_test.values
    predictions_test["predicted_profit"] = y_pred_test
    predictions_test["residual"] = residuals.values
    save_csv(predictions_test, outdir / OUTPUT_FILES.predictions_test)
    LOGGER.info("Saved holdout predictions")

    plot_scatter_actual_vs_pred(
        y_true=y_test,
        y_pred=y_pred_test,
        out_path=charts_dir / "actual_vs_predicted.png",
        title="Actual vs Predicted Profit (Holdout Test Set)",
    )
    plot_hist(residuals, charts_dir / "residuals_hist.png", title="Residuals on Holdout Test Set")
    LOGGER.info("Saved diagnostic plots")

    importance = _feature_importance(holdout_estimator, X_test, y_test)
    save_csv(importance, outdir / OUTPUT_FILES.feature_importance)
    LOGGER.info("Saved feature importance table")
    plot_bar(
        names=importance["feature"][:15],
        values=importance["importance"][:15],
        out_path=charts_dir / "feature_importance.png",
        title="Feature Importance for Selected Model",
        ylabel="Importance",
    )
    LOGGER.info("Saved feature importance chart")

    LOGGER.info("Starting selected model CV")
    selected_cv = _selected_model_cv(clone(holdout_estimator), X, y, cv)
    LOGGER.info("Finished selected model CV")
    holdout_mae = float(mean_absolute_error(y_test, y_pred_test))
    holdout_r2 = float(r2_score(y_test, y_pred_test))

    interval_half_width, interval_empirical_coverage = _conformal_half_width(
        residuals, PREDICTION_INTERVAL_COVERAGE
    )

    metrics = {
        "selected_model": selected_model,
        "selection_rule": "lowest mean CV MAE among non-baseline models on the training split",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "rows": {
            "total": int(len(train_df)),
            "train": int(len(X_train)),
            "holdout_test": int(len(X_test)),
        },
        "holdout": {
            "r2": holdout_r2,
            "mae": holdout_mae,
            "baseline_r2": float(baseline_row["holdout_r2"]),
            "baseline_mae": float(baseline_row["holdout_mae"]),
            "prediction_interval": {
                "method": "split_conformal_absolute_residual",
                "target_coverage": PREDICTION_INTERVAL_COVERAGE,
                "half_width_eur": interval_half_width,
                "empirical_coverage_holdout": interval_empirical_coverage,
            },
        },
        "cross_validation_selected_model_full_data": {
            "cv_splits": CV_SPLITS,
            **selected_cv,
        },
        "note": (
            "Candidate scoring uses the selected model refit on all training rows after "
            "holdout evaluation. Diagnostic plots use only holdout predictions."
        ),
    }
    save_json(metrics, outdir / OUTPUT_FILES.metrics)
    LOGGER.info("Saved metrics")

    # Refit the selected model on all available training data for candidate scoring.
    LOGGER.info("Refitting selected model on all data")
    final_model = clone(holdout_estimator).fit(X, y)
    LOGGER.info("Saving model")
    joblib.dump(final_model, outdir / OUTPUT_FILES.model)
    LOGGER.info("Saved model")

    predictions_all = train_df[["lat", "lon"]].copy()
    predictions_all["actual_profit"] = y.values
    predictions_all["predicted_profit"] = final_model.predict(X)
    predictions_all["residual"] = (
        predictions_all["actual_profit"] - predictions_all["predicted_profit"]
    )
    save_csv(predictions_all, outdir / OUTPUT_FILES.predictions_all)
    LOGGER.info("Saved all-row predictions")

    metadata = _make_metadata(
        selected_model=selected_model,
        X=X,
        holdout_mae=holdout_mae,
        cv_mae_mean=selected_cv["mae_mean"],
        cv_mae_std=selected_cv["mae_std"],
        interval_coverage=PREDICTION_INTERVAL_COVERAGE,
        interval_half_width=interval_half_width,
    )
    save_json(metadata, outdir / OUTPUT_FILES.metadata)
    LOGGER.info("Saved metadata")

    LOGGER.info("Selected model: %s", selected_model)
    LOGGER.info("Holdout MAE: %.2f | Holdout R2: %.3f", holdout_mae, holdout_r2)
    LOGGER.info("Artifacts saved to: %s", outdir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train coffee shop profit regression model.")
    parser.add_argument("--db", default="coffee.db", help="Path to SQLite DB.")
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
    run_training(Path(args.db), Path(args.sql), Path(args.outdir))


if __name__ == "__main__":
    main()
