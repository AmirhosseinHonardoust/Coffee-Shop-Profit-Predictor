#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from config import COLUMN_RANGES

PathLike = str | Path


def _get_pyplot():
    """Import matplotlib lazily with a headless-safe backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def setup_logging(level: int = logging.INFO) -> None:
    """Configure simple CLI logging."""
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def ensure_outdir(path: PathLike) -> Path:
    """Ensure a directory exists and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: PathLike, *, index: bool = False) -> Path:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index)
    return out


def save_json(obj: dict, path: PathLike) -> Path:
    """Save a JSON-serializable object to disk with indentation."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return out


def load_json(path: PathLike) -> dict:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    """Validate that all required columns exist."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns for {dataset_name}: {sorted(missing)}. "
            "Please check your input CSV schema."
        )


def validate_dataframe(
    df: pd.DataFrame,
    required: set[str],
    dataset_name: str,
    *,
    ranges: dict[str, tuple[float | None, float | None]] | None = None,
) -> None:
    """Validate schema, missing values, numeric values, and simple ranges."""
    validate_columns(df, required, dataset_name)
    ranges = COLUMN_RANGES if ranges is None else ranges

    required_df = df[list(required)].copy()
    missing_counts = required_df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(
            f"Missing values in {dataset_name}: {missing_counts.to_dict()}. "
            "Please clean or impute values before loading."
        )

    for column in required:
        if column in ranges:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Column '{column}' in {dataset_name} must be numeric.")
            min_value, max_value = ranges[column]
            if min_value is not None and (df[column] < min_value).any():
                bad = int((df[column] < min_value).sum())
                raise ValueError(
                    f"Column '{column}' in {dataset_name} has {bad} value(s) below {min_value}."
                )
            if max_value is not None and (df[column] > max_value).any():
                bad = int((df[column] > max_value).sum())
                raise ValueError(
                    f"Column '{column}' in {dataset_name} has {bad} value(s) above {max_value}."
                )


def plot_scatter_actual_vs_pred(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    out_path: PathLike,
    *,
    title: str = "Actual vs Predicted",
) -> Path:
    """Scatter plot of actual vs predicted with y=x reference line."""
    import numpy as np

    plt = _get_pyplot()

    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted)
    ax.set_xlabel("Actual profit (€)")
    ax.set_ylabel("Predicted profit (€)")
    ax.set_title(title)

    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi])

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_hist(series: Iterable[float], out_path: PathLike, *, title: str) -> Path:
    """Histogram for a numeric series."""
    plt = _get_pyplot()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(list(series), bins=20)
    ax.set_title(title)
    ax.set_xlabel("Residual: actual − predicted (€)")
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_bar(
    names: Iterable[str],
    values: Iterable[float],
    out_path: PathLike,
    *,
    title: str = "Feature Importance",
    ylabel: str = "Importance",
) -> Path:
    """Simple bar chart for feature importances or metrics."""
    plt = _get_pyplot()

    names = list(names)
    values = list(values)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, values)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
