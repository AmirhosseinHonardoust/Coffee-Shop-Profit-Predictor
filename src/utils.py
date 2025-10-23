#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import matplotlib.pyplot as plt
import pandas as pd

PathLike = Union[str, Path]


def ensure_outdir(path: PathLike) -> Path:
    """
    Ensure a directory exists and return it as a Path.

    Parameters
    ----------
    path : str | Path
        Folder path to create if missing.

    Returns
    -------
    Path
        Created/existing directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: PathLike, *, index: bool = False) -> Path:
    """
    Save a DataFrame to CSV, creating parent directories as needed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index)
    return out


def save_json(obj: dict, path: PathLike) -> Path:
    """
    Save a JSON-serializable object to disk with indentation.
    """
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return out


def plot_scatter_actual_vs_pred(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    out_path: PathLike,
    *,
    title: str = "Actual vs Predicted",
) -> Path:
    """
    Scatter plot of actual vs predicted with y=x reference line.
    """
    import numpy as np

    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi])

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_hist(series: Iterable[float], out_path: PathLike, *, title: str) -> Path:
    """
    Histogram for a numeric series.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(list(series), bins=30)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_bar(
    names: Iterable[str],
    values: Iterable[float],
    out_path: PathLike,
    *,
    title: str = "Feature Importance",
) -> Path:
    """
    Simple bar chart for feature importances or metrics.
    """
    names = list(names)
    values = list(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, values)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
