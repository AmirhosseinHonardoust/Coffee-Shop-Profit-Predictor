<div align="center">

# Coffee Shop Profit Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![SQLite](https://img.shields.io/badge/SQLite-Feature%20Engineering-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-orange)
![Code Quality](https://img.shields.io/badge/Code%20Quality-Ruff%20%7C%20Black%20%7C%20mypy-green)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Coffee-Shop-Profit-Predictor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Coffee-Shop-Profit-Predictor/actions/workflows/ci.yml)

</div>

A machine learning project that turns coffee-shop location data into a **monthly-profit estimate**, using **SQL feature engineering**, a **scikit-learn regression workflow**, **honest holdout evaluation**, **multi-model comparison**, **business-readable candidate ranking with risk bands**, and an **installable command-line package**.

> **Important:** This project is a **site-selection decision-support demo**, not a production financial forecasting system.
>
> It estimates likely monthly profit from the available location features and should be read alongside the limitations below. It does not guarantee real-world profitability, does not include costs such as labor or goods sold, and should never be the sole basis for a lease or investment decision.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Generating Data](#generating-data)
- [Building the Database](#building-the-database)
- [Training the Model](#training-the-model)
- [Scoring Candidate Sites](#scoring-candidate-sites)
- [SQL Feature Engineering](#sql-feature-engineering)
- [Model Output](#model-output)
- [Prediction Intervals](#prediction-intervals)
- [Model Artifacts and Loading Safety](#model-artifacts-and-loading-safety)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Data Statement](#data-statement)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Coffee-shop site selection is often presented as if a model can simply pick a winning location. In reality, a regression model cannot guarantee future profit; it can only estimate likely profit from the features it was given. A profit estimate is only useful if it can support a defensible action:

- rank candidate locations by predicted monthly profit
- flag candidates whose feature profile is far from the training data
- explain which features push a prediction up or down
- communicate uncertainty and limitations clearly

This project demonstrates an end-to-end, honest regression workflow on a labeled set of location features. It loads data into SQLite, builds reusable SQL feature views, compares multiple regression models with cross-validation, evaluates the selected model on a holdout set, and scores new candidate sites with risk bands and business-readable drivers.

The goal is to show how a regression model can be turned into a **responsible decision-support tool**, not just a single R² or MAE score.

---

## What This Project Does

This project can:

- Load training and candidate-location data from CSV files
- Store the data in a SQLite database
- Engineer reusable features through SQL views
- Compare multiple regression models with cross-validation
- Compare every model against a mean baseline
- Tune hyperparameters for Ridge and ElasticNet
- Evaluate the selected model on a separate holdout set
- Save trained model artifacts with metadata
- Score new candidate locations and rank them by predicted profit
- Add risk bands and positive/negative driver explanations
- Generate diagnostic charts
- Run automated tests and CI smoke workflows

---

## What This Project Does Not Do

This project does **not**:

- Guarantee actual future profit
- Replace a full real-estate or finance analysis
- Include labor cost, cost of goods sold, store size, opening hours, or seasonality
- Use live market data
- Validate predictions against real store openings
- Provide a formal statistical prediction interval
- Prove that a feature causally increases or decreases profit

A production site-selection system would need real transaction data, lease terms, store size, local regulations, labor costs, competitor quality, seasonality, and geographic validation.

---

## Key Features

- **SQLite feature store** built from CSV inputs
- **SQL feature engineering** through reusable views, so training and candidate sites are transformed identically (no train/serve skew)
- **Centralized configuration** in `config.py` for features, ranges, labels, and output filenames
- **Input validation** for schema, missing values, numeric types, and impossible ranges
- **Multi-model comparison** across Linear Regression, Ridge, ElasticNet, Random Forest, and Gradient Boosting
- **Mean-baseline comparison** so improvements are read honestly
- **Cross-validation** and **hyperparameter tuning** for Ridge and ElasticNet
- **Holdout test-set evaluation**, with all diagnostic plots drawn from unseen data only
- **Candidate ranking** with **heuristic risk bands** and **business-readable positive/negative drivers**
- **Saved model artifacts** via joblib, plus a metadata sidecar describing the training feature profile
- **Installable package** exposing `coffee-build-db`, `coffee-train`, and `coffee-score` commands
- **Ruff + Black + mypy** quality gates with **pre-commit** hooks
- **Unit tests and GitHub Actions CI** across Python 3.10–3.12
- **Honest documentation** of data and model limitations

---

## System Workflow

```text
CSV location data
        ↓
SQLite database
        ↓
SQL feature views (shared by train and candidates)
        ↓
Input validation
        ↓
Train/holdout split
        ↓
Model comparison + cross-validation
        ↓
Selected regression pipeline
        ↓
Holdout evaluation
        ↓
Candidate-site scoring
        ↓
Ranked recommendations with risk bands and drivers
```

---

## Project Structure

```text
Coffee-Shop-Profit-Predictor/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── locations_train.csv
│   └── locations_candidates.csv
│
├── outputs/
│   ├── charts/
│   │   ├── actual_vs_predicted.png
│   │   ├── feature_importance.png
│   │   ├── model_comparison_mae.png
│   │   └── residuals_hist.png
│   ├── feature_importance.csv
│   ├── metrics.json
│   ├── model.joblib
│   ├── model_comparison.csv
│   ├── model_metadata.json
│   ├── predictions_all.csv
│   ├── predictions_test.csv
│   └── scored_candidates.csv
│
├── src/
│   └── coffee_shop_predictor/
│       ├── __init__.py
│       ├── config.py
│       ├── create_db.py
│       ├── generate_data.py
│       ├── queries.sql
│       ├── run_pipeline.py
│       ├── score_new_sites.py
│       ├── train_regression.py
│       └── utils.py
│
├── tests/
│   ├── test_generate_data.py
│   ├── test_pipeline.py
│   ├── test_scoring.py
│   ├── test_training.py
│   └── test_workflow.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

> The `outputs/` directory is regenerated by training and is not tracked in version control. CI rebuilds it on every run and uploads it as a build artifact.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Coffee-Shop-Profit-Predictor.git
cd Coffee-Shop-Profit-Predictor
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the Package

Install in editable mode. This pulls in all dependencies and registers the `coffee-generate-data`, `coffee-build-db`, `coffee-train`, `coffee-score`, and `coffee-pipeline` commands:

```bash
pip install -e .
```

For development tools (Ruff, Black, mypy, pre-commit):

```bash
pip install -e ".[dev]"
```

If you have `make`, `make dev` does the same and installs the pre-commit hooks.

---

## Quick Start

Run the entire workflow (build database, train, score) with one command:

```bash
coffee-pipeline
```

Or with `make`:

```bash
make pipeline
```

To run the stages individually instead:

Build the database:

```bash
coffee-build-db --train data/locations_train.csv --candidates data/locations_candidates.csv
```

Train and evaluate the model:

```bash
coffee-train
```

Score candidate sites:

```bash
coffee-score
```

---

## Generating Data

The bundled CSVs are reproducible through a seeded generator. It samples each
feature from a documented distribution and synthesizes `profit` from a
transparent linear formula plus Gaussian noise (full specification in the
`generate_data.py` module docstring).

```bash
coffee-generate-data --n-train 220 --n-candidates 60 --seed 42 --out-dir data/generated
```

- The same `--seed` always produces the same data.
- Calibrated to the reference sample: a model trained on generated data reaches a
  comparable cross-validated R² (about 0.53-0.56).
- Writing to `--out-dir data` overwrites the bundled CSVs, so the example points
  at a separate directory.

> The signs in the profit formula match the documented business priors: rent and
> competition reduce profit, every other feature increases it.

---

## Building the Database

```bash
coffee-build-db --train data/locations_train.csv --candidates data/locations_candidates.csv --db coffee.db
```

This will:

- Read the training and candidate CSV files
- Validate required columns and numeric ranges
- Create a SQLite database
- Save the data into database tables

Created database:

```text
coffee.db
```

---

## Training the Model

```bash
coffee-train --db coffee.db --outdir outputs
```

This will:

- Load SQL feature views
- Split the training data into train and holdout sets
- Compare multiple regression models with cross-validation
- Select the best non-baseline model by cross-validated MAE
- Evaluate the selected model on the holdout test set
- Refit the selected model on all data for candidate scoring
- Save predictions, metrics, comparison results, and charts

Generated outputs include:

```text
outputs/model.joblib
outputs/metrics.json
outputs/model_metadata.json
outputs/model_comparison.csv
outputs/feature_importance.csv
outputs/predictions_test.csv
outputs/predictions_all.csv
outputs/charts/
```

> The packaged SQL is located relative to the installed code, so `--sql` is optional. Pass it only to override the bundled `queries.sql`.

---

## Scoring Candidate Sites

```bash
coffee-score --db coffee.db --model outputs/model.joblib --outdir outputs
```

This will:

- Load candidate-site features from SQLite
- Apply the trained model
- Predict monthly profit and rank candidates
- Add risk bands and business-readable drivers
- Save the scored candidate file

Generated output:

```text
outputs/scored_candidates.csv
```

---

## SQL Feature Engineering

Feature creation lives in `queries.sql`, which builds two SQLite views so the same transformations are applied to training locations and candidate sites:

```text
features_train
features_candidates
```

Engineered features:

<div align="center">

| Feature | Formula | Purpose |
|---|---|---|
| `demand_adj` | `foot_traffic / (1 + competition)` | Demand adjusted for competitive intensity |
| `wknd_traffic` | `weekend_activity × foot_traffic` | Weekend demand potential |
| `price_income` | `coffee_price × (median_income / 1000)` | Price-income fit / affordability signal |
| `promo_comp_adj` | `promo_spend / (1 + competition)` | Promotion adjusted for competition |

</div>

> Keeping feature logic in SQL means a candidate site is never transformed differently from the training data it is compared against.

---

## Model Output

Candidate scoring returns ranked results with the following fields:

<div align="center">

| Field | Meaning |
|---|---|
| `rank` | Candidate rank by predicted monthly profit |
| `lat`, `lon` | Candidate coordinates |
| `predicted_profit` | Point estimate of monthly profit in euros |
| `prediction_low_eur`, `prediction_high_eur` | Lower/upper bound of the conformal prediction interval |
| `interval_coverage` | Target coverage of the interval (e.g. 0.80) |
| `interval_half_width_eur` | Half-width of the interval (the `±` value) |
| `risk_band` | Extrapolation-risk level from feature-profile distance |
| `profile_distance` | Distance from the training-data feature profile |
| `main_positive_drivers` | Business-readable features helping the score |
| `main_negative_drivers` | Business-readable features hurting the score |

</div>

Example ranked result:

```text
Rank: 1
Predicted profit: €2,004.10
80% interval: €1,415.21 - €2,592.99  (half-width ±€588.89)
Risk band: high  (profile distance 1.32)
Main positive drivers: high competition-adjusted promotion; high competition-adjusted demand; high coffee price
Main negative drivers: no major red flags
```

> Two complementary signals: the prediction interval quantifies uncertainty around the point estimate, while the risk band flags how far the candidate sits from the training profile (extrapolation risk). A candidate can have a tight interval but still be high-risk if its feature profile is unusual.

---

## Prediction Intervals

Each prediction comes with an interval, not just a point estimate. The interval
is built with **split-conformal prediction** on the holdout residuals:

- compute the absolute residuals on the holdout set
- take the quantile of those residuals at the target coverage level
- report each prediction as `predicted_profit ± half_width`

This is distribution-free and gives approximate marginal coverage under
exchangeability: roughly `interval_coverage` of true values are expected to fall
inside the interval. The empirical coverage on the holdout set is recorded in
`metrics.json` as a sanity check.

The default target coverage is 80% (`PREDICTION_INTERVAL_COVERAGE` in
`config.py`). The interval width is the same for every candidate, which is the
standard behaviour of basic split conformal; per-candidate extrapolation risk is
communicated separately through the risk band.

---

## Model Artifacts and Loading Safety

The trained model is stored as a joblib (pickle) file, alongside a metadata sidecar describing the training feature profile used for risk bands:

```text
outputs/model.joblib
outputs/model_metadata.json
```

> **Deserializing a pickle executes arbitrary code**, so only load model files you produced yourself or fully trust — never one downloaded from an untrusted source. Candidate scoring reads `model_metadata.json` to compute feature-profile distance; if it is missing, scoring falls back to conservative default thresholds.

---

## Evaluation Metrics

Evaluation uses a train/holdout split with cross-validation on the training split, plus a mean-baseline comparison so improvements are read honestly.

<div align="center">

| Metric | Why it matters |
|---|---|
| Holdout R² | Variance explained on unseen data |
| Holdout MAE | Average euro error on unseen data |
| Mean-baseline R² / MAE | Honest floor: what predicting the average achieves |
| 5-fold CV R² / MAE | Stability of the estimate across folds |
| Interval coverage | Empirical share of holdout actuals inside the 80% interval |

</div>

The current run selects **ElasticNet** (`alpha = 0.5`, `l1_ratio = 0.8`):

<div align="center">

| Metric | Value |
|---|---|
| Holdout R² | 0.647 |
| Holdout MAE | €383.58 |
| Mean baseline R² | -0.030 |
| Mean baseline MAE | €699.39 |
| 5-fold CV R² | 0.530 ± 0.110 |
| 5-fold CV MAE | €413.67 ± €46.65 |
| 80% interval half-width | ±€588.89 |
| Empirical interval coverage (holdout) | 81.8% |

</div>

> The model improves clearly over the mean baseline, but the cross-validation spread shows results should be read carefully because the dataset is small.

---

## Visual Reports

### Model diagnostics

<div align="center">

| Actual vs Predicted | Residuals Histogram |
|---|---|
| ![Actual vs predicted](https://github.com/user-attachments/assets/1975709e-240c-41b6-b539-f4c61177df82) | ![Residuals histogram](https://github.com/user-attachments/assets/69b37b0f-4c5e-4640-9463-2145121fb2c8) |
| **Analysis:** Holdout-only predictions plotted against actual profit, with a y=x reference line. Using unseen data makes this diagnostic more honest than plotting on training data. | **Analysis:** The distribution of holdout residuals (actual − predicted). A narrower, centered spread means lower and less biased prediction error. |

</div>

### Feature signal and model selection

<div align="center">

| Feature Importance | Model Comparison (MAE) |
|---|---|
| ![Feature importance](https://github.com/user-attachments/assets/7d06bcca-69be-451f-9fcd-e13a22ed5df8) | ![Model comparison MAE](https://github.com/user-attachments/assets/d0d9131d-876f-4a68-9477-9e99d801288e) |
| **Analysis:** Standardized coefficients for the selected ElasticNet model. These are model associations, not causal proof. | **Analysis:** Cross-validated MAE per model on the training split. This shows whether the selected model meaningfully beats simpler alternatives and the mean baseline. |

</div>

---

## Testing and CI

The test suite covers both the workflow and the business logic that candidate
scoring exposes:

- source files compile and the CSV-to-SQLite load builds the expected views
- input validation rejects impossible values
- the data generator is deterministic for a fixed seed and passes validation
- the data generator is deterministic for a fixed seed and passes validation
- feature z-scores honor the metadata mean/std (with a zero-std fallback)
- risk bands map distances to low / medium / high and apply the error multipliers
- driver explanations surface high competition and low rent correctly, cap at
  `top_n`, and fall back to "balanced profile" / "no major red flags"
- a training smoke run writes every artifact and beats the mean baseline
- an end-to-end pipeline run produces ranked candidates with prediction intervals

Run unit tests locally:

```bash
python -m unittest discover -s tests -v
```

Lint, format, and type check:

```bash
ruff check src tests
black --check src tests
mypy src
```

The GitHub Actions workflow checks:

- package installation in editable mode
- linting with Ruff
- format checking with Black
- type checking with mypy
- unit tests
- a full training and scoring smoke workflow
- training artifact and output validation
- artifact upload across Python 3.10, 3.11, and 3.12

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

The project separates responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `config.py` | Central settings: features, ranges, labels, and output filenames |
| `utils.py` | Validation, JSON/CSV IO, and headless-safe plotting helpers |
| `create_db.py` | Loads and validates CSVs into SQLite tables |
| `train_regression.py` | Model comparison, cross-validation, holdout evaluation, and artifacts |
| `score_new_sites.py` | Candidate scoring, risk bands, and driver explanations |
| `queries.sql` | SQL feature views shared by training and candidate sites |

</div>

Tooling is configured through `pyproject.toml` (Ruff, Black, mypy) with pre-commit hooks in `.pre-commit-config.yaml`.

---

## Data Statement

This repository is presented as a **demonstration / simulated retail analytics project**.

The bundled dataset contains:

```text
220 training locations
60 candidate locations
```

The bundled CSVs are reproducible through `coffee-generate-data`, which samples each feature from a documented distribution and synthesizes `profit` from a transparent linear formula plus noise. This keeps the dataset parameterizable (size and seed) and makes the generative process auditable rather than opaque.

The data is suitable for demonstrating a portfolio machine learning workflow, but it should not be treated as verified real business data. If the project is later connected to real data, this section should be updated with the data source, collection period, geographic coverage, feature definitions, usage permissions, and known data quality issues.

---

## Limitations

This project has important limitations:

- The dataset is small and simulated, not a production corpus
- The model does not include store size, total rent, labor cost, or cost of goods sold
- The model does not include opening hours, seasonality, competitor quality, or brand effects
- Risk bands are heuristic, not formal confidence intervals
- The model may perform poorly on locations outside the training profile
- High accuracy on the bundled data does not imply real-world accuracy
- Predictions are decision-support estimates, not final answers

The project is strongest as a portfolio demonstration of an honest, reproducible regression workflow.

---

## Responsible Use

This repository is intended for:

- machine learning and data-analytics education
- SQL feature engineering and regression practice
- demonstrating honest holdout evaluation
- retail analytics workflow design
- responsible-ML documentation practice
- portfolio demonstration

It should not be used as-is for:

- real lease-signing decisions without deeper analysis
- financial forecasting without validation
- automated investment decisions
- replacing expert real-estate judgment
- any high-stakes business decision without human review

Any real deployment would require diverse, validated data, total-cost modeling, geographic validation, and a human review process.

---

## Future Improvements

Potential next improvements:

- Add real store size and compute total monthly rent
- Add labor cost, cost of goods sold, opening hours, and seasonality
- Add competitor density and quality using geographic distance
- Add SHAP or permutation-importance explanations
- Add map-based candidate visualization
- Add a Streamlit dashboard and Docker support

---

## Tech Stack

- Python
- pandas
- NumPy
- SQLite
- scikit-learn
- matplotlib
- joblib
- unittest
- Ruff
- Black
- mypy
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is intended for educational, research, and portfolio purposes.

If you use or modify this project, please keep the responsible-use notes and limitations clear.
