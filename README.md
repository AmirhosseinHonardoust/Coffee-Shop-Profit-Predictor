<div align="center">
  
# Coffee Shop Profit Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![SQLite](https://img.shields.io/badge/SQLite-Feature%20Engineering-lightgrey) ![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-orange) ![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-green) [![CI](https://github.com/AmirhosseinHonardoust/Coffee-Shop-Profit-Predictor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Coffee-Shop-Profit-Predictor/actions/workflows/ci.yml)


</div>

A professional machine learning project that predicts the **monthly profit** of potential coffee-shop locations using **SQL feature engineering** and a **Python / scikit-learn regression workflow**. The project loads location data into SQLite, creates reusable engineered features, compares multiple regression models, evaluates performance honestly, and ranks new candidate sites with business-readable risk explanations.

> **Important:** This project is a **site-selection decision-support demo**, not a production financial forecasting system.
> It does not guarantee real-world profitability. It estimates likely profit from the available features and should be interpreted with the project limitations in mind.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Features](#features)
- [Charts and Visual Analysis](#charts-and-visual-analysis)
- [How the Model Works](#how-the-model-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Building the Database](#building-the-database)
- [Training the Model](#training-the-model)
- [Scoring Candidate Sites](#scoring-candidate-sites)
- [Model Output](#model-output)
- [Evaluation](#evaluation)
- [Testing](#testing)
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

Coffee-shop site selection is a practical business problem. A strong location can increase sales, while a poor location can create high rent costs, low traffic, and weak profitability.

This project demonstrates a clean end-to-end analytics workflow:

- Load training and candidate-location data from CSV files
- Store the data in a SQLite database
- Create reusable SQL feature views
- Train and compare regression models
- Evaluate the selected model on a holdout test set
- Score new candidate sites
- Save metrics, charts, predictions, and ranked recommendations
- Document limitations clearly

The goal is to demonstrate:

- SQL + Python integration
- Practical feature engineering
- Reproducible machine learning workflow design
- Honest model evaluation
- Business-readable candidate ranking
- Professional portfolio documentation

---

## What This Project Does

This project can:

- Predict monthly profit for coffee-shop locations
- Engineer features using SQL views
- Compare multiple regression models
- Use cross-validation for model selection
- Compare against a mean baseline
- Generate holdout test-set diagnostics
- Save trained model artifacts
- Score new candidate locations
- Rank candidates by predicted profitability
- Add risk bands and explanation columns
- Run automated tests
- Document data and model limitations

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
- Make final business decisions without human review

A production site-selection system would require real transaction data, lease terms, store size, local regulations, labor costs, competitor quality, seasonality, and geographic validation.

---

## Features

- **SQLite database creation** from CSV inputs
- **SQL feature engineering** through reusable views
- **Centralized configuration** in `src/config.py`
- **Data validation** for required columns, missing values, numeric types, and impossible values
- **Regression model comparison** using scikit-learn
- **Mean baseline model** for honest comparison
- **Linear Regression, Ridge, ElasticNet, Random Forest, and Gradient Boosting** experiments
- **Cross-validation** for model comparison
- **Hyperparameter tuning** for Ridge and ElasticNet
- **Holdout test-set evaluation**
- **Saved sklearn model** using `joblib`
- **Candidate-site ranking**
- **Risk band assignment**
- **Business-readable positive and negative drivers**
- **Charts for model diagnostics**
- **Unit tests** with Python `unittest`
- **Portfolio-ready README documentation**

---

## Charts and Visual Analysis

The project automatically generates visual outputs during training to make model behavior easier to understand.

Generated charts are saved in:

```text
outputs/charts/
```

The main charts include:

| Chart | Purpose |
|---|---|
| Actual vs Predicted | Compares holdout actual profit with predicted profit |
| Residuals Histogram | Shows model error distribution |
| Feature Importance | Shows the strongest model signals |
| Model Comparison MAE | Compares models using mean absolute error |

### Actual vs Predicted Profit

<img width="550" height="400" alt="actual_vs_predicted" src="https://github.com/user-attachments/assets/1975709e-240c-41b6-b539-f4c61177df82" />

This chart uses **holdout test-set predictions only**. That makes the diagnostic more honest than plotting predictions on data the model already trained on.

### Residuals Histogram

<img width="550" height="350" alt="residuals_hist" src="https://github.com/user-attachments/assets/69b37b0f-4c5e-4640-9463-2145121fb2c8" />

The residual chart shows how far predictions are from actual profit values. A narrower distribution means lower prediction error.

### Feature Importance

<img width="550" height="350" alt="feature_importance" src="https://github.com/user-attachments/assets/7d06bcca-69be-451f-9fcd-e13a22ed5df8" />

For the selected ElasticNet model, feature importance is based on standardized coefficients. These values should be interpreted as **model associations**, not causal proof.

### Model Comparison

<img width="550" height="350" alt="model_comparison_mae" src="https://github.com/user-attachments/assets/d0d9131d-876f-4a68-9477-9e99d801288e" />

The model comparison chart helps show whether the selected model meaningfully improves over simpler alternatives and the mean baseline.

---

## How the Model Works

The project uses a structured regression workflow:

```text
CSV location data
        ↓
SQLite database
        ↓
SQL feature views
        ↓
Data validation
        ↓
Train/test split
        ↓
Model comparison + cross-validation
        ↓
Selected regression pipeline
        ↓
Holdout evaluation
        ↓
Candidate-site scoring
        ↓
Ranked recommendations with risk bands
```

### SQL Feature Engineering

Feature creation is performed in `src/queries.sql`, which creates two SQLite views:

```text
features_train
features_candidates
```

Engineered features include:

| Feature | Formula | Purpose |
|---|---|---|
| `demand_adj` | `foot_traffic / (1 + competition)` | Demand adjusted for competitive intensity |
| `wknd_traffic` | `weekend_activity × foot_traffic` | Weekend demand potential |
| `price_income` | `coffee_price × (median_income / 1000)` | Price-income fit / affordability signal |
| `promo_comp_adj` | `promo_spend / (1 + competition)` | Promotion adjusted for competition |

### Regression Models

The training script compares several models:

- Mean Baseline
- Linear Regression
- Ridge
- ElasticNet
- Random Forest
- Gradient Boosting

ElasticNet is currently selected because it has the best mean cross-validated MAE among the non-baseline models in the current run.

---

## Project Structure

```text
Coffee-Shop-Profit-Predictor/
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
│   ├── config.py
│   ├── create_db.py
│   ├── queries.sql
│   ├── score_new_sites.py
│   ├── train_regression.py
│   └── utils.py
│
├── tests/
│   └── test_workflow.py
│
├── README.md
└── requirements.txt
```

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

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## Building the Database

Run:

```bash
python src/create_db.py --train data/locations_train.csv --candidates data/locations_candidates.csv --db coffee.db
```

This will:

- Read the training CSV file
- Read the candidate-site CSV file
- Validate required columns and numeric ranges
- Create a SQLite database
- Save the data into database tables

Created database:

```text
coffee.db
```

---

## Training the Model

Run:

```bash
python src/train_regression.py --db coffee.db --sql src/queries.sql --outdir outputs
```

This will:

- Load SQL feature views
- Split the training data into train and holdout sets
- Compare multiple regression models
- Run cross-validation
- Select the best non-baseline model by cross-validated MAE
- Evaluate the selected model on the holdout test set
- Save the trained model
- Save predictions, metrics, model comparison results, and charts

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

---

## Scoring Candidate Sites

Run:

```bash
python src/score_new_sites.py --db coffee.db --sql src/queries.sql --model outputs/model.joblib --outdir outputs
```

This will:

- Load candidate-site features from SQLite
- Apply the trained model
- Predict monthly profit
- Rank candidate locations
- Add risk bands
- Add business-readable positive and negative drivers
- Save the scored candidate file

Generated output:

```text
outputs/scored_candidates.csv
```

---

## Model Output

Candidate scoring returns ranked results with fields such as:

| Field | Meaning |
|---|---|
| `rank` | Candidate rank by predicted monthly profit |
| `lat`, `lon` | Candidate coordinates |
| `predicted_profit` | Estimated monthly profit in euros |
| `risk_band` | Heuristic risk level based on feature profile distance |
| `expected_error_eur` | Approximate expected error based on model MAE |
| `profile_distance` | Distance from the training-data feature profile |
| `main_positive_drivers` | Business-readable features helping the score |
| `main_negative_drivers` | Business-readable features hurting the score |

Example output format:

```text
Rank: 1
Predicted profit: €2,004.10
Risk band: high
Main positive drivers: high competition-adjusted promotion; high competition-adjusted demand; high coffee price
Main negative drivers: no major red flags
```

The risk band is a practical warning flag, not a formal statistical confidence interval.

---

## Evaluation

The project uses a more responsible evaluation workflow than a single in-sample score.

Evaluation includes:

- Train/test split
- Mean baseline comparison
- Cross-validation
- Holdout R²
- Holdout MAE
- Model comparison table
- Test-set diagnostic charts

Metrics are saved to:

```text
outputs/metrics.json
```

Model comparison is saved to:

```text
outputs/model_comparison.csv
```

Charts are saved to:

```text
outputs/charts/
```

### Current Results

The current run selects **ElasticNet**.

| Metric | Value |
|---|---:|
| Holdout R² | `0.647` |
| Holdout MAE | `€383.58` |
| Mean baseline R² | `-0.030` |
| Mean baseline MAE | `€699.39` |
| 5-fold CV R² | `0.530 ± 0.110` |
| 5-fold CV MAE | `€413.67 ± €46.65` |

### Why This Matters

A common mistake in beginner machine learning projects is reporting overly optimistic scores from training data. This project avoids that by using a separate holdout test set and cross-validation.

The model improves clearly over the mean baseline, but the cross-validation spread shows that results should still be interpreted carefully because the dataset is small.

---

## Testing

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

The tests check important project behavior, including:

- Python source files compile
- CSV data loads into SQLite
- SQL feature views are created correctly
- Required engineered feature columns exist
- Data validation rejects impossible values

Example passing result:

```text
Ran 3 tests
OK
```

---

## Code Quality

The project includes several maintainability improvements:

- Centralized project settings in `src/config.py`
- Shared validation helpers in `src/utils.py`
- Clear separation between database creation, model training, and candidate scoring
- Reusable SQL feature definitions
- Saved model artifacts
- Saved evaluation outputs
- Automated tests

These choices make the project easier to reproduce, review, and extend.

---

## Data Statement

This repository is presented as a **demonstration / simulated retail analytics project**.

The included dataset contains:

```text
220 training locations
60 candidate locations
```

The data is suitable for demonstrating a portfolio machine learning workflow, but it should not be treated as verified real business data unless the source is updated and documented.

If this project is later connected to real data, this section should be updated with:

- Data source
- Collection period
- Geographic coverage
- Feature definitions
- Usage permissions
- Known data quality issues

---

## Limitations

This project has important limitations.

The model:

- Uses a small dataset
- Uses demonstration/simulated data
- Does not include store size
- Does not convert `rent_per_sqm` into total rent
- Does not include labor cost
- Does not include cost of goods sold
- Does not include opening hours
- Does not include seasonality
- Does not include competitor quality
- Does not include brand effects
- Does not include local regulations or lease terms
- Uses heuristic risk bands, not formal confidence intervals
- May perform poorly on locations outside the training-data profile
- Should not be used as the only basis for investment decisions

High performance on the included data does not guarantee high performance in real-world site selection.

---

## Responsible Use

This project is intended for:

- Machine learning education
- Data analyst / data scientist portfolio demonstration
- SQL feature engineering practice
- Regression modeling practice
- Retail analytics workflow design
- Responsible model documentation practice

It should not be used for:

- Real lease-signing decisions without deeper analysis
- Financial forecasting without validation
- Automated investment decisions
- Replacing expert real-estate judgment
- High-stakes business decisions without human review

Predictions should be treated as **decision-support estimates**, not final answers.

---

## Future Improvements

Possible future improvements include:

- Add real store size and calculate total monthly rent
- Add labor cost and cost of goods sold
- Add opening hours and seasonality
- Add competitor density using geographic distance
- Add competitor quality and brand strength
- Add local demographic and transit data
- Add map-based candidate visualization
- Add bootstrapped prediction intervals
- Add SHAP or permutation importance explanations
- Add a Streamlit dashboard
- Add Docker support
- Add GitHub Actions CI
- Add linting and formatting checks
- Add a full EDA notebook

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

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is intended for educational and portfolio purposes.

If you use or modify this project, please keep the limitations and responsible-use notes clear.
