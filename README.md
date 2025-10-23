# Coffee Shop Profit Predictor (SQL + Python)

Predict the **monthly profit** of potential coffee-shop locations using **SQL (SQLite)** for data preparation and **Python** for regression modeling and visualization.

This project simulates a **data-driven site selection workflow**, helping retail managers or founders decide **where to open their next store** based on real-world features such as rent, income, foot traffic, and local events.

---

## Overview

- **Goal:** Estimate monthly profit for candidate locations.  
- **Tech Stack:** SQLite, pandas, scikit-learn, matplotlib.  
- **Core Methods:** Feature engineering in SQL, ElasticNet regression for balanced interpretability and performance.  
- **Outputs:**  
  - Metrics (`R²`, `MAE`)  
  - Profit predictions  
  - Feature importance  
  - Diagnostic plots  

---

## Project Structure

```
coffee-shop-profit-predictor/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ locations_train.csv
│  └─ locations_candidates.csv
├─ src/
│  ├─ create_db.py
│  ├─ queries.sql
│  ├─ train_regression.py
│  ├─ score_new_sites.py
│  └─ utils.py
└─ outputs/
   ├─ metrics.json
   ├─ feature_importance.csv
   ├─ predictions_train.csv
   ├─ scored_candidates.csv
   └─ charts/
      ├─ actual_vs_predicted.png
      ├─ residuals_hist.png
      └─ feature_importance.png
```

---

## Data Fields

| Column | Description |
|--------|-------------|
| `lat`, `lon` | Coordinates of the shop location |
| `foot_traffic` | Average daily pedestrians |
| `rent_per_sqm` | Monthly rent per square meter (€) |
| `competition` | Number of similar cafés within 500 m |
| `median_income` | Median neighborhood income (€) |
| `office_density` | Offices within 500 m |
| `weekend_activity` | Weekend leisure index (0–1) |
| `events_per_month` | Number of local events monthly |
| `coffee_price` | Average sale price (€) |
| `promo_spend` | Monthly local promo budget (€) |
| `profit` | Monthly profit (€), **target variable** |

---

## SQL Feature Engineering

Feature creation is done in `queries.sql`, producing two views:  
`features_train` and `features_candidates`.

**Key engineered features:**
- `demand_adj` = foot traffic / (1 + competition)  
- `wknd_traffic` = weekend activity × traffic  
- `price_income` = coffee price × income factor  
- `promo_comp_adj` = promo spend / (1 + competition)

These normalized interaction features help the regression model capture **competitive intensity**, **affordability**, and **demand dynamics**.

---

## How to Run

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Load CSVs into SQLite
python src/create_db.py --train data/locations_train.csv --candidates data/locations_candidates.csv --db coffee.db

# Train and evaluate the model
python src/train_regression.py --db coffee.db --sql src/queries.sql --outdir outputs

# Score new candidate sites
python src/score_new_sites.py --db coffee.db --sql src/queries.sql --model outputs/model.joblib --outdir outputs
```

---

## Results and Visualizations

### **Actual vs Predicted Profit**
<img width="1050" height="900" alt="actual_vs_predicted" src="https://github.com/user-attachments/assets/fa7ef156-4248-4dc6-a2ed-09ff80577570" />

Each point represents a store location.  
- The **diagonal line (y = x)** indicates perfect prediction.  
- Points close to the line show accurate profit estimation.  
- A strong linear trend confirms the model’s predictive ability.  
- Outliers correspond to unusual cases (very high rent or event density).

**Insight:**  
The model captures real-world profitability patterns.  
Occasional underestimation of high-profit stores suggests nonlinear effects that could be explored later.

---

### **Feature Importance (Standardized Coefficients)**
<img width="1200" height="750" alt="feature_importance" src="https://github.com/user-attachments/assets/03852ce8-fc3a-456a-b33f-5a35da4dd11c" />

| Feature | Effect | Interpretation |
|----------|---------|----------------|
| `events_per_month` | ↑ | More events → higher profits. |
| `rent_per_sqm` | ↓ | Rent strongly decreases profit. |
| `price_income` | ↑ | High-income areas tolerate higher prices. |
| `demand_adj` | ↑ | Traffic normalized by competition. |
| `promo_comp_adj` | ↑ | Promotional spending offsets competition. |

**Insight:**  
Rent and local events dominate the financial outcome.  
Marketing and demographics play measurable supporting roles.

---

### **Residuals (Actual − Predicted)**
<img width="1050" height="750" alt="residuals_hist" src="https://github.com/user-attachments/assets/9f882334-28ca-4318-8444-a64221d085a8" />

- Residuals are roughly **bell-shaped and centered near 0**, meaning the model has no systematic bias.  
- Few outliers → stable performance.  
- Balanced tails indicate good generalization.

**Insight:**  
Model errors are random and well-distributed.  
Future work could explore nonlinear regressors or tree-based ensembles.

---

## Model Summary

| Metric | Description | Value (example) |
|--------|--------------|----------------|
| `R²` | Explained variance (goodness of fit) | ≈ 0.85 |
| `MAE` | Average absolute prediction error (€) | ≈ 180 € |

ElasticNet combines **L1 and L2 regularization**, preventing overfitting while keeping coefficients interpretable.

---

## Key Business Insights

1. **Event density** is the top profit driver, opening near cultural or office events maximizes demand.  
2. **Rent** remains the most significant cost factor. Even small rent changes have major profit impacts.  
3. **Wealthier neighborhoods** sustain higher pricing strategies.  
4. **Balanced marketing** spending helps counter intense competition.

---

## Technical Notes

- **Database:** SQLite (`coffee.db`)  
- **Model:** ElasticNet (α = 0.05, L₁ = 0.2)  
- **Preprocessing:** Standard scaling via `ColumnTransformer`  
- **Evaluation:** R², MAE, diagnostic plots  
- **Reproducibility:** Random seed = 42  

---

## Portfolio Value

This project showcases:
- **Practical SQL + Python integration**
- **Realistic data-science workflow**: data → SQL features → ML → interpretability  
- **Business storytelling through visuals**
- **Model transparency and actionable insights**

Ideal for demonstrating **retail analytics**, **forecasting**, and **data-driven decision-making** in a professional portfolio.
