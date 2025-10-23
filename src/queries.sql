-- =========================================================
-- Feature engineering for Coffee Shop Profit Predictor
-- - demand_adj     : foot traffic adjusted for competition
-- - wknd_traffic   : weekend activity scaled by traffic
-- - price_income   : price × income (affordability/tolerance)
-- - promo_comp_adj : promo spend adjusted for competition
-- =========================================================

DROP VIEW IF EXISTS features_train;
CREATE TEMP VIEW features_train AS
SELECT
  lat,
  lon,
  foot_traffic,
  rent_per_sqm,
  competition,
  median_income,
  office_density,
  weekend_activity,
  events_per_month,
  coffee_price,
  promo_spend,
  (foot_traffic * 1.0) / (1 + competition)                         AS demand_adj,
  (weekend_activity * 1.0) * (foot_traffic * 1.0)                   AS wknd_traffic,
  (coffee_price * 1.0) * ((median_income * 1.0) / 1000.0)           AS price_income,
  (promo_spend * 1.0) / (1 + competition)                           AS promo_comp_adj,
  profit
FROM locations_train;

DROP VIEW IF EXISTS features_candidates;
CREATE TEMP VIEW features_candidates AS
SELECT
  lat,
  lon,
  foot_traffic,
  rent_per_sqm,
  competition,
  median_income,
  office_density,
  weekend_activity,
  events_per_month,
  coffee_price,
  promo_spend,
  (foot_traffic * 1.0) / (1 + competition)                         AS demand_adj,
  (weekend_activity * 1.0) * (foot_traffic * 1.0)                   AS wknd_traffic,
  (coffee_price * 1.0) * ((median_income * 1.0) / 1000.0)           AS price_income,
  (promo_spend * 1.0) / (1 + competition)                           AS promo_comp_adj
FROM locations_candidates;

-- Final SELECTs are executed explicitly from Python.
-- SELECT * FROM features_train;
-- SELECT * FROM features_candidates;
