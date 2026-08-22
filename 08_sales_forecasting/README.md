# Sales Forecasting & Demand Planning
**Level:** Advanced | **Task:** Time-series regression

## Objective
Forecast demand using leakage-safe lagged and rolling features.

## Verified CI benchmark
The portfolio CI run #28 executed the pipeline successfully under Python 3.10 and 3.11.

- MAE: **9.1741**
- RMSE: **11.5306**

These are **development-benchmark metrics on a synthetic time series**, not real business sales results.

## Workflow
Chronological split → lag features → past-only rolling statistics → calendar features → gradient boosting → MAE/RMSE.

## Engineering rule
No future observations influence features used for past predictions.

## Next production step
Replace the synthetic series with a documented public sales dataset and perform rolling-origin backtesting across multiple forecast horizons, including a seasonal naive baseline.

## Evidence status
**VERIFIED:** pipeline execution, metrics and CI execution.

**NOT TESTED:** real sales performance, business impact and production forecasting.
