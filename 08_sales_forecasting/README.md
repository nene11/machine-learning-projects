# Sales Forecasting & Demand Planning
**Level:** Advanced | **Task:** Time-series regression

## Objective
Build a leakage-safe forecasting workflow using lagged demand, rolling statistics and calendar features.

## Workflow
Chronological split → lag/rolling feature engineering → naive baseline → gradient boosting → backtesting → MAE/RMSE → error analysis.

## Key engineering rule
No future observations may influence features used to predict the past.

**Execution status:** NOT EXECUTED in this repository snapshot.
