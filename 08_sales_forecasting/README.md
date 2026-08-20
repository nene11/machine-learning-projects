# Sales Forecasting & Demand Planning
**Level:** Advanced | **Task:** Time-series regression

## Objective
Forecast demand using leakage-safe lagged and rolling features.

## Executed benchmark
The committed pipeline was executed locally with a fixed seed. **VERIFIED:** MAE 9.6388 and RMSE 11.9492 on a synthetic time-series development benchmark.

These are development-benchmark metrics, not real business sales results.

## Workflow
Chronological split → lag features → rolling statistics based only on past values → calendar feature → gradient boosting → MAE/RMSE.

## Engineering rule
No future observations influence features used for past predictions.

## Next production step
Replace the synthetic series with a documented public sales dataset and perform multi-window rolling-origin backtesting.
