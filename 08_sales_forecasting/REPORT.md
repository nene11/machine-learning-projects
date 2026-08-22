# Project 08 — Sales Forecasting & Demand Planning

## Objective
Forecast a univariate demand series while preventing temporal leakage through chronological splitting and past-only lag/rolling features.

## Dataset and scope
The current project uses a **synthetic development benchmark**. It is suitable for validating the forecasting pipeline, but it is not evidence of real company sales performance.

## Pipeline
1. Generate/load the development time series.
2. Create lagged features from past observations.
3. Create rolling statistics using only historical values.
4. Add calendar/time features where applicable.
5. Use a chronological train/test split.
6. Train the gradient-boosting regression model.
7. Evaluate with MAE and RMSE.

## Verified execution
GitHub Actions portfolio CI run #28 executed the project successfully under Python 3.10 and 3.11.

### Metrics — Python 3.11 execution
| Metric | Value |
|---|---:|
| MAE | 9.1741 |
| RMSE | 11.5306 |

These are **VERIFIED development-benchmark metrics**, not business forecasting results.

## Leakage control
No future observations are used to construct features for earlier predictions. The chronological split is part of the model-development design.

## Testing
Portfolio CI completed successfully and the cross-project smoke-test suite passed.

## Limitations and next step
The synthetic series must be replaced with a documented public or real business sales dataset before using this project as evidence of domain performance. A stronger next iteration should add rolling-origin backtesting across multiple forecast horizons and compare against seasonal naive baselines.

## Evidence status
- Pipeline execution: VERIFIED
- Metrics: VERIFIED
- Real sales dataset: NOT USED
- Business impact: NOT TESTED
- Production forecasting: NOT READY
