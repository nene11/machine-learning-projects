# Project 08 — Sales Forecasting & Demand Planning

## Objective
Forecast daily bike-sharing demand from information available before the forecast date.

## Dataset
UCI Bike Sharing Dataset (`day.csv`). The pipeline downloads the public dataset at runtime.

## Data preparation
- Parse and sort dates chronologically.
- Predict `cnt` (total rentals).
- Use season, year, month, holiday, weekday, working-day, weather, temperature, humidity and windspeed.
- Exclude `casual` and `registered` because they are components of `cnt` and would leak target information.

## Validation design
An 80/20 chronological split is used without random shuffling.

## Model
`HistGradientBoostingRegressor` with fixed random seed and regularization.

## Metrics
The pipeline calculates MAE and RMSE on the chronological holdout and writes them to `results_real.json`. Final real-data values will only be reported after the updated CI run completes.

## Reproducibility
```bash
python src/train.py
```

## Limitations
This public benchmark is not a production demand-planning system. Rolling-origin backtesting, forecasting intervals, operational covariates and a seasonal-naive baseline should be added before operational use.

## Evidence status
- Real public dataset: **VERIFIED in code**
- Leakage exclusions: **VERIFIED in code**
- Chronological split: **VERIFIED in code**
- Final real-data metrics: **NOT YET VERIFIED after migration**
