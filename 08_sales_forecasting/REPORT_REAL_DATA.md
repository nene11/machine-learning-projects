# Sales / Energy Forecasting — Real Public Dataset

## Dataset
UCI Individual Household Electric Power Consumption (dataset 235), with 2,075,259 one-minute measurements over almost four years.

## Method
Daily aggregation of global active power → chronological 80/20 split → lag features → shifted rolling means → calendar features → HistGradientBoostingRegressor → comparison with a naive lag baseline.

## Leakage control
The split is chronological. Rolling features are shifted one day so the target day cannot contribute to its own predictors.

## Evidence
`src/train_real.py` is the reproducible execution path. Metrics are written to `results_real.json` after execution. Until the GitHub Actions run is observed as successful, metrics are **NOT VERIFIED** in this report.

## Limitation
This is an energy-demand forecasting benchmark rather than a retail-sales dataset. It strengthens the time-series forecasting evidence in the portfolio; a true retail-sales forecast can be added later if a suitable public dataset is selected.
