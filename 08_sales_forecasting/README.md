# Sales Forecasting & Demand Planning
**Level:** Advanced | **Task:** Time-series regression

## Objective
Forecast daily bike-sharing demand using a chronological split and leakage-safe features.

## Dataset
UCI Bike Sharing Dataset, `day.csv`. The training script downloads the public dataset at runtime from the UCI repository.

## Method
Chronological 80/20 split → calendar/weather features → HistGradientBoostingRegressor → MAE/RMSE.

`casual` and `registered` are deliberately excluded because they are components of the target `cnt` and would leak target information.

## Verification
**VERIFIED in code:** real UCI dataset loading, chronological split, leakage exclusions, deterministic model configuration and metric generation.

**NOT YET VERIFIED:** final real-dataset metrics until the updated CI run completes.

## Reproducibility
```bash
pip install -r requirements.txt
python src/train.py
```

The script creates `results_real.json` and does not hard-code benchmark numbers.

## Business interpretation
The model is a portfolio demonstration of demand forecasting. Its error should not be presented as a business KPI or operational forecast accuracy without domain-specific validation.

## Source
UCI Bike Sharing Dataset: https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset
