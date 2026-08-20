# Detailed Report — House Price Regression

Objective: estimate residential sale prices from mixed numerical and categorical features.

Pipeline: load OpenML Ames data; remove identifier; log-transform target; split data; impute missing values; scale numeric features; one-hot encode categoricals; train Elastic Net; evaluate RMSE, MAE and R².

The preprocessing is encapsulated in a single pipeline to prevent train/test leakage.

**Results: NOT EXECUTED.** Run `src/train.py` to generate verified metrics.

Portfolio value: demonstrates regression, regularization, feature preprocessing and reproducible evaluation.