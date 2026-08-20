"""Real-data forecasting pipeline using UCI household power consumption.

Downloads the official UCI archive at runtime, aggregates to daily consumption,
creates leakage-safe lag/rolling/calendar features, performs a chronological split,
and evaluates a HistGradientBoostingRegressor against a naive lag-1 baseline.
"""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import io, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
raw = urlopen(URL, timeout=120).read()
with ZipFile(io.BytesIO(raw)) as z:
    member = "household_power_consumption.txt"
    with z.open(member) as f:
        df = pd.read_csv(f, sep=";", na_values="?", low_memory=False)

df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
df["global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
df = df[["timestamp", "global_active_power"]].dropna().set_index("timestamp")
# Aggregate one-minute measurements to daily kWh.
daily = (df["global_active_power"].resample("D").sum() / 60.0).rename("target").to_frame()
for lag in (1, 7, 14, 28):
    daily[f"lag_{lag}"] = daily["target"].shift(lag)
daily["roll7"] = daily["target"].shift(1).rolling(7).mean()
daily["roll28"] = daily["target"].shift(1).rolling(28).mean()
daily["dow"] = daily.index.dayofweek
daily["month"] = daily.index.month
daily = daily.dropna()

cut = int(len(daily) * 0.8)
train, test = daily.iloc[:cut], daily.iloc[cut:]
features = [c for c in daily.columns if c != "target"]
model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
model.fit(train[features], train["target"])
pred = model.predict(test[features])
naive = test["lag_1"].to_numpy()

results = {
    "dataset": "UCI Individual Household Electric Power Consumption (dataset 235)",
    "daily_rows": int(len(daily)),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "missing_rows_after_cleaning": int(df["global_active_power"].isna().sum()),
    "model_mae": float(mean_absolute_error(test["target"], pred)),
    "model_rmse": float(mean_squared_error(test["target"], pred) ** 0.5),
    "naive_mae": float(mean_absolute_error(test["target"], naive)),
    "naive_rmse": float(mean_squared_error(test["target"], naive) ** 0.5),
    "leakage_control": "chronological split; all rolling features shifted by one day",
}
(ROOT / "results_real.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
