"""Leakage-safe forecasting on the real UCI Bike Sharing dataset."""
from pathlib import Path
from urllib.request import urlopen
import zipfile
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results_real.json"


def load_data() -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    archive = DATA / "bike_sharing.zip"
    if not archive.exists():
        with urlopen(URL, timeout=60) as r:
            archive.write_bytes(r.read())
    with zipfile.ZipFile(archive) as z:
        with z.open("day.csv") as f:
            df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["dteday"])
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    df = load_data()
    target = "cnt"
    # registered/casual are components of cnt and are excluded to avoid target leakage.
    features = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
    X = df[features].copy()
    y = df[target].astype(float)
    cut = int(len(df) * 0.8)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
    model.fit(X.iloc[:cut], y.iloc[:cut])
    pred = model.predict(X.iloc[cut:])
    results = {
        "dataset": "UCI Bike Sharing Dataset day.csv",
        "records": int(len(df)),
        "train_records": int(cut),
        "test_records": int(len(df) - cut),
        "features": features,
        "excluded_leakage_features": ["casual", "registered"],
        "split": "chronological 80/20",
        "mae": float(mean_absolute_error(y.iloc[cut:], pred)),
        "rmse": float(np.sqrt(mean_squared_error(y.iloc[cut:], pred))),
    }
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
