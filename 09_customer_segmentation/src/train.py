"""RFM customer segmentation on the real UCI Online Retail dataset."""
from pathlib import Path
from urllib.request import urlopen
import zipfile
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results_real.json"


def load_data() -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    archive = DATA / "online_retail.zip"
    if not archive.exists():
        with urlopen(URL, timeout=60) as r:
            archive.write_bytes(r.read())
    with zipfile.ZipFile(archive) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            return pd.read_csv(f, encoding="ISO-8859-1")


def main() -> None:
    df = load_data()
    required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    # Credit notes/cancellations are excluded from positive-value customer behavior.
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (~df["InvoiceNo"].astype(str).str.startswith("C"))].copy()
    df["revenue"] = df["Quantity"] * df["UnitPrice"]
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("revenue", "sum"),
    )
    rfm = rfm[(rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)]
    X = rfm.apply(lambda c: __import__("numpy").log1p(c)).replace([float("inf"), float("-inf")], 0)
    X = StandardScaler().fit_transform(X)
    scores = {}
    models = {}
    for k in range(2, 7):
        m = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = m.fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
        models[k] = (m, labels)
    best_k = max(scores, key=scores.get)
    best_model, labels = models[best_k]
    results = {
        "dataset": "UCI Online Retail",
        "raw_records": int(len(df)),
        "customers": int(len(rfm)),
        "features": ["Recency", "Frequency", "Monetary"],
        "k_search": list(range(2, 7)),
        "silhouette_by_k": scores,
        "selected_k": int(best_k),
        "selected_silhouette": float(scores[best_k]),
    }
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
