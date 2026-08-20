"""Real-data RFM customer segmentation using UCI Online Retail."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import io, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
raw = urlopen(URL, timeout=120).read()
with ZipFile(io.BytesIO(raw)) as z:
    member = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
    df = pd.read_excel(io.BytesIO(z.read(member)))

rows_raw = len(df)
df = df.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]).copy()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (~df["InvoiceNo"].astype(str).str.startswith("C"))]
df["revenue"] = df["Quantity"] * df["UnitPrice"]
reference = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    recency=("InvoiceDate", lambda s: (reference - s.max()).days),
    frequency=("InvoiceNo", "nunique"),
    monetary=("revenue", "sum"),
)
# Log transform reduces the influence of highly skewed transaction behavior.
X = np.log1p(rfm)
X = StandardScaler().fit_transform(X)
results_by_k = {}
for k in range(2, 9):
    labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
    results_by_k[k] = float(silhouette_score(X, labels))
best_k = max(results_by_k, key=results_by_k.get)
model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
rfm["cluster"] = model.fit_predict(X)
profiles = rfm.groupby("cluster")[['recency','frequency','monetary']].agg(['count','mean','median']).round(2)
profiles.to_csv(ROOT / "cluster_profiles_real.csv")
results = {
    "dataset": "UCI Online Retail (dataset 352)",
    "raw_rows": int(rows_raw),
    "customer_rows": int(len(rfm)),
    "candidate_k_silhouette": results_by_k,
    "selected_k": int(best_k),
    "selected_silhouette": float(results_by_k[best_k]),
    "cleaning": "removed missing CustomerID/date, cancellations, non-positive quantity and non-positive price",
}
(ROOT / "results_real.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
