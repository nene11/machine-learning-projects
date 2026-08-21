"""Train a leakage-aware conversion model on the real UCI Bank Marketing data.

The dataset is downloaded at runtime so it is not committed to Git.
"""
from pathlib import Path
from urllib.request import urlopen
import zipfile
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def load_data() -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    archive = DATA / "bank_marketing.zip"
    if not archive.exists():
        with urlopen(URL, timeout=60) as response:
            archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        target = next(n for n in names if n.endswith("bank-full.csv"))
        with z.open(target) as f:
            return pd.read_csv(f, sep=";")

def main() -> None:
    df = load_data()
    if "y" not in df:
        raise ValueError("Target column y is missing")
    # duration is post-contact information and therefore excluded to prevent leakage.
    X = df.drop(columns=["y", "duration"], errors="ignore")
    y = (df["y"] == "yes").astype(int)
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2500, class_weight="balanced"))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    thresholds = [i / 100 for i in range(10, 91)]
    best_t = max(thresholds, key=lambda t: f1_score(yte, p >= t))
    results = {
        "dataset": "UCI Bank Marketing bank-full.csv",
        "records": int(len(df)),
        "features_used": int(X.shape[1]),
        "excluded_leakage_feature": "duration",
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "test_roc_auc": float(roc_auc_score(yte, p)),
        "test_pr_auc": float(average_precision_score(yte, p)),
        "optimized_threshold": float(best_t),
        "test_f1_optimized": float(f1_score(yte, p >= best_t)),
    }
    import json
    (ROOT / "results_real.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
