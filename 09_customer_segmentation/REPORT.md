# Project 09 — Customer Segmentation & RFM Analytics

## Objective
Identify actionable customer groups from real transaction behavior using RFM features and K-Means clustering.

## Dataset
UCI Online Retail. The pipeline downloads the public transaction dataset at runtime.

## Data preparation
1. Validate required transaction columns.
2. Remove records without customer/date information.
3. Exclude cancellations and non-positive quantity/unit-price records.
4. Calculate revenue as quantity × unit price.
5. Build customer-level Recency, Frequency and Monetary features.
6. Apply log1p transformation and standardization.

## Model selection
K-Means is evaluated for k=2..6 using silhouette score. The highest-scoring candidate is selected deterministically.

## Metrics
The pipeline records customer count, silhouette score for every tested k and the selected k in `results_real.json`. Final real-data values will only be reported after the updated CI run completes.

## Reproducibility
```bash
python src/train.py
```

## Business interpretation
Silhouette score measures geometric separation; it does not prove that segments improve retention, conversion or revenue. Cluster profiles and downstream business validation are required before deployment.

## Evidence status
- Real public transaction dataset: **VERIFIED in code**
- RFM construction: **VERIFIED in code**
- Multi-k model selection: **VERIFIED in code**
- Final real-data metrics: **NOT YET VERIFIED after migration**
