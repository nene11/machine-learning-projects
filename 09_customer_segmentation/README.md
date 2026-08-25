# Customer Segmentation & RFM Analytics
**Level:** Intermediate–Advanced | **Task:** Unsupervised learning

## Objective
Identify actionable customer groups from real transaction behavior using RFM features and K-Means clustering.

## Dataset
UCI Online Retail dataset. The training script downloads the public dataset at runtime from UCI.

## Method
Transaction cleaning → positive sales filtering → RFM construction → log transform → standardization → K-Means model selection for k=2..6 → silhouette evaluation.

Cancelled invoices and non-positive quantity/unit-price records are excluded from behavioral segmentation.

## Verification
**VERIFIED in code:** real UCI transaction loading, RFM feature construction, multiple-k evaluation, deterministic K-Means configuration and metric generation.

**NOT YET VERIFIED:** final real-dataset metrics until the updated CI run completes.

## Reproducibility
```bash
pip install -r requirements.txt
python src/train.py
```

The script creates `results_real.json` and does not hard-code benchmark numbers.

## Business interpretation
Segments should be validated against business outcomes before being used for targeting. Silhouette score measures geometric cluster separation; it does not prove marketing uplift.

## Source
UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online%2Bretail
