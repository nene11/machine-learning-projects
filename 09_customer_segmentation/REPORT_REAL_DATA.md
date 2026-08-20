# Customer Segmentation — Real Public Dataset

## Dataset
UCI Online Retail, dataset 352. It contains 541,909 transactions from a UK-based online retailer covering 01-Dec-2010 to 09-Dec-2011.

## Method
1. Load the official dataset at runtime.
2. Remove records without customer/date/transaction fields required for RFM.
3. Remove cancellations and non-positive quantity/price records.
4. Compute customer-level Recency, Frequency and Monetary value.
5. Apply log transformation and standardization.
6. Evaluate K=2..8 using silhouette score.
7. Select the best K and profile customer clusters.

## Evidence
`src/train_real.py` is the reproducible execution path. Metrics are written to `results_real.json` after execution. Until the GitHub Actions run is observed as successful, metrics are **NOT VERIFIED** in this report.

## Business interpretation
The final clusters should be named from their RFM profiles rather than from arbitrary cluster numbers. Recommended actions should be based on recency, purchase frequency and monetary value.
