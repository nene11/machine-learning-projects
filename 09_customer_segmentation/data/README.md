# Data

This project uses the official UCI Online Retail dataset (dataset 352).

The dataset contains 541,909 transactions from a UK-based non-store online retailer between December 2010 and December 2011. Raw data is not committed to Git; the training script downloads the official dataset at runtime.

The segmentation workflow builds customer-level Recency, Frequency and Monetary (RFM) features, removes cancellation/invalid transactions according to documented rules, scales the features, evaluates candidate K values with silhouette score, and profiles the resulting clusters.
