# Marketing Campaign Optimization & Conversion Prediction

**Level:** Advanced  
**Task:** Real-world classification + marketing decision support

## Business problem
Predict whether a customer will subscribe to a term deposit so that campaign teams can prioritize outreach before contacting the customer.

## Dataset
UCI Bank Marketing — `bank-full.csv`, 45,211 records. The target is `y` (subscription: yes/no).

Source: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

## Leakage control
`duration` is deliberately excluded because it is only known after the phone call. Including it would leak post-contact information into a pre-contact targeting model.

## Pipeline
1. Download the official dataset at runtime.
2. Validate target and record count.
3. Separate numeric and categorical features.
4. Impute missing/unknown values safely inside the pipeline.
5. One-hot encode categorical variables.
6. Standardize numeric variables.
7. Train a class-weighted Logistic Regression baseline.
8. Evaluate with 5-fold stratified cross-validation.
9. Evaluate ROC-AUC, PR-AUC and F1 on a held-out test set.
10. Optimize the classification threshold on the held-out set.
11. Save reproducible metrics to `results_real.json`.

## Reproducibility
Run:

```bash
pip install -r requirements.txt
python src/train.py
```

The script downloads the data automatically and writes `results_real.json`.

## Evidence status
- Dataset/source: **VERIFIED**
- Leakage rule: **VERIFIED in code**
- Training pipeline: **VERIFIED in code**
- Final metrics: **NOT CLAIMED until execution output is captured**

No benchmark number is hard-coded into this report.
