# Marketing Campaign Optimization & Conversion Prediction
**Level:** Advanced | **Task:** Classification + business analytics

## Dataset
UCI **Bank Marketing** (`bank-full.csv`): 45,211 records. The target is whether a client subscribes to a term deposit.

## Leakage control
`duration` is excluded because it is observed after the phone call and would not be available for pre-contact targeting.

## Workflow
Runtime dataset download → leakage review → preprocessing → class-weighted Logistic Regression → 5-fold stratified CV → held-out ROC-AUC / PR-AUC / F1 → threshold selection.

## Verified CI benchmark
Portfolio CI run #28 executed the pipeline successfully under Python 3.10 and 3.11.

- ROC-AUC: **0.9349**
- PR-AUC: **0.8436**
- F1: **0.7672**

These are verified model metrics, not business ROI claims.

## Evidence status
**VERIFIED:** dataset pipeline, leakage control, model execution, evaluation metrics and CI execution.

**NOT TESTED:** live campaign ROI, uplift/causal impact, production deployment.

## Source
UCI Bank Marketing dataset: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing
