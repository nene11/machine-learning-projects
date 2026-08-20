# Marketing Campaign Optimization & Conversion Prediction
**Level:** Advanced | **Task:** Classification + business analytics

## Dataset
UCI **Bank Marketing** (`bank-full.csv`): 45,211 records and 16 input variables in the full dataset. The task is to predict whether a client subscribes to a term deposit. Source: UCI Machine Learning Repository. The dataset is licensed CC BY 4.0. citeturn0search0

## Leakage control
`duration` is excluded because it is observed after the phone call and would not be available for pre-contact targeting.

## Workflow
Data download → quality handling → stratified split → preprocessing → logistic baseline → 5-fold stratified CV → ROC-AUC / PR-AUC → optimized F1 threshold → limitations.

## Execution status
**REAL DATA PIPELINE: EXECUTED BY GITHUB ACTIONS ON PUSH — remote run must be checked before treating the metrics as VERIFIED.**

The repository intentionally does not hard-code unverified metrics. The workflow produces `results_real.json` as an artifact.

## Source
urlUCI Bank Marketing datasethttps://archive.ics.uci.edu/dataset/222/bank%2Bmarketing
