# Project 07 — Marketing Campaign Optimization & Conversion Prediction

## Objective
Build a pre-contact classification pipeline to predict whether a client will subscribe to a term deposit, with leakage control and reproducible evaluation.

## Dataset
- UCI Bank Marketing, `bank-full.csv`
- 45,211 records
- Target: `y` (yes/no)
- Dataset downloaded at runtime; it is not committed to Git.

## Data preparation and leakage control
`duration` is excluded because it is observed after the phone call. Including it would leak post-contact information into a pre-contact targeting model.

Numeric features are median-imputed and standardized. Categorical features are most-frequent-imputed and one-hot encoded inside a single scikit-learn pipeline.

## Modeling
1. Stratified 80/20 train/test split.
2. Class-weighted Logistic Regression baseline.
3. Five-fold stratified cross-validation on the training set.
4. Held-out evaluation using ROC-AUC, PR-AUC and F1.
5. F1-oriented threshold selection.

## Verified execution
GitHub Actions portfolio CI run #28 executed the project successfully under Python 3.10 and 3.11.

### Metrics — Python 3.11 execution
| Metric | Value |
|---|---:|
| ROC-AUC | 0.9349 |
| PR-AUC | 0.8436 |
| F1 | 0.7672 |

These are **VERIFIED model-evaluation results** and are not claims of business ROI.

## Testing
The portfolio CI collected four smoke tests across projects 07, 09 and 10; all passed. The workflow uses pytest `--import-mode=importlib` to avoid test-module name collisions.

## Limitations
- Logistic Regression is a baseline; tree-based models and calibrated decision policies remain possible improvements.
- Threshold selection must be revalidated against campaign costs and benefits before deployment.
- No uplift/causal modeling or live campaign experiment was performed.

## Evidence status
- Dataset: VERIFIED
- Leakage control: VERIFIED
- Training execution: VERIFIED
- Metrics: VERIFIED
- Business impact/ROI: NOT TESTED
- Production deployment: NOT TESTED
