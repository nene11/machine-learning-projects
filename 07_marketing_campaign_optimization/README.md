# Marketing Campaign Optimization & Conversion Prediction
**Level:** Advanced | **Task:** Classification + business analytics

## Objective
Predict conversion probability and translate model outputs into targeting decisions.

## Executed benchmark
The committed pipeline was executed locally with a fixed random seed. **VERIFIED:** ROC-AUC 0.9349, PR-AUC 0.8436, F1 0.7672.

The benchmark uses scikit-learn synthetic classification data. These numbers are **not** business campaign results and must not be presented as such.

## Workflow
Leakage review → preprocessing → logistic baseline → probability prediction → ROC-AUC / PR-AUC / F1 → business thresholding concept → limitations.

## Next production step
Replace the synthetic scaffold with a documented public campaign dataset and add calibration, cost-sensitive threshold optimization and business lift analysis.
