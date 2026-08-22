# Machine Learning Portfolio — Nihal Idris Fadol Alsaied

**Data Scientist | Machine Learning | Analytics | MSc Computer Science**

A structured portfolio covering classical ML, NLP, time-series forecasting, computer vision, business analytics and production-oriented ML engineering.

## Portfolio map
| # | Project | Focus | Evidence |
|---|---|---|---|
| 01 | Customer Churn | Classification | Existing project |
| 02 | House Prices | Regression | Existing project |
| 03 | Credit Risk | Risk & Calibration | Existing project |
| 04 | News Topic Classification | NLP | Existing project |
| 05 | Energy Demand Forecasting | Time Series | Existing project |
| 06 | CIFAR-10 | Deep Learning | Existing project |
| 07 | Marketing Campaign Optimization | Conversion / Business ML | **Verified CI; real UCI data** |
| 08 | Sales Forecasting | Demand Planning | **Verified CI; synthetic development benchmark** |
| 09 | Customer Segmentation | Unsupervised ML | **Verified CI; synthetic development benchmark** |
| 10 | End-to-End ML Pipeline | Production-oriented ML | **Verified CI; public sklearn dataset** |
| 11 | Medical Imaging Transfer Learning | Deep Learning / CV | Blueprint; dataset execution pending |

## Verified CI results
Portfolio CI run **#28** completed successfully on Python 3.10 and 3.11. Both matrix jobs completed all four training steps and the smoke-test suite.

| Project | Verified result | Dataset scope |
|---|---|---|
| 07 Marketing | ROC-AUC **0.9349**, PR-AUC **0.8436**, F1 **0.7672** | UCI Bank Marketing, 45,211 records |
| 08 Sales Forecasting | MAE **9.1741**, RMSE **11.5306** | Synthetic development benchmark |
| 09 Customer Segmentation | Silhouette **0.7151**, 4 clusters | Synthetic development benchmark |
| 10 End-to-End ML | ROC-AUC **0.9954** | Wisconsin Diagnostic Breast Cancer via scikit-learn |

These are verified model-execution metrics. They are **not claims of employment outcomes, business ROI or production performance**.

## Engineering standard
Problem definition → data quality → leakage controls → preprocessing → baseline → modeling → evaluation → limitations → reproducibility → automated CI. Evidence is explicitly classified as VERIFIED / NOT TESTED / UNKNOWN.

## CI
GitHub Actions uses `actions/checkout@v6` and `actions/setup-python@v6`. Portfolio CI run #28 passed on Python 3.10 and 3.11. Pytest collected 4 smoke tests and reported **4 passed** using `--import-mode=importlib`.

## Portfolio positioning
The strongest CV-facing evidence is Project 07 (real public marketing data + leakage-aware classification) and Project 10 (reproducible ML pipeline + CI). Projects 08 and 09 are useful engineering demonstrations but should remain explicitly labeled as development benchmarks until moved to documented real datasets.

## Target roles
Data Scientist · Machine Learning Engineer · Data Analyst / Marketing Analytics · AI Engineer
