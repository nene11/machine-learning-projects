# Project 10 — End-to-End ML Production Pipeline

## Objective
Build a reproducible, testable ML pipeline that can be extended toward an inference service and operated through CI.

## Dataset
Wisconsin Diagnostic Breast Cancer dataset loaded through scikit-learn.

## Pipeline
1. Load the public dataset.
2. Validate input structure.
3. Apply deterministic preprocessing.
4. Train the classification model.
5. Evaluate on a held-out test set.
6. Keep the training/evaluation flow reproducible.
7. Run smoke tests in CI.

## Verified execution
GitHub Actions portfolio CI run #28 executed the project successfully under Python 3.10 and 3.11.

### Metric — Python 3.11 execution
| Metric | Value |
|---|---:|
| ROC-AUC | 0.9954 |

This is a **VERIFIED held-out evaluation result** for the public dataset. It is not a claim of production performance.

## CI verification
The complete portfolio CI run passed. The smoke-test suite collected 4 tests and reported `4 passed` under Python 3.11. The same training/test workflow also passed under Python 3.10.

## Production readiness
**NOT READY for production deployment.** The repository does not yet verify deployment infrastructure, monitoring/alerting, secrets management, load/performance testing, model/data drift controls, backup/recovery or operational incident procedures.

## Evidence status
- Public dataset execution: VERIFIED
- Model evaluation: VERIFIED
- CI execution: VERIFIED
- Deployment: NOT TESTED
- Monitoring: NOT TESTED
- Production readiness: NOT READY

## Portfolio value
Demonstrates reproducible ML engineering, evaluation, test automation and CI integration rather than only notebook-level modeling.
