# Portfolio Verification & Self-Audit

## Scope
Projects 07–10 and the portfolio CI workflow.

## Evidence source
GitHub Actions `portfolio-ci` run #28 (`32538270911`).

## Verification matrix
| Gate | Status | Evidence |
|---|---|---|
| Core training execution | PASS | Projects 07–10 training steps completed successfully |
| Python 3.10 compatibility | PASS | CI matrix job completed successfully |
| Python 3.11 compatibility | PASS | CI matrix job completed successfully |
| Smoke tests | PASS | 4 tests collected, 4 passed |
| Test import collision | PASS | pytest `--import-mode=importlib`; renamed tests where applied |
| Node.js Actions compatibility | PASS | checkout@v6 and setup-python@v6 executed successfully |
| Project 07 real dataset | PASS | UCI Bank Marketing, 45,211 records |
| Project 07 leakage control | PASS | `duration` excluded in code |
| Project 07 metrics | PASS | ROC-AUC 0.9349, PR-AUC 0.8436, F1 0.7672 |
| Project 08 metrics | PASS | MAE 9.1741, RMSE 11.5306 |
| Project 09 metrics | PASS | Silhouette 0.7151, 4 clusters |
| Project 10 metric | PASS | ROC-AUC 0.9954 |
| Documentation updated | PASS | Project READMEs, reports and root portfolio README updated |
| Business impact | NOT TESTED | No live business experiment |
| Production deployment | NOT TESTED | No deployment/monitoring evidence |

## Known limitations
- Projects 08 and 09 still use synthetic development benchmarks and must not be represented as real business outcomes.
- Project 10 is production-oriented but is not production-ready; deployment, monitoring, security operations, load testing and drift controls remain unverified.
- Project 07 demonstrates predictive performance on public data, not causal campaign uplift or ROI.

## Definition of Done
For this portfolio-verification phase: **PASS**. Core execution, CI compatibility, smoke tests, verified metrics and documentation updates are complete. Production readiness is intentionally not claimed.

## Next recommended portfolio improvements
1. Replace Project 08 with a documented public sales dataset and rolling-origin backtesting.
2. Replace Project 09 with a documented public transaction dataset and RFM-based business validation.
3. Add model comparison and calibration/error analysis to Project 07.
4. Extend Project 10 with packaging, inference API, monitoring and deployment tests before claiming production readiness.
