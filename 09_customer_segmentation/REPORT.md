# Project 09 — Customer Segmentation & RFM Analytics

## Objective
Demonstrate unsupervised customer segmentation using standardized behavioral features and K-Means, with cluster-count evaluation through silhouette score.

## Dataset and scope
The current implementation uses a **synthetic clustered development benchmark**. It validates the clustering workflow but does not represent real customers or business segments.

## Pipeline
1. Prepare behavioral features.
2. Scale features to comparable ranges.
3. Train K-Means for candidate cluster structures.
4. Evaluate cluster quality with silhouette score.
5. Select the development configuration.
6. Produce cluster labels for downstream profiling.

## Verified execution
GitHub Actions portfolio CI run #28 executed the project successfully under Python 3.10 and 3.11.

### Metrics — Python 3.11 execution
| Metric | Value |
|---|---:|
| Number of clusters | 4 |
| Silhouette score | 0.7151 |

These are **VERIFIED algorithmic benchmark results**, not evidence of real customer behavior or marketing uplift.

## Testing
Portfolio CI completed successfully. Four smoke tests across the relevant projects passed, with pytest configured to avoid duplicate test-module import conflicts.

## Limitations and next step
A portfolio-grade business segmentation should derive RFM features from a documented public transaction dataset, compare multiple K values, profile clusters with interpretable statistics and validate segments against downstream business outcomes.

## Evidence status
- Clustering execution: VERIFIED
- Silhouette score: VERIFIED
- Real customer data: NOT USED
- Business segment validity: NOT TESTED
- Production use: NOT READY
