# Customer Segmentation & RFM Analytics
**Level:** Intermediate–Advanced | **Task:** Unsupervised learning

## Objective
Identify customer groups using behavioral features and evaluate cluster quality.

## Verified CI benchmark
Portfolio CI run #28 executed the pipeline successfully under Python 3.10 and 3.11.

- Clusters: **4**
- Silhouette score: **0.7151**

These are **development-benchmark results on a synthetic clustered dataset**, not evidence of real customer segments.

## Workflow
Feature preparation → scaling → K-Means → silhouette evaluation → cluster profiling concept → marketing-action mapping.

## Next production step
Derive RFM features from a documented public transaction dataset, compare multiple K values and validate the resulting personas against business outcomes.

## Evidence status
**VERIFIED:** clustering execution, silhouette metric and CI execution.

**NOT TESTED:** real customer segmentation validity, campaign uplift and production use.
