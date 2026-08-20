# Detailed Report — News Topic Classification

Objective: classify documents into topical categories using a compact, reproducible NLP pipeline.

Steps: load train/test benchmark; remove metadata fields; fit TF-IDF with uni/bi-grams on training data only; train Linear SVM; evaluate accuracy and macro-F1; inspect per-class errors as a next analysis step.

The pipeline demonstrates sparse feature engineering and a strong classical NLP baseline without unnecessary deep-learning complexity.

**Results: NOT EXECUTED.** Run `src/train.py` before claiming metrics.