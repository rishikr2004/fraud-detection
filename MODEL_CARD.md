# Model Card: Credit Card Fraud Detection

## Overview
LightGBM classifier predicting whether a credit card transaction is
fraudulent, trained on anonymized transaction data.

## Intended use
Portfolio/demo project illustrating an end-to-end ML engineering pipeline
for a highly imbalanced classification problem. **Not intended for
production use on real financial data without significant additional
validation.**

## Training data
- Kaggle "Credit Card Fraud Detection" dataset: 284,807 transactions from
  European cardholders over 2 days, 492 labeled as fraud (0.17%).
- Features V1–V28 are PCA-transformed and anonymized; only `Time` and
  `Amount` are raw/interpretable.

## Metrics
- PR-AUC (primary metric — accuracy is misleading at this imbalance rate)
- ROC-AUC (secondary)
- Threshold chosen via a cost model (see `src/evaluate.py`), not the
  default 0.5 — assumes a fixed cost per missed fraud and per false alarm.

## Known limitations / failure modes
1. **Anonymized features limit interpretability.** V1–V28 are PCA
   components with no real-world meaning, so the model can't be explained
   to a fraud analyst or customer in plain terms.
2. **No merchant, location, or device data.** Real fraud systems typically
   use much richer features (IP, device fingerprint, merchant category,
   velocity checks). This model only sees what's in the dataset.
3. **Temporal drift.** The dataset covers just 2 days. Real fraud patterns
   evolve constantly (new attack vectors, seasonal spending shifts) — a
   model trained on this snapshot would degrade in production without
   retraining and drift monitoring.
4. **Cost assumptions are illustrative.** `COST_PER_MISSED_FRAUD` and
   `COST_PER_FALSE_ALARM` in `evaluate.py` are placeholder values, not
   derived from real business data.
5. **No fairness/bias audit.** The dataset has no demographic information,
   so subgroup fairness cannot be assessed here — a real deployment would
   need this analysis using additional data.

## What would need to change for production
- Real-time feature pipeline (not static PCA components)
- Drift detection and automated retraining triggers
- A/B testing framework to validate threshold changes safely
- Human-in-the-loop review queue for borderline cases
- Regulatory/compliance review (fraud models in finance are often audited)
