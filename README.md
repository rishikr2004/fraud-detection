# Credit Card Fraud Detection
[![CI](https://github.com/rishikr2004/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/rishikr2004/fraud-detection/actions/workflows/ci.yml)

An end-to-end ML pipeline that detects fraudulent credit card transactions,
built to demonstrate the full ML engineering lifecycle: data → training →
tracking → serving → deployment → CI/CD.

## Problem

Detect fraudulent transactions in a highly imbalanced dataset (~0.17% fraud).
The core challenge isn't modeling accuracy — it's handling extreme class
imbalance, choosing the right metric, and picking a decision threshold based
on real business cost tradeoffs (a missed fraud costs far more than a false
alarm).

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
— 284,807 transactions, 492 frauds. Features V1–V28 are PCA-transformed for
anonymity; `Time` and `Amount` are the only raw, interpretable features.

Download `creditcard.csv` and place it in `data/`.

## Project structure

```
fraud-detection/
├── data/                   # raw data (not committed — see .gitignore)
├── src/
│   ├── data_loader.py      # load + stratified split
│   ├── train.py            # training + MLflow tracking
│   ├── evaluate.py         # PR-AUC, threshold selection
│   └── serve.py            # FastAPI app
├── tests/
│   └── test_serve.py
├── models/                 # saved model artifacts (gitignored)
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
└── MODEL_CARD.md
```

## Key design decisions

1. **Stratified split happens before any resampling.** Balancing (SMOTE,
   undersampling) before the train/test split leaks information and inflates
   test metrics — a common mistake in fraud-detection portfolios.
2. **Metric: PR-AUC, not accuracy.** At 0.17% positive rate, a model that
   predicts "not fraud" every time scores 99.8% accuracy and is useless.
3. **Threshold is chosen from a cost model, not fixed at 0.5.** See
   `src/evaluate.py` — the threshold is picked by assuming a cost per missed
   fraud vs. cost per false alarm, not an arbitrary default.
4. **Model card included.** See `MODEL_CARD.md` for known limitations and
   failure modes.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place creditcard.csv in data/

# 3. Train
python src/train.py

# 4. Serve locally
uvicorn src.serve:app --reload

# 5. Or run with Docker
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## API

`POST /predict`
```json
{
  "features": [0.1, -1.2, 0.5, ...]  // V1-V28, Time, Amount — 30 values
}
```
Response:
```json
{ "fraud_probability": 0.87, "is_fraud": true, "threshold_used": 0.42 }
```

## Limitations

See `MODEL_CARD.md`.
