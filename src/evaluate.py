"""
Pick a decision threshold based on a business cost model instead of the
default 0.5. This is the key differentiator for this project: it shows
product/business thinking, not just modeling.

Assumption (edit these to match your own reasoning in the README):
  - Missing a fraud (false negative) costs ~ the average fraudulent amount.
  - A false alarm (false positive) costs a fixed review/friction cost.
"""
import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve

from data_loader import load_raw_data, get_train_test_split
from train import scale_features

# --- Business cost assumptions (tune these and justify in your README) ---
COST_PER_MISSED_FRAUD = 500   # avg fraud loss if undetected ($)
COST_PER_FALSE_ALARM = 5      # cost of manual review / customer friction ($)


def find_optimal_threshold(y_true, y_proba) -> tuple[float, dict]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    best_threshold = 0.5
    best_cost = float("inf")
    n_fraud = int(y_true.sum())

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        tp = r * n_fraud
        fn = n_fraud - tp
        fp = tp * (1 - p) / p if p > 0 else 0

        total_cost = fn * COST_PER_MISSED_FRAUD + fp * COST_PER_FALSE_ALARM
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t

    return best_threshold, {"estimated_cost": best_cost}


def main():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    X_train, X_test, _ = scale_features(X_train, X_test)

    model = joblib.load("models/model.joblib")
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold, info = find_optimal_threshold(y_test, y_proba)
    print(f"Optimal threshold: {threshold:.4f}")
    print(f"Estimated cost at this threshold: ${info['estimated_cost']:.2f}")
    print(f"(vs. default 0.5 threshold — compare this yourself for the README)")

    with open("models/threshold.txt", "w") as f:
        f.write(str(threshold))


if __name__ == "__main__":
    main()
