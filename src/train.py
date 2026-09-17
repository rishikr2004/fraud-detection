"""
Train a fraud detection model, comparing baseline (Logistic Regression) vs.
LightGBM with class weighting, tracked in MLflow.

Run: python src/train.py
"""
import joblib
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from data_loader import load_raw_data, get_train_test_split

MODEL_DIR = "models"
EXPERIMENT_NAME = "fraud-detection"


def scale_features(X_train, X_test):
    """Scale Amount and Time; V1-V28 are already PCA components (roughly scaled)."""
    scaler = StandardScaler()
    cols_to_scale = ["Time", "Amount"]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    return X_train_scaled, X_test_scaled, scaler


def train_baseline(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    # scale_pos_weight ~ (# negative / # positive), lets the model natively
    # handle imbalance without altering the data distribution via resampling.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "pr_auc": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_raw_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # --- Baseline ---
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        baseline = train_baseline(X_train, y_train)
        metrics = evaluate(baseline, X_test, y_test)
        mlflow.log_params({"model": "logistic_regression", "class_weight": "balanced"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(baseline, "model")
        print("Baseline:", metrics)

    # --- Main model ---
    with mlflow.start_run(run_name="lightgbm_main"):
        model = train_lightgbm(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_params({"model": "lightgbm", "imbalance_strategy": "scale_pos_weight"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
        print("LightGBM:", metrics)

    # Save artifacts for serving
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/model.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    print(f"Saved model + scaler to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
