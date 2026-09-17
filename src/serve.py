"""
FastAPI app serving the fraud detection model.

Run: uvicorn src.serve:app --reload
"""
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Fraud Detection API", version="1.0.0")

MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"
THRESHOLD_PATH = "models/threshold.txt"
DEFAULT_THRESHOLD = 0.5

_model = None
_scaler = None
_threshold = DEFAULT_THRESHOLD


def _load_artifacts():
    global _model, _scaler, _threshold
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model not found. Run `python src/train.py` first.")
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH) as f:
                _threshold = float(f.read().strip())


class PredictRequest(BaseModel):
    # 30 features: Time, V1..V28, Amount (order must match training data columns)
    features: list[float] = Field(..., min_length=30, max_length=30)


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float


@app.on_event("startup")
def startup():
    try:
        _load_artifacts()
    except RuntimeError:
        # Allow the app to start even without a trained model (e.g. for CI health checks)
        pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _load_artifacts()
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it first.")

    x = np.array(req.features).reshape(1, -1)
    # NOTE: assumes column order [Time, V1..V28, Amount] matches training.
    # Time and Amount (index 0 and -1) get scaled; V1-V28 pass through.
    x_scaled = x.copy()
    time_amount = _scaler.transform(x[:, [0, -1]])
    x_scaled[:, 0] = time_amount[:, 0]
    x_scaled[:, -1] = time_amount[:, 1]

    proba = float(_model.predict_proba(x_scaled)[0, 1])
    return PredictResponse(
        fraud_probability=proba,
        is_fraud=proba >= _threshold,
        threshold_used=_threshold,
    )
