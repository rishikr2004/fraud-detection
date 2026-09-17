from fastapi.testclient import TestClient
from src.serve import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_requires_30_features():
    response = client.post("/predict", json={"features": [0.1, 0.2]})
    assert response.status_code == 422  # validation error, too few features
