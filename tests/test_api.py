import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_predict_endpoint(trained, monkeypatch):
    import pandas as pd
    from fastapi.testclient import TestClient

    card, art = trained
    if not card["quality_gate_passed"]:
        pytest.skip("no model promoted on the tiny test dataset")
    monkeypatch.setenv("ARTIFACTS_DIR", str(art))
    from serve import api

    api.predictor.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    row = pd.read_csv(art / "reference_features.csv.gz").iloc[0].to_dict()
    r = client.post("/predict", json={"vehicles": [{"vehicle_id": "V1", "features": row}]})
    assert r.status_code == 200
    assert r.json()["predictions"][0]["vehicle_id"] == "V1"
    bad = client.post("/predict", json={"vehicles": [{"vehicle_id": "V1", "features": {"km": 1}}]})
    assert bad.status_code == 422
