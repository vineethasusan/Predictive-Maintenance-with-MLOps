"""Breakdown-risk prediction API.

Run:   uvicorn serve.api:app --reload      (docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fleetpm.features import FEATURES  # noqa: E402
from fleetpm.predictor import Predictor, explain_drivers  # noqa: E402

app = FastAPI(title="Fleet Breakdown-Risk API", version="1.0.0",
              description="Scores commercial vehicles for breakdown risk in the next 14 days.")
METRICS = {"requests": 0, "predictions": 0, "errors": 0, "latency_ms_total": 0.0}


@lru_cache(maxsize=1)
def predictor() -> Predictor:
    return Predictor(os.getenv("ARTIFACTS_DIR", "artifacts"))


class VehicleFeatures(BaseModel):
    vehicle_id: str | int = Field(..., examples=["DXB-1042"])
    features: dict[str, float] = Field(..., description=f"Must contain: {', '.join(FEATURES)}")


class PredictRequest(BaseModel):
    vehicles: list[VehicleFeatures] = Field(..., min_length=1, max_length=5000)
    explain: bool = False


@app.middleware("http")
async def track(request: Request, call_next):
    t0 = time.perf_counter()
    METRICS["requests"] += 1
    try:
        response = await call_next(request)
    except Exception:
        METRICS["errors"] += 1
        raise
    METRICS["latency_ms_total"] += (time.perf_counter() - t0) * 1000
    return response


@app.get("/health")
def health():
    p = predictor()
    return {"status": "ok", "model": p.card["algorithm"], "version": p.card["version"]}


@app.get("/model-info")
def model_info():
    c = predictor().card
    return {k: c[k] for k in ("model_name", "version", "algorithm", "target", "threshold", "train_period",
                              "test_period", "test_metrics", "cost_assumptions_aed")}


@app.get("/metrics")
def metrics():
    n = max(METRICS["requests"], 1)
    return {**METRICS, "avg_latency_ms": round(METRICS["latency_ms_total"] / n, 2)}


@app.post("/predict")
def predict(req: PredictRequest):
    p = predictor()
    missing = sorted(set(FEATURES) - set(req.vehicles[0].features))
    if missing:
        raise HTTPException(422, f"Missing features: {missing[:8]}{'…' if len(missing) > 8 else ''}")
    rows = [v.features for v in req.vehicles]
    scores = p.score_features(rows)
    METRICS["predictions"] += len(scores)
    out = []
    for v, s in zip(req.vehicles, scores):
        item = {"vehicle_id": v.vehicle_id, **s}
        if req.explain and s["risk_band"] != "low":
            item["top_risk_drivers"] = explain_drivers(p, v.features)
        out.append(item)
    out.sort(key=lambda r: -r["risk_score"])
    return {"model_version": p.card["version"], "threshold": p.threshold, "predictions": out}
