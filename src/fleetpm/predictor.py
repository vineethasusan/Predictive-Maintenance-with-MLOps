"""Load the production model and score vehicles. Shared by the API and batch jobs."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, build_features


def risk_band(p: float, threshold: float) -> str:
    if p >= threshold:
        return "high"
    if p >= threshold * 0.5:
        return "medium"
    return "low"


class Predictor:
    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        d = Path(artifacts_dir)
        self.model = joblib.load(d / "model.joblib")
        self.card = json.loads((d / "model_card.json").read_text())
        self.threshold = float(self.card["threshold"])

    def score_features(self, rows: list[dict]) -> list[dict]:
        X = pd.DataFrame(rows)[FEATURES].astype(float).to_numpy()
        p = self.model.predict_proba(X)[:, 1]
        return [{"risk_score": round(float(s), 4), "risk_band": risk_band(float(s), self.threshold),
                 "inspect_now": bool(s >= self.threshold)} for s in p]

    def score_fleet(self, raw: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
        """Score every vehicle on a given day from raw daily telemetry (batch use)."""
        feats = build_features(raw)
        day = pd.Timestamp(as_of) if as_of else feats["date"].max()
        today = feats[feats["date"] == day].copy()
        today["risk_score"] = self.model.predict_proba(today[FEATURES].astype(float).to_numpy())[:, 1]
        today["risk_band"] = [risk_band(p, self.threshold) for p in today["risk_score"]]
        today["inspect_now"] = today["risk_score"] >= self.threshold
        cols = ["vehicle_id", "vehicle_type", "age_years", "km_since_service", "dtc_7d_sum",
                "coolant_trend", "risk_score", "risk_band", "inspect_now"]
        return today[cols].sort_values("risk_score", ascending=False).reset_index(drop=True)


def explain_drivers(predictor: Predictor, row: dict, top: int = 3) -> list[str]:
    """Simple, model-agnostic explanation: which features push this vehicle's risk up most.

    Each feature is replaced by the fleet-typical value from the training reference and
    the drop in risk is measured (a lightweight local sensitivity analysis).
    """
    ref = pd.read_csv(Path("artifacts") / "reference_features.csv.gz")
    base = pd.DataFrame([row])[FEATURES].astype(float)
    p0 = predictor.model.predict_proba(base.to_numpy())[0, 1]
    impacts = []
    for f in FEATURES:
        alt = base.copy()
        alt[f] = float(np.median(ref[f]))
        impacts.append((p0 - predictor.model.predict_proba(alt.to_numpy())[0, 1], f))
    return [f for d, f in sorted(impacts, reverse=True)[:top] if d > 0]
