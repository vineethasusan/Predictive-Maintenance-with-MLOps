"""Data drift monitoring with the Population Stability Index (PSI).

PSI compares this week's feature distribution with the training reference.
Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift
(retrain or investigate, e.g. a new vehicle type or a faulty sensor batch).

    python -m fleetpm.monitor --current data/telemetry.csv.gz --days 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, build_features

PSI_WARN, PSI_ALERT = 0.10, 0.25
RETRAIN_MIN_ALERTS = 5  # several features shifting together = the fleet or sensors have changed


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: compare category shares instead
        values = np.union1d(np.unique(reference), np.unique(current))
        ref = np.array([(reference == v).mean() for v in values])
        cur = np.array([(current == v).mean() for v in values])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref = np.histogram(reference, edges)[0] / len(reference)
        cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    rows = []
    for f in FEATURES:
        v = psi(reference[f].to_numpy(float), current[f].to_numpy(float))
        status = "alert" if v > PSI_ALERT else "warn" if v > PSI_WARN else "ok"
        rows.append({"feature": f, "psi": round(v, 4), "status": status})
    rows.sort(key=lambda r: -r["psi"])
    n_alert = sum(r["status"] == "alert" for r in rows)
    return {"features": rows, "n_alert": n_alert, "n_warn": sum(r["status"] == "warn" for r in rows),
            "retrain_recommended": n_alert >= RETRAIN_MIN_ALERTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="artifacts/reference_features.csv.gz")
    ap.add_argument("--current", default="data/telemetry.csv.gz")
    ap.add_argument("--days", type=int, default=30, help="Use the last N days of --current")
    ap.add_argument("--out", default="reports/drift_report.json")
    a = ap.parse_args()
    ref = pd.read_csv(a.reference)
    raw = pd.read_csv(a.current, parse_dates=["date"])
    cur = build_features(raw)
    cur = cur[cur["date"] > cur["date"].max() - pd.Timedelta(days=a.days)]
    rep = drift_report(ref, cur)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(f"Drift: {rep['n_alert']} alerts, {rep['n_warn']} warnings, retrain recommended: {rep['retrain_recommended']}")
    for r in rep["features"][:5]:
        print(f"  {r['feature']:<28} PSI {r['psi']:.3f}  {r['status']}")


if __name__ == "__main__":
    main()
