"""Daily batch scoring: produce the workshop priority list.

    python -m fleetpm.score_fleet --data data/telemetry.csv.gz --out reports/workshop_list.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from .predictor import Predictor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telemetry.csv.gz")
    ap.add_argument("--as-of")
    ap.add_argument("--out", default="reports/workshop_list.csv")
    a = ap.parse_args()
    scored = Predictor().score_fleet(pd.read_csv(a.data, parse_dates=["date"]), a.as_of)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(a.out, index=False)
    print(scored["risk_band"].value_counts().to_string())
    print(f"\nTop 10 vehicles to inspect:\n{scored.head(10).to_string(index=False)}")
