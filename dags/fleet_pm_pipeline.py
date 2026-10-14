"""Airflow DAG: weekly retraining and daily scoring for fleet predictive maintenance.

    ingest -> validate -> train (MLflow) -> quality gate -> drift check -> score fleet

Place this file in your Airflow `dags/` folder (docker-compose in this repo does it for you).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

PROJECT = Path("/opt/fleet-pm")
DATA = PROJECT / "data" / "telemetry.csv.gz"
ARTIFACTS = PROJECT / "artifacts"

default_args = {"owner": "data-science", "retries": 2, "retry_delay": timedelta(minutes=5)}


@dag(
    dag_id="fleet_predictive_maintenance",
    schedule="0 2 * * 0",          # Sundays 02:00 (UAE weekend, low workshop activity)
    start_date=datetime(2026, 10, 1),
    catchup=False,
    default_args=default_args,
    tags=["fleet", "ml", "predictive-maintenance"],
)
def fleet_pm_pipeline():
    @task
    def ingest() -> str:
        """In production: pull last night's telemetry from the warehouse (e.g. Databricks SQL).
        Here: regenerate the synthetic dataset so the demo is self-contained."""
        import sys
        sys.path.insert(0, str(PROJECT / "src"))
        from fleetpm.data import generate
        df = generate()
        DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA, index=False)
        return str(DATA)

    @task
    def validate(path: str) -> str:
        import pandas as pd
        df = pd.read_csv(path, parse_dates=["date"])
        checks = {
            "no_missing_ids": df["vehicle_id"].notna().all(),
            "no_negative_km": (df["km"] >= 0).all(),
            "plausible_coolant": df["coolant_temp_avg"].between(40, 140).all(),
            "plausible_battery": df["battery_voltage"].between(9, 16).all(),
            "fresh_data": (pd.Timestamp.now() - df["date"].max()).days < 400,
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"Data validation failed: {failed}")
        return path

    @task
    def train(path: str) -> dict:
        import sys
        sys.path.insert(0, str(PROJECT / "src"))
        from fleetpm.train import train as run_training
        return run_training(path, str(ARTIFACTS))

    @task
    def quality_gate(card: dict) -> dict:
        if not card["quality_gate_passed"]:
            raise ValueError(f"Model rejected by quality gate: {json.dumps(card.get('gate'))}")
        return card

    @task
    def drift_check(path: str) -> dict:
        import sys
        import pandas as pd
        sys.path.insert(0, str(PROJECT / "src"))
        from fleetpm.features import build_features
        from fleetpm.monitor import drift_report
        ref = pd.read_csv(ARTIFACTS / "reference_features.csv.gz")
        cur = build_features(pd.read_csv(path, parse_dates=["date"]))
        cur = cur[cur["date"] > cur["date"].max() - pd.Timedelta(days=30)]
        rep = drift_report(ref, cur)
        (PROJECT / "reports").mkdir(exist_ok=True)
        (PROJECT / "reports" / "drift_report.json").write_text(json.dumps(rep, indent=2))
        return {"n_alert": rep["n_alert"], "retrain_recommended": rep["retrain_recommended"]}

    @task
    def score_fleet(path: str, _card: dict) -> int:
        import sys
        import pandas as pd
        sys.path.insert(0, str(PROJECT / "src"))
        from fleetpm.predictor import Predictor
        scored = Predictor(ARTIFACTS).score_fleet(pd.read_csv(path, parse_dates=["date"]))
        scored.to_csv(PROJECT / "reports" / "workshop_list.csv", index=False)
        return int(scored["inspect_now"].sum())

    data = validate(ingest())
    card = quality_gate(train(data))
    drift_check(data)
    score_fleet(data, card)


fleet_pm_pipeline()
