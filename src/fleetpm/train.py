"""Train, compare and register the breakdown-risk model.

    python -m fleetpm.train --data data/telemetry.csv.gz

Steps
  1. Build features and labels, split by time (train | 14-day gap | test).
  2. Train a logistic-regression baseline and a gradient-boosting model.
  3. On a validation slice of the training period (never the test set), pick
     each model's AED-maximising threshold, then pick the model that saves most.
  4. Report ML metrics and business impact on the held-out test period.
  5. Log everything to MLflow (if available) and save the model + model card.
  6. Quality gate: only promote a model that beats random ranking, saves money
     and is at least as good as the current production model.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import business
from .features import FEATURES, HORIZON_DAYS, X_y, make_dataset, time_split

MIN_LIFT = 3.0  # PR-AUC must be at least 3x the base rate

try:  # MLflow is optional so the project runs anywhere; use it when installed
    import mlflow
    import mlflow.sklearn

    HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    HAS_MLFLOW = False


def models(seed: int = 42) -> dict:
    return {
        "logistic_baseline": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=150, random_state=seed),
    }


def failures_in(raw: pd.DataFrame, start, end) -> pd.DataFrame:
    f = raw[(raw["failure"] == 1) & (raw["date"] > start) & (raw["date"] <= end)]
    return f[["vehicle_id", "date"]]


def evaluate(model, df: pd.DataFrame, raw: pd.DataFrame, threshold: float) -> dict:
    X, y = X_y(df)
    p = model.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)
    scored = df[["vehicle_id", "date"]].assign(score=p)
    # failures that the period's prediction rows could have warned about
    fails = failures_in(raw, df["date"].min(), df["date"].max() + pd.Timedelta(days=HORIZON_DAYS))
    pol = business.simulate(scored, fails, threshold)
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "positive_rate": round(float(y.mean()), 4),
        **{f"biz_{k}": v for k, v in pol.to_dict().items()},
    }


def train(data_path: str, out_dir: str = "artifacts", experiment: str = "fleet-predictive-maintenance",
          seed: int = 42) -> dict:
    t0 = time.perf_counter()
    raw = pd.read_csv(data_path, parse_dates=["date"])
    df = make_dataset(raw)
    train_df, test_df, cutoff = time_split(df)
    # validation = last 60 days of the training window, used only to choose the threshold
    v_cut = train_df["date"].max() - pd.Timedelta(days=60)
    fit_df = train_df[train_df["date"] <= v_cut - pd.Timedelta(days=HORIZON_DAYS)]
    val_df = train_df[train_df["date"] > v_cut]

    if HAS_MLFLOW:
        mlflow.set_experiment(experiment)

    results = {}
    for name, model in models(seed).items():
        run = mlflow.start_run(run_name=name) if HAS_MLFLOW else None
        try:
            Xf, yf = X_y(fit_df)
            model.fit(Xf, yf)
            val_scored = val_df[["vehicle_id", "date"]].assign(score=model.predict_proba(X_y(val_df)[0])[:, 1])
            val_fails = failures_in(raw, val_df["date"].min(), val_df["date"].max() + pd.Timedelta(days=HORIZON_DAYS))
            val_policy = business.best_threshold(val_scored, val_fails)
            thr = val_policy.threshold
            val_pr_auc = average_precision_score(X_y(val_df)[1], val_scored["score"])

            # refit on the full training window with the chosen threshold
            Xt, yt = X_y(train_df)
            model.fit(Xt, yt)
            metrics = evaluate(model, test_df, raw, thr)
            results[name] = {"model": model, "threshold": thr, "metrics": metrics,
                             "val_savings_aed": val_policy.net_savings_aed, "val_pr_auc": round(val_pr_auc, 4)}

            if HAS_MLFLOW:
                mlflow.log_params({"model": name, "horizon_days": HORIZON_DAYS, "threshold": thr,
                                   "train_rows": len(train_df), "test_rows": len(test_df),
                                   "cutoff": str(cutoff.date()), "n_features": len(FEATURES)})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                mlflow.log_metrics({"val_savings_aed": val_policy.net_savings_aed, "val_pr_auc": val_pr_auc})
                mlflow.sklearn.log_model(model, "model")
        finally:
            if run:
                mlflow.end_run()

    # Model selection uses the VALIDATION period only (the test period stays untouched),
    # and optimises the business objective: net AED savings, with PR-AUC as tie-breaker.
    best_name = max(results, key=lambda n: (results[n]["val_savings_aed"], results[n]["val_pr_auc"]))
    best, base = results[best_name], results["logistic_baseline"]
    m = best["metrics"]
    # Quality gate: the model must (1) be clearly better than random ranking,
    # (2) save money under the cost assumptions, and (3) not be worse than the
    # model currently in production (champion / challenger).
    lift = m["pr_auc"] / max(m["positive_rate"], 1e-9)
    champion_card = Path(out_dir) / "model_card.json"
    champion_pr_auc = None
    if champion_card.exists():
        try:
            champion_pr_auc = json.loads(champion_card.read_text())["test_metrics"]["pr_auc"]
        except (KeyError, json.JSONDecodeError):
            champion_pr_auc = None
    gate = {
        "pr_auc_lift_vs_random": round(lift, 2),
        "lift_ok": lift >= MIN_LIFT,
        "saves_money": m["biz_net_savings_aed"] > 0,
        "champion_pr_auc": champion_pr_auc,
        "beats_champion": champion_pr_auc is None or m["pr_auc"] >= champion_pr_auc - 0.01,
    }
    gate_passed = gate["lift_ok"] and gate["saves_money"] and gate["beats_champion"]
    if not gate_passed:
        card = {"quality_gate_passed": False, "gate": gate, "candidate_metrics": m, "algorithm": best_name}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "rejected_candidate.json").write_text(json.dumps(card, indent=2))
        return card

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], out / "model.joblib")
    # reference feature distribution for drift monitoring
    train_df[FEATURES].sample(min(20000, len(train_df)), random_state=seed).to_csv(
        out / "reference_features.csv.gz", index=False)

    card = {
        "model_name": "fleet-breakdown-risk",
        "version": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "algorithm": best_name,
        "target": f"breakdown within the next {HORIZON_DAYS} days",
        "threshold": best["threshold"],
        "features": FEATURES,
        "train_period": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
        "test_period": [str(test_df["date"].min().date()), str(test_df["date"].max().date())],
        "test_metrics": best["metrics"],
        "baseline_metrics": base["metrics"],
        "validation": {n: {"net_savings_aed": r["val_savings_aed"], "pr_auc": r["val_pr_auc"]} for n, r in results.items()},
        "quality_gate_passed": True,
        "gate": gate,
        "cost_assumptions_aed": {"breakdown": business.BREAKDOWN_COST, "inspection": business.INSPECTION_COST,
                                 "prevention_rate": business.PREVENTION_RATE},
        "training_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telemetry.csv.gz")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    card = train(a.data, a.out)
    if not card["quality_gate_passed"]:
        print(f"Quality gate FAILED, production model unchanged: {card['gate']}")
        raise SystemExit(1)
    m = card["test_metrics"]
    print(f"Best model: {card['algorithm']}  (quality gate PASSED, lift {card['gate']['pr_auc_lift_vs_random']}x)")
    for name, v in card["validation"].items():
        print(f"  validation  {name:<18} savings AED {v['net_savings_aed']:>9,.0f} | PR-AUC {v['pr_auc']}")
    print(f"  TEST        PR-AUC {m['pr_auc']} | ROC-AUC {m['roc_auc']} | positive rate {m['positive_rate']:.2%}")
    print(f"  Test period: {m['biz_failures_caught']}/{m['biz_failures']} breakdowns flagged in advance, "
          f"{m['biz_inspections']} inspections")
    print(f"  Estimated net savings: AED {m['biz_net_savings_aed']:,.0f} ({m['biz_savings_pct']}% of breakdown cost)")
    print(f"Saved model and model card to {a.out}/")

if __name__ == "__main__":
    main()
