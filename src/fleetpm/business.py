"""Translate model scores into workshop decisions and AED savings.

Policy simulated on the test period:
  * each day, a vehicle whose risk score >= threshold is sent for a preventive
    inspection, unless it was already inspected in the last 14 days;
  * an inspection costs INSPECTION_COST and prevents a breakdown in the next
    14 days with probability PREVENTION_RATE;
  * a breakdown that is not prevented costs BREAKDOWN_COST (towing, emergency
    repair, vehicle downtime, missed deliveries / SLA penalties).

All cost assumptions are parameters, so they can be replaced with a client's
real figures in a workshop session.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

BREAKDOWN_COST = 6500.0
INSPECTION_COST = 600.0
PREVENTION_RATE = 0.7
COOLDOWN_DAYS = 14


@dataclass
class PolicyResult:
    threshold: float
    inspections: int
    failures: int
    failures_caught: int
    cost_without_model_aed: float
    cost_with_model_aed: float
    net_savings_aed: float
    savings_pct: float

    def to_dict(self):
        return asdict(self)


def simulate(scored: pd.DataFrame, failures: pd.DataFrame, threshold: float,
             breakdown_cost: float = BREAKDOWN_COST, inspection_cost: float = INSPECTION_COST,
             prevention_rate: float = PREVENTION_RATE) -> PolicyResult:
    """scored: vehicle_id, date, score.  failures: vehicle_id, date of each breakdown in the period."""
    inspections = []
    for vid, grp in scored[scored["score"] >= threshold].sort_values("date").groupby("vehicle_id"):
        last = None
        for d in grp["date"]:
            if last is None or (d - last).days >= COOLDOWN_DAYS:
                inspections.append((vid, d))
                last = d
    insp = pd.DataFrame(inspections, columns=["vehicle_id", "date"])

    caught = 0
    if len(insp) and len(failures):
        by_vehicle = {v: g["date"].to_numpy() for v, g in insp.groupby("vehicle_id")}
        for vid, fdate in failures[["vehicle_id", "date"]].itertuples(index=False):
            dates = by_vehicle.get(vid)
            if dates is None:
                continue
            delta = (np.datetime64(pd.Timestamp(fdate), "D") - dates.astype("datetime64[D]")).astype(int)
            if np.any((delta >= 1) & (delta <= COOLDOWN_DAYS)):
                caught += 1

    n_fail = len(failures)
    without = n_fail * breakdown_cost
    prevented = caught * prevention_rate
    with_model = len(insp) * inspection_cost + (n_fail - prevented) * breakdown_cost
    savings = without - with_model
    return PolicyResult(round(float(threshold), 4), len(insp), n_fail, caught, round(without, 0),
                        round(with_model, 0), round(savings, 0), round(100 * savings / without, 1) if without else 0.0)


def best_threshold(scored: pd.DataFrame, failures: pd.DataFrame, grid=None) -> PolicyResult:
    grid = grid if grid is not None else np.unique(np.quantile(scored["score"], np.linspace(0.80, 0.999, 60)))
    results = [simulate(scored, failures, t) for t in grid]
    return max(results, key=lambda r: r.net_savings_aed)
