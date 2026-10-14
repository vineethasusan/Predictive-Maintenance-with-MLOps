import pandas as pd

from fleetpm.business import BREAKDOWN_COST, INSPECTION_COST, simulate


def _scored(rows):
    return pd.DataFrame(rows, columns=["vehicle_id", "date", "score"]).assign(date=lambda d: pd.to_datetime(d.date))


def test_caught_failure_saves_money():
    scored = _scored([(1, "2026-03-01", 0.9), (1, "2026-03-02", 0.9)])
    fails = pd.DataFrame({"vehicle_id": [1], "date": pd.to_datetime(["2026-03-10"])})
    r = simulate(scored, fails, threshold=0.5, prevention_rate=1.0)
    assert r.inspections == 1                      # cooldown: no repeat inspection next day
    assert r.failures_caught == 1
    assert r.net_savings_aed == BREAKDOWN_COST - INSPECTION_COST


def test_no_flags_means_no_savings():
    scored = _scored([(1, "2026-03-01", 0.1)])
    fails = pd.DataFrame({"vehicle_id": [1], "date": pd.to_datetime(["2026-03-10"])})
    r = simulate(scored, fails, threshold=0.5)
    assert r.inspections == 0 and r.net_savings_aed == 0
