"""Synthetic daily vehicle telemetry with realistic failure behaviour.

One row = one vehicle on one day. Breakdowns are drawn from a hazard model, so
failure risk genuinely rises with things a telematics unit can observe:
mileage since last service, engine temperature, oil pressure, battery voltage,
fault codes (DTCs), harsh driving and vehicle age. Sensor readings degrade in
the 1-3 weeks before a failure, which is what a predictive model can learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VEHICLE_TYPES = {
    # type: (daily km mean, service interval km)
    "Light Van": (140, 7500),
    "Pickup": (110, 10000),
    "Bus": (180, 10000),
    "Truck": (160, 10000),
}


def generate(n_vehicles: int = 400, start: str = "2025-10-01", days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for vid in range(1, n_vehicles + 1):
        vtype = rng.choice(list(VEHICLE_TYPES))
        km_mean, interval = VEHICLE_TYPES[vtype]
        age = int(rng.integers(0, 11))
        frailty = rng.lognormal(0, 0.35)              # some vehicles are simply weaker
        aggressive = rng.random() < 0.12              # driver style on this vehicle
        km_since_service = rng.uniform(0, interval)
        days_since_service = int(km_since_service / km_mean)
        degradation = 0.0                              # hidden wear state, 0..1+
        failing_in = None                              # countdown once a failure is "seeded"

        for d in dates:
            active = rng.random() > (0.12 if d.dayofweek == 4 else 0.04)
            km = max(0.0, rng.normal(km_mean, km_mean * 0.25)) if active else 0.0
            engine_h = km / rng.uniform(30, 45) + (rng.uniform(0.3, 1.5) if active else 0)
            idle_h = engine_h * rng.uniform(0.05, 0.25)
            harsh = rng.poisson(km / 100 * (2.5 if aggressive else 0.8))

            km_since_service += km
            days_since_service += 1
            overdue = max(0.0, km_since_service / interval - 1)
            degradation += km / 1e5 * frailty * (1 + age / 6) * (1 + 2.5 * overdue) + harsh * 0.0004

            # Seed a failure with probability driven by the hidden state
            if failing_in is None:
                hazard = 0.0004 * np.exp(3.2 * degradation) * (1 + overdue)
                if rng.random() < min(hazard, 0.25):
                    # ~35% of breakdowns are sudden (tyre, accident damage, electrical) with almost no warning
                    failing_in = int(rng.integers(1, 3)) if rng.random() < 0.35 else int(rng.integers(5, 22))
            pre = 0.0 if failing_in is None else 1 - failing_in / 22  # 0 -> 1 as failure nears

            coolant = rng.normal(88, 3.0) + 3 * degradation + 4.5 * pre * rng.random()
            coolant_max = coolant + abs(rng.normal(6, 2.5)) + 5 * pre * rng.random()
            oil = rng.normal(42, 3.5) - 4 * degradation - 5 * pre * rng.random()
            battery = rng.normal(13.9, 0.18) - 0.15 * age / 10 - 0.35 * pre * rng.random()
            dtc = rng.poisson(0.04 + 0.12 * degradation + 0.6 * pre)

            failure = 0
            if failing_in is not None:
                failing_in -= 1
                if failing_in <= 0:
                    failure = 1
            rows.append((vid, vtype, age, d, round(km, 1), round(engine_h, 2), round(idle_h, 2), int(harsh),
                         round(coolant, 1), round(coolant_max, 1), round(oil, 1), round(battery, 2), int(dtc),
                         round(km_since_service, 0), days_since_service, failure))

            # Workshop events reset state
            if failure:
                degradation *= 0.35
                km_since_service, days_since_service, failing_in = 0.0, 0, None
            elif km_since_service > interval * rng.uniform(0.9, 1.35) and rng.random() < 0.25:
                degradation *= 0.55                      # scheduled service
                km_since_service, days_since_service = 0.0, 0

    cols = ["vehicle_id", "vehicle_type", "age_years", "date", "km", "engine_hours", "idle_hours",
            "harsh_events", "coolant_temp_avg", "coolant_temp_max", "oil_pressure_psi", "battery_voltage",
            "dtc_count", "km_since_service", "days_since_service", "failure"]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/telemetry.csv.gz")
    ap.add_argument("--vehicles", type=int, default=400)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.vehicles, days=a.days, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df):,} rows, {df.failure.sum()} failures, {df.vehicle_id.nunique()} vehicles -> {a.out}")
