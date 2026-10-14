"""Feature engineering and labelling.

Label: `fail_next_14d` = 1 if the vehicle breaks down in the next 14 days
(not counting today). This is the question operations actually asks:
"which vehicles should come to the workshop this fortnight?"

All features use only information available up to and including the
prediction day, and the train/test split is by time, so there is no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_DAYS = 14

SENSOR_COLS = ["coolant_temp_avg", "coolant_temp_max", "oil_pressure_psi", "battery_voltage", "dtc_count"]
USAGE_COLS = ["km", "engine_hours", "idle_hours", "harsh_events"]

FEATURES = [
    "age_years", "km_since_service", "days_since_service", "vehicle_type_code",
    *[f"{c}_7d" for c in SENSOR_COLS + USAGE_COLS],
    *[f"{c}_30d" for c in SENSOR_COLS + USAGE_COLS],
    "coolant_trend", "oil_trend", "battery_trend", "dtc_7d_sum", "km_since_service_ratio",
    "failures_last_180d",
]

TYPE_CODES = {"Light Van": 0, "Pickup": 1, "Bus": 2, "Truck": 3}
SERVICE_INTERVAL = {"Light Van": 7500, "Pickup": 10000, "Bus": 10000, "Truck": 10000}


def add_label(df: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    df = df.sort_values(["vehicle_id", "date"]).copy()
    g = df.groupby("vehicle_id")["failure"]
    # sum of failures in the next `horizon` days, excluding today
    future = g.transform(lambda s: s[::-1].rolling(horizon, min_periods=1).sum()[::-1].shift(-1))
    df["fail_next_14d"] = (future.fillna(0) > 0).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["vehicle_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby("vehicle_id")
    for c in SENSOR_COLS + USAGE_COLS:
        df[f"{c}_7d"] = g[c].transform(lambda s: s.rolling(7, min_periods=1).mean())
        df[f"{c}_30d"] = g[c].transform(lambda s: s.rolling(30, min_periods=1).mean())
    # Trends: short-term vs long-term (positive = getting hotter / higher)
    df["coolant_trend"] = df["coolant_temp_avg_7d"] - df["coolant_temp_avg_30d"]
    df["oil_trend"] = df["oil_pressure_psi_7d"] - df["oil_pressure_psi_30d"]
    df["battery_trend"] = df["battery_voltage_7d"] - df["battery_voltage_30d"]
    df["dtc_7d_sum"] = g["dtc_count"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["km_since_service_ratio"] = df["km_since_service"] / df["vehicle_type"].map(SERVICE_INTERVAL)
    df["vehicle_type_code"] = df["vehicle_type"].map(TYPE_CODES)
    # failures in the previous 180 days (strictly before today)
    df["failures_last_180d"] = g["failure"].transform(lambda s: s.shift(1).rolling(180, min_periods=1).sum()).fillna(0)
    return df


def make_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_label(build_features(raw))
    # Rows where the vehicle has just failed are in the workshop: not a prediction moment
    df = df[df["failure"] == 0]
    # Need 30 days of history and a full 14-day label window
    first = df.groupby("vehicle_id")["date"].transform("min")
    last_day = df["date"].max()
    return df[(df["date"] >= first + pd.Timedelta(days=30)) &
              (df["date"] <= last_day - pd.Timedelta(days=HORIZON_DAYS))].reset_index(drop=True)


def time_split(df: pd.DataFrame, test_days: int = 90):
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    # gap of HORIZON_DAYS so training labels never look into the test period
    train = df[df["date"] <= cutoff - pd.Timedelta(days=HORIZON_DAYS)]
    test = df[df["date"] > cutoff]
    return train, test, cutoff


def X_y(df: pd.DataFrame):
    return df[FEATURES].astype(float).to_numpy(), df["fail_next_14d"].to_numpy(dtype=int)


def feature_frame(records: list[dict]) -> np.ndarray:
    """Turn API payloads (already-aggregated features) into a model matrix."""
    return pd.DataFrame(records)[FEATURES].astype(float).to_numpy()
