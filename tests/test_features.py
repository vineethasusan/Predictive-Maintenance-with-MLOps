import pandas as pd

from fleetpm.features import FEATURES, HORIZON_DAYS, add_label, make_dataset, time_split


def test_label_looks_forward_only():
    df = pd.DataFrame({"vehicle_id": 1, "date": pd.date_range("2026-01-01", periods=30),
                       "failure": [0] * 20 + [1] + [0] * 9})
    lab = add_label(df)["fail_next_14d"].tolist()
    assert lab[20] == 0                       # failure day itself is not "next 14 days"
    assert lab[19] == 1 and lab[6] == 1       # 1 and 14 days before
    assert lab[5] == 0                        # 15 days before


def test_dataset_has_all_features_and_no_nans(raw):
    ds = make_dataset(raw)
    assert set(FEATURES) <= set(ds.columns)
    assert not ds[FEATURES].isna().any().any()


def test_time_split_has_gap_and_no_overlap(raw):
    ds = make_dataset(raw)
    train, test, cutoff = time_split(ds, test_days=60)
    assert train["date"].max() <= cutoff - pd.Timedelta(days=HORIZON_DAYS)
    assert test["date"].min() > cutoff


def test_rolling_features_do_not_use_future(raw):
    from fleetpm.features import build_features

    full = build_features(raw)
    cut = raw[raw["date"] <= raw["date"].min() + pd.Timedelta(days=100)]
    part = build_features(cut)
    day = part["date"].max()
    a = full[full["date"] == day].set_index("vehicle_id")["coolant_temp_avg_7d"]
    b = part[part["date"] == day].set_index("vehicle_id")["coolant_temp_avg_7d"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())
