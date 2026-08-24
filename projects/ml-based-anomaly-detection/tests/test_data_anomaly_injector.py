import pandas as pd
import numpy as np

from src.data.data_anomaly_injector import (
    TSAnomalyConfig,
    normalize_timeseries,
    add_anomaly_columns,
    finalize_timeseries,
    inject_ts_anomalies,
)


def test_normalize_timeseries_grouping_and_conversion():
    raw = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "invalid", "2025-01-02", None],
            "daily_trip_count": [10, 20, 5, 15, 100],
        }
    )

    cfg = TSAnomalyConfig(min_rows=0)
    out = normalize_timeseries(raw, cfg)

    assert len(out) == 2
    assert list(out["date"]) == [
        pd.to_datetime("2025-01-01").date(),
        pd.to_datetime("2025-01-02").date(),
    ]
    assert (
        out.loc[
            out["date"] == pd.to_datetime("2025-01-01").date(), "daily_trip_count"
        ].iloc[0]
        == 30.0
    )
    assert (
        out.loc[
            out["date"] == pd.to_datetime("2025-01-02").date(), "daily_trip_count"
        ].iloc[0]
        == 15.0
    )


def test_add_anomaly_columns_defaults():
    df = pd.DataFrame(
        {"date": [pd.to_datetime("2025-01-01")], "daily_trip_count": [10.0]}
    )
    out = add_anomaly_columns(df)
    assert "true_anomaly" in out.columns
    assert "true_anomaly_type" in out.columns
    assert "anomaly_id" in out.columns
    assert out["true_anomaly"].iloc[0] == 0
    assert out["true_anomaly_type"].iloc[0] == "none"
    assert out["anomaly_id"].iloc[0] == -1


def test_finalize_timeseries_clips_and_rounds():
    cfg = TSAnomalyConfig(min_rows=0, clamp_non_negative=True)
    df = pd.DataFrame(
        {"date": [pd.to_datetime("2025-01-01")], "daily_trip_count": [-3.7]}
    )
    df = add_anomaly_columns(df)
    out = finalize_timeseries(df, cfg)

    assert out["daily_trip_count"].iloc[0] == 0.0


def test_inject_ts_anomalies_injects_classes_of_anomalies():
    cfg = TSAnomalyConfig(
        min_rows=5,
        zero_load_count=1,
        spike_day_count=1,
        label_drift_count=1,
        drift_window_range=(2, 2),
        seed=42,
    )

    dates = pd.date_range("2025-01-01", periods=8, freq="D")
    df = pd.DataFrame(
        {"date": dates, "daily_trip_count": np.arange(100, 108, dtype=float)}
    )

    out = inject_ts_anomalies(df, cfg)

    assert "true_anomaly" in out.columns
    assert "true_anomaly_type" in out.columns
    assert "anomaly_id" in out.columns

    # At least 3 rows should be marked anomaly (zero+spike + label drift 2 rows), maybe 4 depending placement.
    assert int(out["true_anomaly"].sum()) >= 3
    assert out["daily_trip_count"].min() >= 0


def test_inject_ts_anomalies_small_df_no_change():
    cfg = TSAnomalyConfig(
        min_rows=50, zero_load_count=1, spike_day_count=1, label_drift_count=1, seed=42
    )
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "daily_trip_count": np.arange(5, dtype=float),
        }
    )

    out = inject_ts_anomalies(df, cfg)

    assert len(out) == len(df)
    assert "true_anomaly" not in out.columns
    assert "anomaly_id" not in out.columns
