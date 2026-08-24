import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TSAnomalyConfig:
    date_col: str = "date"
    value_col: str = "daily_trip_count"
    seed: Optional[int] = None
    zero_load_count: int = 2
    spike_day_count: int = 3
    label_drift_count: int = 1
    spike_factor_range: Tuple[float, float] = (1.5, 3.0)
    drift_window_range: Tuple[int, int] = (10, 30)
    drift_strength_range: Tuple[float, float] = (0.3, 0.7)
    min_rows: int = 30
    clamp_non_negative: bool = True


def normalize_timeseries(df, cfg):
    """Ensure we have a clean date-value time series with daily frequency."""
    if cfg.date_col not in df.columns or cfg.value_col not in df.columns:
        raise KeyError(
            f"Expected columns '{cfg.date_col}' and '{cfg.value_col}'. Found: {list(df.columns)}"
        )

    out = df.copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="coerce")
    out[cfg.value_col] = pd.to_numeric(out[cfg.value_col], errors="coerce")
    out = out.dropna(subset=[cfg.date_col])
    out[cfg.date_col] = out[cfg.date_col].dt.date
    out = (
        out.groupby(cfg.date_col, as_index=False)[cfg.value_col]
        .sum()
        .sort_values(cfg.date_col)
        .reset_index(drop=True)
    )
    out[cfg.value_col] = out[cfg.value_col].astype(float)
    return out


def add_anomaly_columns(df):
    """Add columns to track injected anomalies. Initially all set to non-anomalous."""
    df["true_anomaly"] = 0
    df["true_anomaly_type"] = "none"
    df["anomaly_id"] = -1
    return df


def finalize_timeseries(df, cfg):
    """Finalize the time series by sorting and clamping values."""
    df = df.sort_values(cfg.date_col)
    if cfg.clamp_non_negative:
        df[cfg.value_col] = df[cfg.value_col].clip(lower=0)
    df[cfg.value_col] = df[cfg.value_col].round()
    return df.reset_index(drop=True)


def free_indices(df_len, used):
    """Get list of indices that are not currently used for anomalies."""
    return [i for i in range(df_len) if i not in used]


def inject_zero_load(df, cfg, rng, used, anomaly_counter):
    """Inject zero load anomalies by setting the value to zero on random days."""
    for _ in range(cfg.zero_load_count):
        candidates = free_indices(len(df), used)
        if not candidates:
            break
        idx = rng.choice(candidates)
        df.loc[idx, cfg.value_col] = 0
        df.loc[idx, ["true_anomaly", "true_anomaly_type", "anomaly_id"]] = (
            1,
            "zero_load",
            anomaly_counter,
        )
        used.add(idx)
        anomaly_counter += 1
    return anomaly_counter


def inject_spike_day(df, cfg, rng, used, anomaly_counter):
    """Inject spike day anomalies by multiplying the value by a random factor on random days."""
    for _ in range(cfg.spike_day_count):
        candidates = free_indices(len(df), used)
        if not candidates:
            break
        idx = rng.choice(candidates)
        factor = rng.uniform(*cfg.spike_factor_range)
        df.loc[idx, cfg.value_col] *= factor
        df.loc[idx, ["true_anomaly", "true_anomaly_type", "anomaly_id"]] = (
            1,
            "spike_day",
            anomaly_counter,
        )
        used.add(idx)
        anomaly_counter += 1
    return anomaly_counter


def inject_label_drift(df, cfg, rng, used, anomaly_counter):
    """Inject label drift anomalies by applying a gradual multiplicative drift over a random window."""
    max_attempts = 50
    attempts = 0
    injected = 0

    while injected < cfg.label_drift_count and attempts < max_attempts:
        attempts += 1

        window = int(rng.integers(cfg.drift_window_range[0], cfg.drift_window_range[1] + 1))
        if window >= len(df):
            continue

        start = int(rng.integers(0, len(df) - window))
        idxs = list(range(start, start + window))

        if any(i in used for i in idxs):
            continue

        strength = rng.uniform(*cfg.drift_strength_range)
        x = np.linspace(-6, 6, window)
        sigmoid = 1 / (1 + np.exp(-x))
        drift_curve = 1 + sigmoid * strength

        df.loc[idxs, cfg.value_col] *= drift_curve
        df.loc[idxs, ["true_anomaly", "true_anomaly_type", "anomaly_id"]] = (
            1,
            "label_drift",
            anomaly_counter,
        )

        used.update(idxs)
        anomaly_counter += 1
        injected += 1

    return anomaly_counter


def inject_ts_anomalies(df_in, config=None):
    """Main function to inject anomalies into a time series DataFrame."""
    cfg = TSAnomalyConfig() if config is None else config
    df = normalize_timeseries(df_in, cfg)

    if len(df) < cfg.min_rows:
        return df

    df = add_anomaly_columns(df)
    rng = np.random.default_rng(cfg.seed)
    used_idx = set()
    counter = 0

    counter = inject_zero_load(df, cfg, rng, used_idx, counter)
    counter = inject_spike_day(df, cfg, rng, used_idx, counter)
    counter = inject_label_drift(df, cfg, rng, used_idx, counter)

    return finalize_timeseries(df, cfg)
