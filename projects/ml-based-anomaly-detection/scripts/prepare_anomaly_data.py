import pandas as pd
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    RAW_DATA_2023,
    AGGREGATED_DATA_ANOMALY_SOURCE,
    ANOMALY_INJECTED_DATA,
)

from src.config_loader import load_config
from src.data.artifact_metadata import write_artifact_metadata
from src.data.data_loader import load_processed_data
from src.features.stl_decomposer import build_stl_features
from src.features.feature_builder import build_time_series_features
from src.data.data_anomaly_injector import inject_ts_anomalies, TSAnomalyConfig


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    rows_before = len(df)
    df = build_stl_features(df)
    df = build_time_series_features(df)
    rows_after = len(df)
    print(
        f"Dropped {rows_before - rows_after} rows with NaN "
        "(feature engineering lags/rolling windows)"
    )
    return df


def _split_anomaly_base(
    df: pd.DataFrame, split_cfg: dict, time_col: str
) -> pd.DataFrame:
    dates = pd.to_datetime(df[time_col])
    out = df[
        (dates >= pd.to_datetime(split_cfg["start"]))
        & (dates <= pd.to_datetime(split_cfg["end"]))
    ].copy()

    if out.empty:
        raise ValueError("Anomaly base split is empty. Check data_split.anomaly.")

    return out


def main():
    print("\n" + "-" * 60)
    print("Prepare anomaly datasets (2023)")
    print("-" * 60)

    cfg = load_config()
    target_col = cfg["target"]["name"]
    time_col = cfg["time"]["name"]

    print(
        f"Loading aggregated anomaly source data from {AGGREGATED_DATA_ANOMALY_SOURCE}"
    )
    df = load_processed_data(AGGREGATED_DATA_ANOMALY_SOURCE)
    print(f"Loaded {len(df)} rows")

    print("\nStage 1: Feature engineering...")
    df_features = _engineer_features(df)

    print("\nStage 2: Select anomaly injection window...")
    base_2023 = _split_anomaly_base(df_features, cfg["data_split"]["anomaly"], time_col)
    print(f"Anomaly base rows: {len(base_2023)}")

    feature_cols = [
        col for col in base_2023.columns if col not in [time_col, target_col]
    ]

    print("\nStage 3: Inject anomalies into 2023 base set...")
    anomaly_cfg = TSAnomalyConfig(seed=42, date_col=time_col, value_col=target_col)
    anomaly_df = inject_ts_anomalies(
        base_2023[[time_col, target_col]].copy(), config=anomaly_cfg
    )

    feature_df = base_2023[[time_col] + feature_cols].copy()
    anomaly_df[time_col] = pd.to_datetime(anomaly_df[time_col]).dt.date
    feature_df[time_col] = pd.to_datetime(feature_df[time_col]).dt.date

    noisy_2023 = anomaly_df.merge(feature_df, on=time_col, how="left")
    noisy_2023["is_injected_anomaly"] = (
        noisy_2023["true_anomaly_type"] != "none"
    ).astype(int)

    anomaly_cols = [
        "true_anomaly",
        "true_anomaly_type",
        "anomaly_id",
        "is_injected_anomaly",
    ]
    final_cols = [time_col, target_col] + feature_cols + anomaly_cols
    noisy_2023 = noisy_2023[final_cols]

    injected_count = int(noisy_2023["is_injected_anomaly"].sum())
    print(f"Injected {injected_count} anomalies")
    print(f"Saving to {ANOMALY_INJECTED_DATA}")
    noisy_2023.to_csv(ANOMALY_INJECTED_DATA, index=False)

    anomaly_meta = write_artifact_metadata(
        ANOMALY_INJECTED_DATA,
        {
            "artifact_role": "anomaly_injected",
            "source_year": 2023,
            "source_raw_file": RAW_DATA_2023,
            "source_aggregated_file": AGGREGATED_DATA_ANOMALY_SOURCE,
            "split": cfg["data_split"]["anomaly"],
            "row_count": len(noisy_2023),
            "injected_anomaly_count": injected_count,
            "anomaly_config": asdict(anomaly_cfg),
            "columns": list(noisy_2023.columns),
        },
    )
    print(f"Wrote metadata sidecar: {anomaly_meta}")

    print("\n" + "-" * 60)
    print("Anomaly data preparation completed successfully.")
    print("-" * 60)


if __name__ == "__main__":
    main()
