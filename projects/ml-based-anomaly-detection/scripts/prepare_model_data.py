import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    PROJECT_ROOT,
    AGGREGATED_DATA_MODEL_SOURCE,
    CALIBRATION_REFERENCE_DATA,
    MODEL_TEST_DATA,
    MODEL_TRAIN_DATA,
    RAW_DATA_2022,
)
from src.config_loader import load_config
from src.data.artifact_metadata import write_artifact_metadata
from src.data.data_loader import load_processed_data
from src.features.feature_builder import build_time_series_features
from src.features.stl_decomposer import build_stl_features


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


def _split_model_data(df: pd.DataFrame, split_cfg: dict, time_col: str):
    dates = pd.to_datetime(df[time_col])

    train = df[dates <= pd.to_datetime(split_cfg["train_end"])].copy()
    test_clean = df[
        (dates >= pd.to_datetime(split_cfg["test_start"]))
        & (dates <= pd.to_datetime(split_cfg["test_end"]))
    ].copy()

    if train.empty:
        raise ValueError("Model-train split is empty. Check data_split.model.")
    if test_clean.empty:
        raise ValueError("Model-test split is empty. Check data_split.model.")

    return train, test_clean


def _split_calibration_data(df: pd.DataFrame, split_cfg: dict, time_col: str):
    dates = pd.to_datetime(df[time_col])
    calibration = df[
        (dates >= pd.to_datetime(split_cfg["start"]))
        & (dates <= pd.to_datetime(split_cfg["end"]))
    ].copy()

    if calibration.empty:
        raise ValueError("Calibration split is empty. Check data_split.calibration.")

    return calibration


def main():
    print("\n" + "-" * 60)
    print("Prepare model datasets (2022)")
    print("-" * 60)

    cfg = load_config()
    time_col = cfg["time"]["name"]

    print(f"Loading aggregated model source data from {AGGREGATED_DATA_MODEL_SOURCE}")
    df = load_processed_data(AGGREGATED_DATA_MODEL_SOURCE)
    print(f"Loaded {len(df)} rows")

    print("\nStage 1: Feature engineering...")
    df_features = _engineer_features(df)

    print("\nStage 2: model train/test split...")
    train_2022, test_clean_2022 = _split_model_data(
        df_features, cfg["data_split"]["model"], time_col
    )

    print(f"Train rows: {len(train_2022)}")
    print(f"Saving to {MODEL_TRAIN_DATA}")
    train_2022.to_csv(MODEL_TRAIN_DATA, index=False)
    train_meta = write_artifact_metadata(
        MODEL_TRAIN_DATA,
        {
            "artifact_role": "model_train",
            "source_year": 2022,
            "source_raw_file": RAW_DATA_2022,
            "source_aggregated_file": AGGREGATED_DATA_MODEL_SOURCE,
            "split": cfg["data_split"]["model"],
            "row_count": len(train_2022),
            "columns": list(train_2022.columns),
        },
        PROJECT_ROOT,
    )
    print(f"Wrote metadata sidecar: {train_meta}")

    print(f"Test rows: {len(test_clean_2022)}")
    print(f"Saving to {MODEL_TEST_DATA}")
    test_clean_2022.to_csv(MODEL_TEST_DATA, index=False)
    test_meta = write_artifact_metadata(
        MODEL_TEST_DATA,
        {
            "artifact_role": "model_test",
            "source_year": 2022,
            "source_raw_file": RAW_DATA_2022,
            "source_aggregated_file": AGGREGATED_DATA_MODEL_SOURCE,
            "split": cfg["data_split"]["model"],
            "row_count": len(test_clean_2022),
            "columns": list(test_clean_2022.columns),
        },
        PROJECT_ROOT,
    )
    print(f"Wrote metadata sidecar: {test_meta}")

    print("\nStage 3: calibration reference split...")
    calibration_2022 = _split_calibration_data(
        df_features, cfg["data_split"]["calibration"], time_col
    )

    print(f"Calibration rows: {len(calibration_2022)}")
    print(f"Saving to {CALIBRATION_REFERENCE_DATA}")
    calibration_2022.to_csv(CALIBRATION_REFERENCE_DATA, index=False)
    calibration_meta = write_artifact_metadata(
        CALIBRATION_REFERENCE_DATA,
        {
            "artifact_role": "calibration_reference",
            "source_year": 2022,
            "source_raw_file": RAW_DATA_2022,
            "source_aggregated_file": AGGREGATED_DATA_MODEL_SOURCE,
            "split": cfg["data_split"]["calibration"],
            "row_count": len(calibration_2022),
            "columns": list(calibration_2022.columns),
        },
        PROJECT_ROOT,
    )
    print(f"Wrote metadata sidecar: {calibration_meta}")

    print("\n" + "-" * 60)
    print("Model data preparation completed successfully.")
    print("-" * 60)


if __name__ == "__main__":
    main()
