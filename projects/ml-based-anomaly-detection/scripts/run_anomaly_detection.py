import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import (
    TUNED_MODEL_PATH,
    CALIBRATION_REFERENCE_DATA,
    ANOMALY_INJECTED_DATA,
    ANOMALY_RESULTS,
    ANOMALY_THRESHOLDS,
)
from src.data.artifact_metadata import write_artifact_metadata
from src.anomaly.utils import print_thresholds, save_thresholds
from src.anomaly.anomaly_detector import AnomalyDetector
from src.models.io import load_model
from src.features.utils import get_feature_columns
from src.config_loader import load_anomaly_detection_policy, load_config


def main():
    cfg = load_config()
    target_col = cfg["target"]["name"]
    time_col = cfg["time"]["name"]
    anomaly_policy = load_anomaly_detection_policy(cfg)

    # Load data
    calibration_reference = pd.read_csv(
        CALIBRATION_REFERENCE_DATA, parse_dates=[time_col]
    )
    anomaly_injected = pd.read_csv(ANOMALY_INJECTED_DATA, parse_dates=[time_col])

    if calibration_reference.empty:
        raise ValueError(
            f"{CALIBRATION_REFERENCE_DATA} is empty. Run prepare_model_data first."
        )
    if anomaly_injected.empty:
        raise ValueError(
            f"{ANOMALY_INJECTED_DATA} is empty. Run prepare_anomaly_data first."
        )

    # Get feature columns
    feature_cols = get_feature_columns()

    # Extract calibration and test data
    X_calibration = calibration_reference[feature_cols]
    y_calibration = calibration_reference[target_col]

    X_test = anomaly_injected[feature_cols]
    y_test = anomaly_injected[target_col]

    # Load tuned model
    model = load_model(TUNED_MODEL_PATH)

    # Predictions
    y_calibration_pred = model.predict(X_calibration)
    y_test_pred = model.predict(X_test)

    # Fit detector on clean calibration residuals, then detect on noisy set
    detector = AnomalyDetector(
        quantiles=anomaly_policy["quantiles"],
        anomaly_levels=anomaly_policy["alert_levels"],
    )
    detector.fit(y_calibration, y_calibration_pred)

    thresholds = detector.get_thresholds()

    # Print thresholds
    print_thresholds(thresholds)

    # Save thresholds
    save_thresholds(thresholds, ANOMALY_THRESHOLDS)

    results = detector.detect(y_test, y_test_pred)

    # Output results
    df_results = anomaly_injected[[time_col] + feature_cols].copy()

    df_results["actual"] = y_test
    df_results["predicted"] = y_test_pred
    df_results["score"] = results["score"].values
    df_results["severity"] = results["severity"].values
    df_results["is_anomaly"] = results["is_anomaly"].values

    for col in [
        "true_anomaly",
        "true_anomaly_type",
        "anomaly_id",
        "is_injected_anomaly",
    ]:
        if col in anomaly_injected.columns:
            df_results[col] = anomaly_injected[col].values

    # Save results
    df_results.to_csv(ANOMALY_RESULTS, index=False)
    print(f"\nSaved anomaly results to {ANOMALY_RESULTS}.")

    sidecar_path = write_artifact_metadata(
        ANOMALY_RESULTS,
        {
            "artifact_role": "anomaly_detection_results",
            "calibration_dataset": CALIBRATION_REFERENCE_DATA,
            "anomaly_dataset": ANOMALY_INJECTED_DATA,
            "model_path": TUNED_MODEL_PATH,
            "quantiles": list(anomaly_policy["quantiles"]),
            "alert_levels": list(anomaly_policy["alert_levels"]),
            "row_count": int(len(df_results)),
            "columns": list(df_results.columns),
        },
    )
    print(f"Wrote metadata sidecar: {sidecar_path}")


if __name__ == "__main__":
    main()
