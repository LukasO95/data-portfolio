from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_DATA_2022 = RAW_DIR / "2022_Yellow_Taxi_Trip_Data.csv"
RAW_DATA_2023 = RAW_DIR / "2023_Yellow_Taxi_Trip_Data.csv"

PROCESSED_DIR = DATA_DIR / "processed"

# Source-specific aggregates are persisted for traceability to raw inputs
AGGREGATED_DATA_MODEL_SOURCE = PROCESSED_DIR / "daily_trip_count_2022.csv"
AGGREGATED_DATA_ANOMALY_SOURCE = PROCESSED_DIR / "daily_trip_count_2023.csv"

# Canonical role-based datasets
MODEL_TRAIN_DATA = PROCESSED_DIR / "model_train.csv"
MODEL_TEST_DATA = PROCESSED_DIR / "model_test.csv"
CALIBRATION_REFERENCE_DATA = PROCESSED_DIR / "calibration_reference.csv"
ANOMALY_INJECTED_DATA = PROCESSED_DIR / "anomaly_injected.csv"

# Model paths
MODEL_DIR = PROJECT_ROOT / "models"
TUNED_MODEL_PATH = MODEL_DIR / "lgbm_tuned.pkl"

# Log paths
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_LOG = LOG_DIR / "results.json"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
INJECTED_ANOMALIES = RESULTS_DIR / "injected_anomalies.png"
ANOMALY_RESULTS = RESULTS_DIR / "anomaly_results.csv"
ANOMALY_THRESHOLDS = RESULTS_DIR / "thresholds.json"
ANOMALY_PLOT = RESULTS_DIR / "anomaly_timeseries.png"
ANOMALY_SCORE_PLOT = RESULTS_DIR / "anomaly_score.png"
ANOMALY_EVAL_SUMMARY = RESULTS_DIR / "evaluation_summary.json"
ANOMALY_DETECTION_RESULTS = RESULTS_DIR / "anomaly_detection_results.png"
