# src/config.py
from pathlib import Path

# Projekt-Root: .../airbnb-price-prediction
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Datenverzeichnisse
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_LISTINGS_PATH = RAW_DATA_DIR / "listings.csv"
PROCESSED_SAMPLE_PATH = PROCESSED_DATA_DIR / "sample_listings.csv"

# Modellverzeichnis
MODELS_DIR = PROJECT_ROOT / "models"

# RandomForest (ohne Log)
RF_MODEL_PATH = MODELS_DIR / "rf_pipeline.pkl"
RF_PREDICTIONS_PATH = MODELS_DIR / "predictions_rf.csv"

# RandomForest (Log-Target)
RF_LOG_MODEL_PATH = MODELS_DIR / "rf_log_pipeline.pkl"
RF_LOG_PREDICTIONS_PATH = MODELS_DIR / "predictions_rf_log.csv"

# XGBoost (Log-Target)
XGB_LOG_MODEL_PATH = MODELS_DIR / "xgb_log_pipeline.pkl"
XGB_LOG_PREDICTIONS_PATH = MODELS_DIR / "predictions_xgb_log.csv"