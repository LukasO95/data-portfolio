# src/airbnb_price_prediction/config.py
from pathlib import Path

# Projekt-Root: .../airbnb-price-prediction
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_LISTINGS_PATH = RAW_DATA_DIR / "listings.csv"
PROCESSED_SAMPLE_PATH = PROCESSED_DATA_DIR / "sample_listings.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RF_MODEL_PATH = MODELS_DIR / "rf_pipeline.pkl"
RF_PREDICTIONS_PATH = MODELS_DIR / "predictions_rf.csv"