# src/evaluate.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

from .config import (
    PROJECT_ROOT,
    RF_MODEL_PATH, RF_PREDICTIONS_PATH,
    RF_LOG_MODEL_PATH, RF_LOG_PREDICTIONS_PATH,
    XGB_LOG_MODEL_PATH, XGB_LOG_PREDICTIONS_PATH,
)

def display_path(path):
    """Gibt den Pfad relativ zum Projektstammverzeichnis zurück."""
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve())


# Mapping
PREDICTION_PATHS = {
    "rf": RF_PREDICTIONS_PATH,
    "rf_log": RF_LOG_PREDICTIONS_PATH,
    "xgb_log": XGB_LOG_PREDICTIONS_PATH,
}

MODEL_PATHS = {
    "rf": RF_MODEL_PATH,
    "rf_log": RF_LOG_MODEL_PATH,
    "xgb_log": XGB_LOG_MODEL_PATH,
}


def load_predictions(model_key):
    """Lädt die gespeicherten Vorhersagen (y_true, y_pred) für ein Modell."""
    path = PREDICTION_PATHS[model_key]
    df = pd.read_csv(path)
    return df


def compute_metrics(y_true, y_pred):
    """Berechnet RMSE, MAE und R²."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_models():
    """
    Lädt für alle Modelle die Vorhersagen, berechnet Metriken
    und bestimmt das beste Modell (kleinster RMSE).

    Rückgabe:
        results: dict[modell_name -> metriken]
        best_model: name des besten Modells (z.B. 'xgb_log')
    """
    results = {}
    for key, path in PREDICTION_PATHS.items():
        try:
            df = pd.read_csv(path)
            print(f"Vorhersagen geladen: {key} ({display_path(path)}) – {len(df)} Zeilen")
        except FileNotFoundError:
            print(f"Datei nicht gefunden: {path}")
            continue

        metrics = compute_metrics(df["y_true"], df["y_pred"])
        results[key] = metrics

    if not results:
        print("Keine Vorhersagedateien gefunden.")
        return {}, None

    # Bestes Modell anhand RMSE auswählen
    best_model = None
    best_rmse = None
    for key, m in results.items():
        if best_rmse is None or m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_model = key

    print(f"\nBestes Modell (nach RMSE): {best_model} (RMSE={best_rmse:.2f})")
    return results, best_model


def load_model(model_key):
    """Lädt ein gespeichertes Modell (Pipeline) anhand des Kürzels."""
    path = MODEL_PATHS[model_key]
    try:
        model = joblib.load(path)
        print(f"Modell geladen: {model_key} ({display_path(path)})")
        return model
    except FileNotFoundError:
        print(f"Modell nicht gefunden: {display_path(path)}")
        return None


def load_best_model():
    """
    Bewertet alle Modelle und lädt das beste Modell.

    Rückgabe:
        model: geladene Pipeline
        best_model_key: z.B. 'xgb_log'
        best_metrics: Metriken des besten Modells
    """
    results, best_key = evaluate_models()
    if best_key is None:
        raise RuntimeError("Kein bestes Modell bestimmbar (keine Ergebnisse).")

    model = load_model(best_key)
    best_metrics = results[best_key]
    return model, best_key, best_metrics