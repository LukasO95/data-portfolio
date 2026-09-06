# src/train_model.py
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

from .config import (
    RF_MODEL_PATH,
    RF_PREDICTIONS_PATH,
    RF_LOG_MODEL_PATH,
    RF_LOG_PREDICTIONS_PATH,
    XGB_LOG_MODEL_PATH,
    XGB_LOG_PREDICTIONS_PATH,
)
from .data import load_sample_listings
from .features import add_amenity_count


TARGET_COL = "price"

NUMERIC_FEATURES = [
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "latitude",
    "longitude",
    "amenity_count",
]

CATEGORICAL_FEATURES = [
    "neighbourhood",
    "room_type",
]

# Preprocessor erstellen (gemeinsame Funktion für alle Modelle)

def create_preprocessor():
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def create_model_pipeline(model):
    """Erstellt eine Pipeline aus Preprocessing und dem übergebenem Modell."""
    pipeline = Pipeline(
        [
            ("preprocess", create_preprocessor()),
            ("model", model),
        ]
    )
    return pipeline


def get_train_test_data(test_size=0.2):
    df = load_sample_listings()
    df = add_amenity_count(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return train_test_split(X, y, test_size=test_size, random_state=42)


def evaluate_and_save(
    pipeline, X_test, y_test, model_path, predictions_path, log_target=False
):
    # Vorhersagen erzeugen
    if log_target:
        y_pred = np.expm1(pipeline.predict(X_test))  # zurück-transformieren
    else:
        y_pred = pipeline.predict(X_test)

    # Metriken berechnen
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Speichern
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    pred_df.to_csv(predictions_path, index=False)

    return {"rmse": rmse, "mae": mae, "r2": r2}


# Modell 1: RandomForest (Original-Target)

def train_rf():
    print("\nTraining: RandomForest (Original-Target)")

    X_train, X_test, y_train, y_test = get_train_test_data()

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipeline = create_model_pipeline(model)

    pipeline.fit(X_train, y_train)

    metrics = evaluate_and_save(
        pipeline, X_test, y_test, RF_MODEL_PATH, RF_PREDICTIONS_PATH, log_target=False
    )

    print("Metriken RF:", metrics)
    return metrics


# Modell 2: RandomForest (log-Target)

def train_rf_log():
    print("\nTraining: RandomForest (log-Target)")

    X_train, X_test, y_train, y_test = get_train_test_data()

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipeline = create_model_pipeline(model)

    y_train_log = np.log1p(y_train)
    pipeline.fit(X_train, y_train_log)

    metrics = evaluate_and_save(
        pipeline,
        X_test,
        y_test,
        RF_LOG_MODEL_PATH,
        RF_LOG_PREDICTIONS_PATH,
        log_target=True,
    )

    print("Metriken RF-Log:", metrics)
    return metrics


# Modell 3: XGBRegressor (log-Target)

def train_xgb_log():
    print("\nTraining: XGBRegressor (log-Target)")

    X_train, X_test, y_train, y_test = get_train_test_data()
    
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    pipeline = create_model_pipeline(model)

    y_train_log = np.log1p(y_train)
    pipeline.fit(X_train, y_train_log)

    metrics = evaluate_and_save(
        pipeline,
        X_test,
        y_test,
        XGB_LOG_MODEL_PATH,
        XGB_LOG_PREDICTIONS_PATH,
        log_target=True,
    )

    print("Metriken XGB-Log:", metrics)
    return metrics


# Alle Modelle trainieren

def train_all_models():
    return {
        "rf": train_rf(),
        "rf_log": train_rf_log(),
        "xgb_log": train_xgb_log(),
    }