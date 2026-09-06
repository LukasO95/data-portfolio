# src/moe.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def create_expert_model():
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


def predict_with_hard_routing(X, router, expert_low, expert_mid, expert_high):
    """
    Vorhersage der Preise mit hartem Routing basierend auf dem Routing-Modell und den Experten-Modellen.
    """

    # Vorhersage der Preis-Segmente mit dem Routing-Modell
    predicted_segments = router.predict(X)

    # Initialisiere eine Serie für die Vorhersagen
    y_pred = pd.Series(index=X.index, dtype=float)

    # Vorhersagen basierend auf den vorhergesagten Segmenten
    y_pred.loc[predicted_segments == "low"] = np.expm1(
        expert_low.predict(X[predicted_segments == "low"])
    )
    y_pred.loc[predicted_segments == "mid"] = np.expm1(
        expert_mid.predict(X[predicted_segments == "mid"])
    )
    y_pred.loc[predicted_segments == "high"] = np.expm1(
        expert_high.predict(X[predicted_segments == "high"])
    )

    return y_pred
