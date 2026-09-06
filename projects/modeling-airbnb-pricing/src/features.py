# src/features.py
import pandas as pd


def count_amenities(amenities):
    """
    Zählt die Anzahl der Ausstattungsmerkmale in einem String wie
    "{Wifi, Kitchen, Heating}" -> 3.

    Rückgabe: 0, wenn der String leer oder NaN ist.
    """
    if pd.isna(amenities):
        return 0

    cleaned = str(amenities).strip("{}").strip()
    if not cleaned:
        return 0

    return sum(1 for a in cleaned.split(",") if a.strip())


def add_amenity_count(df, drop_original = True):
    """
    Fügt dem DataFrame basierend der Spalte 'amenities' 
    eine Spalte 'amenity_count' hinzu.
    """
    if "amenities" not in df.columns:
        raise KeyError("Spalte 'amenities' nicht im DataFrame vorhanden.")

    df_out = df.copy()
    df_out["amenity_count"] = df_out["amenities"].apply(count_amenities)

    if drop_original:
        df_out = df_out.drop(columns=["amenities"])

    return df_out
