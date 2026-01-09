# src/airbnb_price_prediction/data.py
import os
import pandas as pd

from .config import RAW_LISTINGS_PATH, PROCESSED_SAMPLE_PATH


def load_raw_data() -> pd.DataFrame:
    if os.path.exists(RAW_LISTINGS_PATH):
        df_raw = pd.read_csv(RAW_LISTINGS_PATH, low_memory=False)
    else:
        raise FileNotFoundError(f"Bitte listings.csv nach {RAW_LISTINGS_PATH} legen.")

    cols = [
        "price",
        "neighbourhood_cleansed",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "review_scores_rating",
        "amenities",
        "latitude",
        "longitude",
    ]
    cols = [c for c in cols if c in df_raw.columns]
    df = df_raw[cols].copy()

    if "neighbourhood_cleansed" in df.columns:
        df = df.rename(columns={"neighbourhood_cleansed": "neighbourhood"})

    return df


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def remove_outliers_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["price"].notna()]
    df = df[df["price"].between(10, 500)]

    if "minimum_nights" in df.columns:
        df = df[df["minimum_nights"].between(1, 30)]

    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_cols = [
        c
        for c in [
            "minimum_nights",
            "number_of_reviews",
            "review_scores_rating",
            "latitude",
            "longitude",
        ]
        if c in df.columns
    ]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in ["room_type", "neighbourhood", "amenities"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


def select_and_sample(df: pd.DataFrame, sample_size: int = 3000) -> pd.DataFrame:
    cols = [
        "price",
        "neighbourhood",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "review_scores_rating",
        "amenities",
        "latitude",
        "longitude",
    ]
    cols = [c for c in cols if c in df.columns]
    df_small = df[cols].copy()

    if len(df_small) > sample_size:
        df_small = df_small.sample(n=sample_size, random_state=42)

    return df_small.reset_index(drop=True)


def make_processed_sample_listings(sample_size: int = 3000):
    print("1) Rohdaten laden ...")
    df = load_raw_data()
    print(df.shape)

    print("2) Preisspalte bereinigen ...")
    df = clean_price_column(df)

    print("3) Ausreißer & fehlende Werte entfernen ...")
    df = remove_outliers_and_nans(df)

    print("4) Fehlende Werte füllen ...")
    df = fill_missing(df)

    print("5) Stichprobe ziehen & speichern ...")
    df_small = select_and_sample(df, sample_size=sample_size)

    os.makedirs(os.path.dirname(PROCESSED_SAMPLE_PATH), exist_ok=True)
    df_small.to_csv(PROCESSED_SAMPLE_PATH, index=False)
    print(f"Fertig ✅ Gespeichert unter {PROCESSED_SAMPLE_PATH}")
    print(f"Stichprobengröße: {len(df_small)} Zeilen")

    return df_small


def load_sample_listings():
    """
    Lädt die verarbeiteten Sample-Daten (sample_listings.csv).

    Hinweis: Stelle sicher, dass vorher das Skript
    `python scripts/make_processed_data.py` ausgeführt wurde.
    """
    if not PROCESSED_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_SAMPLE_PATH} existiert nicht. "
            "Bitte zuerst `python scripts/make_processed_data.py` ausführen."
        )

    return pd.read_csv(PROCESSED_SAMPLE_PATH)
