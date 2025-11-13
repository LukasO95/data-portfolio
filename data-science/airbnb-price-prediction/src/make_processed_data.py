import os
import pandas as pd

RAW_PATH = "data/raw/listings.csv"
OUTPUT_PATH = "data/processed/sample_listings.csv"


def load_raw_data() -> pd.DataFrame:
    if os.path.exists(RAW_PATH):
        df_raw = pd.read_csv(RAW_PATH, low_memory=False)
    else:
        raise FileNotFoundError("Bitte listings.csv nach data/raw/ legen.")

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


def main():
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
    df_small = select_and_sample(df, sample_size=3000)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_small.to_csv(OUTPUT_PATH, index=False)
    print(f"Fertig ✅ Gespeichert unter {OUTPUT_PATH} ({len(df_small)} Zeilen)")


if __name__ == "__main__":
    main()
