import pandas as pd


def load_raw_data(file_path):
    """Load raw data."""
    df = pd.read_csv(
        file_path,
        usecols=["tpep_pickup_datetime", "fare_amount", "total_amount", "trip_distance"],
        low_memory=False
    )

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="raise"
    )

    return df


def load_processed_data(path):
    """Load processed data."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    df["date"] = df["date"].dt.date
    
    return df