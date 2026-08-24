import pandas as pd
from src.config_loader import load_config


def build_time_series_features(df, date_col="date", value_col="daily_trip_count"):
    """Build time series features based on config.yaml"""
    config = load_config()
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Build lags
    for lag in config['features']['lags']:
        df[lag['name']] = df[value_col].shift(lag['shift'])

    # Build rolling stats
    for roll in config['features']['rolling']:
        shifted_col = df[value_col].shift(1)
        if roll['func'] == 'median':
            df[roll['name']] = shifted_col.rolling(roll['window']).median()
        elif roll['func'] == 'std':
            df[roll['name']] = shifted_col.rolling(roll['window']).std()

    # Build calendar features
    for cal in config['features']['calendar']:
        if cal['expr'] == 'weekday':
            df[cal['name']] = df[date_col].dt.weekday
        elif cal['expr'] == 'day':
            df[cal['name']] = df[date_col].dt.day
        elif cal['expr'] == 'month':
            df[cal['name']] = df[date_col].dt.month

    df = df.dropna().reset_index(drop=True)
    return df
