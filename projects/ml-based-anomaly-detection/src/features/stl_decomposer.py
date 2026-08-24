import pandas as pd
from statsmodels.tsa.seasonal import STL
from src.config_loader import load_config


def build_stl_features(
    df,
    date_col="date",
    value_col="daily_trip_count",
):
    """Build STL decomposition features based on config.yaml"""
    config = load_config()
    stl_config = config["features"]["stl"]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    df = df.asfreq("D")
    df[value_col] = df[value_col].interpolate()

    stl = STL(df[value_col], period=stl_config["period"], robust=True)
    result = stl.fit()

    # Add only the configured components
    for component in stl_config["components"]:
        if component == "trend":
            df["trend"] = result.trend
        elif component == "seasonal":
            df["seasonal"] = result.seasonal
        elif component == "residual":
            df["residual"] = result.resid

    return df.reset_index()
