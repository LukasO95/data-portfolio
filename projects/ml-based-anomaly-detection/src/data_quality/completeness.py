import pandas as pd


def check_missing_days(daily_df):
    """Calculate the completeness of a specified column in a DataFrame."""

    expected_dates = pd.date_range(
        start=daily_df["date"].min(), end=daily_df["date"].max(), freq="D"
    )

    observed_dates = pd.to_datetime(daily_df["date"].unique())
    missing_dates = expected_dates.difference(observed_dates)

    return {
        "missing_days_count": len(missing_dates),
        "missing_days:": missing_dates.date.tolist(),
    }
