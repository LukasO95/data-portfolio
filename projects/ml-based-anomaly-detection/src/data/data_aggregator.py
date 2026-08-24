def aggregate_daily_trip_count(df):
    """Aggregate daily trip count."""
    df["date"] = df["tpep_pickup_datetime"].dt.date

    daily = (
        df.groupby("date", dropna=False)   # Include NaT as a separate group
          .size()
          .reset_index(name="daily_trip_count")
          .sort_values("date")
    )

    return daily
