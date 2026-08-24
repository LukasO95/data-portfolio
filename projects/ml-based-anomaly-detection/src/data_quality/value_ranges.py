def check_non_negative(daily_df, column="daily_trip_count"):
    """Check if a column contains only non-negative values."""

    negative_mask = daily_df[column] < 0
    negative_indices = daily_df[negative_mask].index.tolist()
    
    return {
        "is_non_negative": negative_mask.sum() == 0,
        "negative_count": negative_mask.sum(),
        "negative_indices": negative_indices
    }