def check_duplicates(daily_df, column="date"):
    """Check if a data point occurs multiple times in the dataset."""
    
    duplicates = daily_df[column].value_counts()
    duplicated_vals = duplicates[duplicates > 1]

    return {
        "has_duplicates": len(duplicated_vals) > 0,
        "duplicate_count": len(duplicated_vals),
        "duplicated_values": duplicated_vals.index.tolist(),
    }
