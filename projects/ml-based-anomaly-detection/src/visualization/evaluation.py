def summarize_detection_results(df_results):
    required = {"true_anomaly", "is_anomaly"}
    missing = required - set(df_results.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df_results.copy()

    df["true_anomaly"] = df["true_anomaly"].astype(int)
    df["is_anomaly"] = df["is_anomaly"].astype(bool)

    tp = int(((df["is_anomaly"]) & (df["true_anomaly"] == 1)).sum())
    fp = int(((df["is_anomaly"]) & (df["true_anomaly"] == 0)).sum())
    fn = int(((~df["is_anomaly"]) & (df["true_anomaly"] == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    by_type = {}
    if "true_anomaly_type" in df.columns:
        for anomaly_type in sorted(
            df.loc[df["true_anomaly"] == 1, "true_anomaly_type"].dropna().unique()
        ):
            subset = df[df["true_anomaly_type"] == anomaly_type]
            tp_t = int(((subset["is_anomaly"]) & (subset["true_anomaly"] == 1)).sum())
            fp_t = int(((subset["is_anomaly"]) & (subset["true_anomaly"] == 0)).sum())
            fn_t = int(((~subset["is_anomaly"]) & (subset["true_anomaly"] == 1)).sum())

            precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0
            recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
            f1_t = (
                2 * precision_t * recall_t / (precision_t + recall_t)
                if (precision_t + recall_t)
                else 0.0
            )

            by_type[anomaly_type] = {
                "tp": tp_t,
                "fp": fp_t,
                "fn": fn_t,
                "precision": precision_t,
                "recall": recall_t,
                "f1": f1_t,
            }

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "by_type": by_type,
    }