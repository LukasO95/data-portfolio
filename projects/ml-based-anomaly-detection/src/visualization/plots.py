from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _get_threshold_value(thresholds, quantile):
    q_key = str(quantile)
    if q_key in thresholds:
        return float(thresholds[q_key]["value"])
    if quantile in thresholds:
        return float(thresholds[quantile]["value"])
    raise KeyError(f"Threshold for quantile {quantile} not found")


def _prepare_dates(df):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


def plot_injected_anomalies(df, output_path=None, show=True):
    df = _prepare_dates(df)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df["date"],
        df["daily_trip_count"],
        label="Daily Trips",
        color="steelblue",
        linewidth=2,
    )

    for anomaly_type, color, label, size in [
        ("zero_load", "red", "Zero Load", 80),
        ("spike_day", "orange", "Spike Day", 80),
        ("label_drift", "purple", "Label Drift", 40),
    ]:
        subset = df[df["true_anomaly_type"] == anomaly_type]
        if not subset.empty:
            ax.scatter(
                subset["date"],
                subset["daily_trip_count"],
                color=color,
                label=label,
                s=size,
                zorder=5,
            )

    ax.set_title("Test Data with Injected Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Trip Count")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_detection_results(df_results, thresholds, output_path=None, show=True):
    df = _prepare_dates(df_results)
    df = df.set_index("date")

    required = {"actual", "predicted", "score", "is_anomaly", "true_anomaly"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for plotting: {sorted(missing)}")

    q90 = _get_threshold_value(thresholds, 0.90)
    q95 = _get_threshold_value(thresholds, 0.95)
    q99 = _get_threshold_value(thresholds, 0.99)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(df.index, df["actual"], label="Actual", linewidth=2)
    axes[0].plot(
        df.index, df["predicted"], linestyle="--", label="Predicted", linewidth=2
    )

    tp_mask = (df["true_anomaly"] == 1) & (df["is_anomaly"])
    fp_mask = (df["true_anomaly"] == 0) & (df["is_anomaly"])
    fn_mask = (df["true_anomaly"] == 1) & (~df["is_anomaly"])

    tp_points = df[tp_mask]
    fp_points = df[fp_mask]
    fn_points = df[fn_mask]

    if not tp_points.empty:
        axes[0].scatter(
            tp_points.index,
            tp_points["actual"],
            color="green",
            marker="o",
            s=80,
            label="True Positive",
            zorder=3,
        )
    if not fp_points.empty:
        axes[0].scatter(
            fp_points.index,
            fp_points["actual"],
            color="orange",
            marker="x",
            s=80,
            label="False Positive",
            zorder=3,
        )
    if not fn_points.empty:
        axes[0].scatter(
            fn_points.index,
            fn_points["actual"],
            color="red",
            marker="^",
            s=80,
            label="False Negative",
            zorder=3,
        )

    axes[0].set_title("Detected Data Pipeline Anomalies")
    axes[0].set_ylabel("Daily Trip Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df["score"], label="Anomaly Score", linewidth=2)
    axes[1].axhline(q90, linestyle="--", color="blue", label="Q90")
    axes[1].axhline(q95, linestyle="--", color="orange", label="Q95")
    axes[1].axhline(q99, linestyle="--", color="red", label="Q99")
    axes[1].set_title("Anomaly Score with Thresholds")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
