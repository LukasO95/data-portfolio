import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ANOMALY_PLOT, ANOMALY_RESULTS, ANOMALY_THRESHOLDS
from src.anomaly.utils import load_thresholds
from src.visualization.plots import plot_detection_results


def main():
    df = pd.read_csv(ANOMALY_RESULTS, parse_dates=["date"])
    thresholds = load_thresholds(ANOMALY_THRESHOLDS)

    plot_detection_results(
        df_results=df,
        thresholds=thresholds,
        output_path=ANOMALY_PLOT,
        show=False,
    )


if __name__ == "__main__":
    main()
