import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ANOMALY_INJECTED_DATA, INJECTED_ANOMALIES
from src.visualization.plots import plot_injected_anomalies


def main():
    df = pd.read_csv(ANOMALY_INJECTED_DATA, parse_dates=["date"])
    plot_injected_anomalies(df, output_path=INJECTED_ANOMALIES, show=False)


if __name__ == "__main__":
    main()
