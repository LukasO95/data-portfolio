import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    PROJECT_ROOT,
    AGGREGATED_DATA_ANOMALY_SOURCE,
    AGGREGATED_DATA_MODEL_SOURCE,
    RAW_DATA_2022,
    RAW_DATA_2023,
)
from src.config_loader import load_config
from src.data.artifact_metadata import write_artifact_metadata
from src.data.data_aggregator import aggregate_daily_trip_count
from src.data.data_loader import load_raw_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate raw taxi trips into daily trip-count datasets by year."
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=[2022, 2023],
        required=True,
        help="Year of the raw input file to aggregate.",
    )
    return parser.parse_args()


def get_io_paths(year):
    if year == 2022:
        return RAW_DATA_2022, AGGREGATED_DATA_MODEL_SOURCE
    return RAW_DATA_2023, AGGREGATED_DATA_ANOMALY_SOURCE


def main():
    args = parse_args()
    cfg = load_config()
    raw_path, processed_path = get_io_paths(args.year)

    print("Loading raw data...")
    df = load_raw_data(raw_path)

    print("Aggregating daily trip counts...")
    daily_data = aggregate_daily_trip_count(df)

    print("Saving processed data...")
    daily_data.to_csv(processed_path, index=False)
    print(f"Saved {len(daily_data)} daily rows to {processed_path}")

    sidecar_path = write_artifact_metadata(
        processed_path,
        {
            "artifact_role": (
                "aggregated_model_source"
                if args.year == 2022
                else "aggregated_anomaly_source"
            ),
            "source_year": args.year,
            "source_raw_file": raw_path,
            "target_column": cfg["target"]["name"],
            "time_column": cfg["time"]["name"],
            "row_count": len(daily_data),
        },
        project_root=PROJECT_ROOT,
    )
    print(f"Wrote metadata sidecar: {sidecar_path}")


if __name__ == "__main__":
    main()
