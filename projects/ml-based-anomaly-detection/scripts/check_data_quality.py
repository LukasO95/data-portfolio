import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ANOMALY_INJECTED_DATA,
    CALIBRATION_REFERENCE_DATA,
    MODEL_TEST_DATA,
    MODEL_TRAIN_DATA,
)
from src.data.data_loader import load_processed_data
from src.data_quality.completeness import check_missing_days
from src.data_quality.duplicates import check_duplicates
from src.data_quality.value_ranges import check_non_negative


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run data quality checks on daily time series."
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "model_train",
            "model_test",
            "calibration_reference",
            "anomaly_injected",
        ],
        default="model_test",
        help="Select which prepared dataset to validate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_paths = {
        "model_train": MODEL_TRAIN_DATA,
        "model_test": MODEL_TEST_DATA,
        "calibration_reference": CALIBRATION_REFERENCE_DATA,
        "anomaly_injected": ANOMALY_INJECTED_DATA,
    }
    data_path = dataset_paths[args.dataset]
    print(f"Loading dataset: {data_path}")
    daily = load_processed_data(data_path)

    print("Running completeness check...")
    completeness_result = check_missing_days(daily)
    print(completeness_result)

    print("Running duplicates check...")
    duplicates_result = check_duplicates(daily)
    print(duplicates_result)

    print("Running value range check...")
    value_ranges_result = check_non_negative(daily)
    print(value_ranges_result)


if __name__ == "__main__":
    main()
