import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from config import (
    MODEL_TRAIN_DATA,
    MODEL_TEST_DATA,
    TUNED_MODEL_PATH,
    LOG_DIR,
    RESULTS_LOG,
)
from src.config_loader import load_config
from src.data.data_loader import load_processed_data
from src.features.utils import get_feature_columns
from src.models.io import load_model, model_exists, save_model
from src.models.lgbm_tuner import tune_lgbm_model


# Parse command line arguments
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--retrain",
        action="store_true",
        help="Force re-tuning even if a saved tuned model exists",
    )
    p.add_argument("--plot", action="store_true", help="Plot test forecast vs actual")
    p.add_argument("--n-iter", type=int, default=25)
    p.add_argument("--n-splits", type=int, default=4)
    return p.parse_args()


# Evaluate on clean test set
def evaluate_on_clean(test, model, feature_cols, target_col):
    X_test = test[feature_cols]
    y_test = test[target_col]

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)

    return {"mae": mae, "mape": mape, "y_true": y_test, "y_pred": preds}


# Training and evaluation flow
def main():
    args = parse_args()
    # Initialize for logging
    cv_mae = None
    best_params = None

    cfg = load_config()
    time_col = cfg["time"]["name"]
    target_col = cfg["target"]["name"]

    model_train = load_processed_data(MODEL_TRAIN_DATA)
    model_test = load_processed_data(MODEL_TEST_DATA)

    if model_train.empty:
        raise ValueError(f"{MODEL_TRAIN_DATA} is empty. Run prepare_model_data first.")
    if model_test.empty:
        raise ValueError(f"{MODEL_TEST_DATA} is empty. Run prepare_model_data first.")

    feature_cols = get_feature_columns()
    X_train = model_train[feature_cols]
    y_train = model_train[target_col]

    print(f"Train rows: {len(model_train)}, Test rows: {len(model_test)}")
    print("Feature columns:", feature_cols)

    if (not args.retrain) and model_exists(TUNED_MODEL_PATH):
        print("Loading existing tuned model:", TUNED_MODEL_PATH)
        model = load_model(TUNED_MODEL_PATH)
    else:
        model, best_params, cv_mae = tune_lgbm_model(
            X_train, y_train, n_splits=args.n_splits, n_iter=args.n_iter
        )
        print("Tuned CV MAE:", cv_mae)
        print("Best params:", best_params)
        save_model(model, TUNED_MODEL_PATH)

    eval_res = evaluate_on_clean(model_test, model, feature_cols, target_col)
    print(f"Model path: {TUNED_MODEL_PATH}")
    print(f"Model-test MAE: {eval_res['mae']:.2f}, MAPE: {eval_res['mape']:.4f}")

    # Log results to JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),  # All CLI args
        "train_rows": len(model_train),
        "test_rows": len(model_test),
        "feature_cols": feature_cols,
        "model_path": str(TUNED_MODEL_PATH),
        "test_mae": eval_res["mae"],
        "test_mape": eval_res["mape"],
        "cv_mae": cv_mae,
        "best_params": best_params,
    }

    # Save to file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_LOG
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)  # default=str for datetime

    print(f"Results logged to {results_file}")

    if args.plot:
        # Forecast plot
        plt.figure(figsize=(12, 5))
        plt.plot(model_test[time_col], eval_res["y_true"], label="actual")
        plt.plot(model_test[time_col], eval_res["y_pred"], label="forecast")
        plt.title("Model test: forecast vs actual")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Residual plot
        residuals = eval_res["y_true"] - eval_res["y_pred"]
        plt.figure(figsize=(12, 5))
        plt.plot(model_test[time_col], residuals)
        plt.title("Residuals over time")
        plt.axhline(y=0, color="red", linestyle="--", label="Zero line")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
