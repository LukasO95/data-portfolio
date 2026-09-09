# Model Airbnb Pricing

## Comparative Regression and Mixture of Experts

This project studies Airbnb nightly-price estimation as a supervised regression problem. It establishes a comparable baseline across tree-based models, selects the strongest model on a common holdout set, and explores whether segment-specific experts can improve predictions for different price ranges.

## Project Objectives

- Build a reproducible pipeline for predicting an Airbnb listing's nightly price.
- Compare regression models under identical data splits and preprocessing.
- Evaluate whether log-transforming a skewed target improves predictive performance.
- Extend the selected baseline with a hard-routed Mixture-of-Experts approach for low-, mid-, and high-price listings.

## Data

- **Source:** [Inside Airbnb](https://insideairbnb.com)
- **Target:** `price`, the listing's nightly rate
- **Working sample:** 15,000 listings, created with a fixed random seed

Place the downloaded `listings.csv` in `data/raw/` before running the data-preparation script. The repository includes a processed sample for exploration, but the preparation step can be rerun from the raw source data.

### Data Preparation

The pipeline keeps the workflow deliberately focused on model comparison:

- Converts price strings to numeric values and filters listings outside the $10-$500 price range.
- Restricts `minimum_nights` to 1-30 and removes rows without a valid price.
- Imputes missing numeric values with medians and categorical values with `Unknown`.
- Derives one supporting feature, `amenity_count`, from the amenities field.
- Uses six numeric inputs (`minimum_nights`, `number_of_reviews`, `review_scores_rating`, `latitude`, `longitude`, and `amenity_count`) and two categorical inputs (`neighbourhood` and `room_type`).

## Comparative Regression

All baseline models use the same 80/20 train-test split (`random_state=42`) and a shared scikit-learn preprocessing pipeline: numeric inputs are standardized and categorical inputs are one-hot encoded. This makes the reported differences attributable to the modeling choices rather than different data splits or feature treatment.

The project compares an original-scale Random Forest with log-target variants of Random Forest and XGBoost. For log-target models, training uses `log1p(price)` and predictions are transformed back with `expm1(price)` before metrics are calculated on the original price scale.

| Model | Target scale | RMSE | MAE | R2 |
| --- | --- | ---: | ---: | ---: |
| Random Forest | Original | 65.56 | 47.08 | 0.386 |
| Random Forest | Log-transformed | **64.84** | **43.92** | **0.400** |
| XGBoost | Log-transformed | 65.49 | 44.96 | 0.387 |

**Selected baseline:** the log-transformed Random Forest, chosen by the lowest RMSE. In this experiment, handling the skewed target distribution produced a modest improvement over the original-scale model and the configured XGBoost run.

## Mixture of Experts Extension

The advanced modeling work investigates a hard-routed Mixture of Experts. A `RandomForestClassifier` first assigns a listing to a low-, mid-, or high-price segment. The router then directs the listing to the matching specialized Random Forest regressor, which produces the final price estimate on the original scale.

This design tests whether specialized regressors can capture patterns that a single global model misses across distinct price segments. The routing and expert-prediction logic is implemented in `src/moe.py`; the experiment is explored in the final notebook and is separate from the baseline training and evaluation scripts above.

## Workflow

1. [Explore the data](notebooks/01_exploratory_data_analysis.ipynb)
2. [Prepare features and baseline models](notebooks/02_feature_engineering_and_model.ipynb)
3. [Evaluate model predictions](notebooks/03_model_evaluation.ipynb)
4. [Explore segmented modeling and Mixture of Experts](notebooks/04_segmented_modeling_and_mixture_of_experts.ipynb)

## Repository Structure

```text
data/       Raw and processed listing data
models/     Saved model pipelines and prediction artifacts
notebooks/  Exploratory, modeling, evaluation, and MoE analyses
scripts/    Reproducible data preparation, training, and evaluation entry points
src/        Data, feature, training, evaluation, and MoE modules
```

## Run the Baseline Pipeline

Install the dependencies:

```bash
pip install -r requirements.txt
```

From the project root, prepare data, train the baseline models, and compare their holdout metrics:

```bash
PYTHONPATH=. python scripts/make_processed_data.py --sample_size 15000
PYTHONPATH=. python scripts/train_models.py
PYTHONPATH=. python scripts/evaluate_models.py
```

Training saves model pipelines and `y_true`/`y_pred` prediction files in `models/`.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, and Jupyter.

## Limitations

- Results are based on one fixed holdout split; cross-validation would provide a more stable estimate of generalization performance.
- The current comparison evaluates configured model variants rather than a documented hyperparameter-search procedure.
- Predictions depend on the geography and collection period of the selected Inside Airbnb dataset.