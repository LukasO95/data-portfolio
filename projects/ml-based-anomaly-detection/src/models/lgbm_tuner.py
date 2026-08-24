from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def tune_lgbm_model(X_train, y_train, random_state=42, n_splits=4, n_iter=10):
    """
    Hyperparameter tuning for LightGBM using TimeSeriesSplit to avoid leakage.

    Returns:
        best_estimator_, best_params_, best_score_, cv_results_
    """
    model = LGBMRegressor(random_state=random_state, verbose=-1)

    # Time-aware CV (keeps order)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Parameter space (MVP-friendly, not too wide)
    param_distributions = {
        "n_estimators": [200, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 5, 8, 12],
        "min_child_samples": [10, 20, 40, 60],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
    }

    # Use MAE as optimization objective (negative because sklearn maximizes)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=random_state,
        n_jobs=1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = -search.best_score_  # convert back to positive MAE
    cv_results = search.cv_results_

    return best_model, best_params, best_score, cv_results
