# Data Science & ML Portfolio

A collection of applied machine learning projects spanning LLM-assisted workflow automation, time-series monitoring, and predictive modeling. Each project pairs a concrete business problem with reproducible code, explicit evaluation boundaries, and documented trade-offs.

## Projects at a Glance

| Project | Problem | Primary Approach | Key Insight |
| --- | --- | --- | --- |
| [Finance Inquiry Intent Classification](projects/finance-inquiry-intent-classification/README.md) | Route unstructured financial-service inquiries to the appropriate CRM queue | LLM classification with PII masking, schema validation, and deterministic fallback routing | Safe fallback matters when LLM output is uncertain |
| [ML-Based Anomaly Detection](projects/ml-based-anomaly-detection/README.md) | Detect quiet data-pipeline failures in daily taxi-trip volumes | STL features, tuned LightGBM forecasting, and residual-based anomaly scoring | Different anomaly types require different alerting policies |
| [Model Airbnb Pricing](projects/modeling-airbnb-pricing/README.md) | Estimate Airbnb nightly prices | Comparative tree-based regression and a Mixture-of-Experts extension | Performance varies across price segments, motivating segment-aware modeling |

## Finance Inquiry Intent Classification

**LLM-assisted CRM routing for financial services.** This project classifies free-text customer inquiries as product, service, or unclear, then routes each inquiry to sales, support, or manual clearing. The emphasis is not only classification: it is a defensive workflow for using an LLM inside an operational process.

- Masks personally identifiable information before classification.
- Normalizes LLM output, validates it against a JSON Schema, and applies deterministic routing rules.
- Routes invalid, low-confidence, or failed requests to manual clearing rather than forcing a prediction.
- Includes rate limiting, classifier-version tracking, and tests for PII masking, normalization, routing, and the mock pipeline.

The included mock provider makes the architecture runnable without external APIs. It is explicitly demonstration-only and should not be interpreted as a measure of real classification quality.

[Explore the project](projects/finance-inquiry-intent-classification/README.md)

## ML-Based Anomaly Detection

**Forecast-based monitoring for data-pipeline failures.** Using NYC Yellow Taxi trip records aggregated to daily volumes, this project detects zero-load outages, duplicate-record spikes, and gradual label drift that can silently degrade downstream data products.

- Builds STL decomposition, lag, rolling-window, and calendar features.
- Tunes a LightGBM forecaster with `RandomizedSearchCV` and `TimeSeriesSplit`.
- Learns residual thresholds from a separate clean calibration dataset, then assigns anomaly severity.
- Injects known anomalies into a held-out source period to create evaluation ground truth where labeled operational failures are unavailable.

Against the injected anomalies, the detector achieved **precision 0.93**, **recall 0.72**, and **F1 0.81**. Zero-load outages and spike days were detected with recall of 1.00; gradual label drift was the most difficult failure mode, with recall of 0.62. These results are based on synthetic anomalies and are presented as an evaluation proxy, not live operational performance.

[Explore the project](projects/ml-based-anomaly-detection/README.md)

## Model Airbnb Pricing

**Comparative regression with a segment-aware extension.** This project frames nightly Airbnb price estimation as a supervised regression task, comparing models on an identical holdout split before exploring specialized models for distinct price ranges.

- Compares original-scale Random Forest, log-target Random Forest, and log-target XGBoost pipelines.
- Applies shared preprocessing and an 80/20 fixed train-test split so model results are directly comparable.
- Selects the log-target Random Forest as the strongest baseline: **RMSE 64.84**, **MAE 43.92**, and **R2 0.400**.
- Explores a hard-routed Mixture of Experts in which a Random Forest classifier assigns listings to low-, mid-, or high-price segments before a specialized regressor predicts the final price.

The Mixture-of-Experts work is an exploratory extension and is separate from the reproducible baseline training and evaluation scripts. Baseline results use one fixed holdout split; cross-validation and a direct MoE benchmark are appropriate next steps.

[Explore the project](projects/modeling-airbnb-pricing/README.md)

## Engineering Practices

Across these projects, the code favors modular source packages, runnable scripts, and notebooks that make the analysis sequence inspectable. The portfolio also emphasizes evaluation discipline: time-aware validation and separate threshold calibration for anomaly detection, comparable preprocessing and splits for regression, and explicit safeguards around non-deterministic LLM outputs. Project READMEs document assumptions and limitations alongside results.

## Explore Further

Each project README contains project-specific setup, dependencies, commands, source data expectations, and detailed methodology:

- [Finance Inquiry Intent Classification](projects/finance-inquiry-intent-classification/README.md)
- [ML-Based Anomaly Detection](projects/ml-based-anomaly-detection/README.md)
- [Model Airbnb Pricing](projects/modeling-airbnb-pricing/README.md)
