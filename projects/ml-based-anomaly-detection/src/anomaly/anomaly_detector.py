import numpy as np
import pandas as pd


# Define the AnomalyDetector class
class AnomalyDetector:
    def __init__(self, quantiles=(0.90, 0.95, 0.99), anomaly_levels=None):
        self.quantiles = self._validate_quantiles(quantiles)
        self.label_map = {
            self.quantiles[0]: "low",
            self.quantiles[1]: "medium",
            self.quantiles[2]: "high",
        }

        if anomaly_levels is None:
            self.anomaly_levels = ["medium", "high"]
        else:
            self.anomaly_levels = list(anomaly_levels)

        self.thresholds = None

    @staticmethod
    def _validate_quantiles(quantiles):
        if not quantiles:
            raise ValueError("quantiles must be a non-empty sequence")

        normalized = tuple(float(q) for q in quantiles)

        if len(normalized) != 3:
            raise ValueError(
                "quantiles must contain exactly three thresholds for low/medium/high"
            )

        if len(set(normalized)) != len(normalized):
            raise ValueError("quantiles must not contain duplicates")

        for q in normalized:
            if q <= 0.0 or q >= 1.0:
                raise ValueError("quantiles must be between 0 and 1 (exclusive)")

        return tuple(sorted(normalized))

    # Fit residual distribution (based on normal data)
    def fit(self, y_true, y_pred):
        residuals = y_true - y_pred
        scores = np.abs(residuals)

        self.thresholds = {
            q: {"value": np.quantile(scores, q), "severity": self.label_map[q]}
            for q in self.quantiles
        }

    # Compute anomaly scores as absolute residuals
    def score(self, y_true, y_pred):
        residuals = y_true - y_pred
        return np.abs(residuals)

    # Map severity based on thresholds
    def _map_severity(self, score):
        if self.thresholds is None:
            raise RuntimeError("Detector is not fitted. Call fit() before detect().")

        sorted_quantiles = sorted(self.thresholds.keys())

        if score < self.thresholds[sorted_quantiles[0]]["value"]:
            return "normal"

        for q in reversed(sorted_quantiles):
            if score >= self.thresholds[q]["value"]:
                return self.thresholds[q]["severity"]

        return self.thresholds[sorted_quantiles[0]]["severity"]

    # Detect anomalies and classify severity
    def detect(self, y_true, y_pred):
        scores = self.score(y_true, y_pred)

        results = []

        for s in scores:
            severity = self._map_severity(s)
            results.append(
                {
                    "score": float(s),
                    "severity": severity,
                    "is_anomaly": severity in self.anomaly_levels,
                }
            )

        return pd.DataFrame(results)

    # Return computed thresholds
    def get_thresholds(self):
        return self.thresholds
