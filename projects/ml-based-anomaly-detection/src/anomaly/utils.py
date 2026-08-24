import json
from typing import Dict, Any


def print_thresholds(thresholds: Dict[float, Dict[str, Any]]) -> None:
    """
    Pretty-print anomaly detection thresholds.

    Expected structure:
    {
        0.9:  {"value": 45123.7, "severity": "low"},
        0.95: {"value": 78456.2, "severity": "medium"},
        0.99: {"value": 121987.5, "severity": "high"}
    }
    """

    print("\n=== Anomaly Detection Thresholds ===")

    for q in sorted(thresholds.keys()):
        entry = thresholds[q]

        value = entry["value"]
        severity = entry.get("severity", "")

        print(f"Q{int(q*100):<3} ({severity:<6}): {value:,.0f}")


def save_thresholds(thresholds: Dict[float, Dict[str, Any]], path: str) -> None:
    """
    Save thresholds to JSON file.

    Converts:
    - float keys → string keys (JSON requirement)
    - numpy types → native Python types
    """

    thresholds_serializable = {
        str(q): {
            "value": float(entry["value"]),
            "severity": entry.get("severity", "")
        }
        for q, entry in thresholds.items()
    }

    with open(path, "w") as f:
        json.dump(thresholds_serializable, f, indent=2)

    print(f"\nSaved thresholds to {path}")


def load_thresholds(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)