import yaml
from pathlib import Path


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_anomaly_detection_policy(config):
    return config["anomaly_detection"]
