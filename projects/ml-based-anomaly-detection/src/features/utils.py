from src.config_loader import load_config

def get_feature_columns():
    """Load feature column names from config."""
    cfg = load_config()
    lag_names = [f["name"] for f in cfg["features"]["lags"]]
    rolling_names = [f["name"] for f in cfg["features"]["rolling"]]
    calendar_names = [f["name"] for f in cfg["features"]["calendar"]]
    stl_names = ["trend", "seasonal"]
    return lag_names + rolling_names + calendar_names + stl_names