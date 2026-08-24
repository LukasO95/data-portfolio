import joblib


def model_exists(path):
    return path.exists()


def load_model(path):
    return joblib.load(path)


def save_model(model, path):
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)