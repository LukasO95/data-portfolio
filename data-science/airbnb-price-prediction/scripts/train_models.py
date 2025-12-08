# scripts/train_models.py
from src.train_model import train_all_models

if __name__ == "__main__":
    results = train_all_models()
    print("\nGesamtergebnis")
    for name, m in results.items():
        print(f"{name}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, R²={m['r2']:.3f}")