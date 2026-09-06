# scripts/evaluate_models.py
from src.evaluate import evaluate_models


def main():
    results, best_model = evaluate_models()

    print("\nMetriken aller Modelle")
    for name, m in results.items():
        print(f"\nModell: {name}")
        print(f"  RMSE: {m['rmse']:.2f}")
        print(f"  MAE:  {m['mae']:.2f}")
        print(f"  R²:   {m['r2']:.3f}")

    if best_model is not None:
        print(f"\nBestes Modell (nach RMSE): {best_model}")
    else:
        print("\nKein bestes Modell bestimmbar.")


if __name__ == "__main__":
    main()
