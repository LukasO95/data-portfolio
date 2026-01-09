# Airbnb Price Prediction

End-to-end machine learning project using real-world Airbnb data to predict nightly prices and evaluate different models.

**Dependencies:** see `requirements.txt`

---

## 🎯 Goal
- Predict **price per night** for Airbnb listings
- Identify **key price drivers**
- Compare multiple models and evaluate performance

---

## 📊 Data
- **Source:** Inside Airbnb
- **Target:** `price` (nightly rate)
- Cleaning, outlier removal, missing value handling, and sampling applied

---

## 🧠 Models
- Random Forest
- Random Forest (log-transformed target)
- XGBoost (log-transformed target)

---

## 📈 Results

| Model               | RMSE | MAE  | R²    |
|--------------------|------|------|-------|
| Random Forest      | 65.56 | 47.08 | 0.386 |
| Random Forest (Log)| 64.84 | 43.92 | 0.400 |
| XGBoost (Log)      | 65.49 | 44.96 | 0.387 |

**Best model:** Random Forest with log-transformed target

---

## 🧰 Tech Stack
Python 3.13.5 · pandas · numpy · scikit-learn · matplotlib · xgboost

---

## 📂 Structure
data / models / notebooks / scripts / src

---

## ▶️ Run
```bash
PYTHONPATH=. python scripts/make_processed_data.py --sample_size 10000
Modeling and evaluation are documented in Jupyter notebooks.

👤 Author
Lukas Ogrzewalla 