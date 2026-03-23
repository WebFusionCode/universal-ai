import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load model
model_package = joblib.load(BASE_DIR / "best_model.pkl")

model = model_package["model"]
feature_columns = model_package["feature_columns"]

print("Loaded model:", type(model))

# Example test data (replace with real values)
data = pd.DataFrame([{
    col: 0 for col in feature_columns
}])

prediction = model.predict(data)

print("Prediction:", prediction)
