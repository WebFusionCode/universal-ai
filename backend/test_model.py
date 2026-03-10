import joblib
import pandas as pd

# Load model
model_package = joblib.load("best_model.pkl")

model = model_package["model"]
feature_columns = model_package["feature_columns"]

print("Loaded model:", type(model))

# Example test data (replace with real values)
data = pd.DataFrame([{
    col: 0 for col in feature_columns
}])

prediction = model.predict(data)

print("Prediction:", prediction)