import re
import os

with open("backend/main.py", "r") as f:
    code = f.read()

# 1. Update paths at the top
code = code.replace('MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")',
                    'TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_model.pkl")\n'
                    'TIME_SERIES_MODEL_PATH = os.path.join(MODEL_DIR, "time_series.pkl")')
code = code.replace('CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")',
                    'IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")')

# Replace global usages of CNN_MODEL_PATH with IMAGE_MODEL_PATH
code = code.replace('CNN_MODEL_PATH', 'IMAGE_MODEL_PATH')

# 2. Fix Tabular save
# Tabular save is around "joblib.dump(\n        {\n            "problem_type": problem_type,"
# Actually, let's just find the tabular joblib.dump and time_series joblib.dump and replace MODEL_PATH

# 3. Update Predict Endpoint
predict_old = """    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet."}

    model_package = joblib.load(MODEL_PATH)"""

predict_new = """    if os.path.exists(IMAGE_MODEL_PATH):
        return {"error": "Latest model is image. Use /predict-image instead."}

    if os.path.exists(TABULAR_MODEL_PATH):
        model_package = joblib.load(TABULAR_MODEL_PATH)
        problem_type = model_package.get("problem_type")
        model_type = "tabular"
    elif os.path.exists(TIME_SERIES_MODEL_PATH):
        model_package = joblib.load(TIME_SERIES_MODEL_PATH)
        problem_type = model_package.get("problem_type")
        model_type = "time_series"
    else:
        return {"error": "No tabular/time-series model found. Train one first."}"""

# Fix predict
if predict_old in code:
    code = code.replace(predict_old, predict_new)

with open("backend/main.py", "w") as f:
    f.write(code)

print("Replaced predict and top vars.")
