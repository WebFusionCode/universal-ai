# =====================================================
# IMPORTS
# =====================================================

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from prophet import Prophet

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import io
import joblib
import zipfile
import shutil
from PIL import Image
from fastapi import WebSocket
import asyncio

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from models.model_library import CLASSIFICATION_MODELS, REGRESSION_MODELS
from utils.problem_detection import detect_problem_type




# =====================================================
# APP SETUP
# =====================================================

app = FastAPI()

UPLOAD_FOLDER = "uploads"
IMAGE_DATASET_FOLDER = "image_dataset"
MODEL_PATH = "best_model.pkl"
CNN_MODEL_PATH = "cnn_model.pth"

training_progress = {"value": 0}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_DATASET_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")







# =====================================================
# PREVIEW
# =====================================================
@app.post("/preview")
async def preview_dataset(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file_path, sep=None, engine="python")
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        return {"error": "Unsupported file format"}

    df.columns = df.columns.str.strip()

    # Basic Info
    rows = len(df)
    columns = list(df.columns)

    # Detect numeric & categorical
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Detect possible date column
    date_column = None

    for col in categorical_cols:

    # Skip numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        try:
        # Try converting column to datetime
            converted = pd.to_datetime(df[col], errors="coerce")

            valid_ratio = converted.notna().sum() / max(len(df), 1)
            unique_ratio = converted.nunique() / max(len(df), 1)

        # Safe time-series detection
            if (
                valid_ratio > 0.8 and
                unique_ratio > 0.5 and
                any(keyword in col.lower() for keyword in ["date", "time", "year", "month"])
            ):
                date_column = col
                df[col] = converted
                break

        except Exception:
            continue

    # Target suggestions (numeric columns are usually good candidates)
    target_suggestions = numeric_cols[:5]

    return {
        "rows": rows,
        "columns": columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "detected_date_column": date_column,
        "suggested_target_columns": target_suggestions
    }










# =====================================================
# GENETIC EVOLUTION ENGINE
# =====================================================
def genetic_evolution(X_train, X_test, y_train, y_test, problem_type, generations=5, population_size=6):

    def create_individual():
        if problem_type == "classification":
            return RandomForestClassifier(
                n_estimators=random.randint(50, 300),
                max_depth=random.choice([None, 5, 10, 20]),
                min_samples_split=random.randint(2, 10)
            )
        else:
            return RandomForestRegressor(
                n_estimators=random.randint(50, 300),
                max_depth=random.choice([None, 5, 10, 20]),
                min_samples_split=random.randint(2, 10)
            )

    def fitness(model):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if problem_type == "classification":
            return accuracy_score(y_test, preds)
        else:
            return -mean_squared_error(y_test, preds)

    population = [create_individual() for _ in range(population_size)]

    best_model = None
    best_score = float("-inf")

    for generation in range(generations):

        scored_population = []

        for model in population:
            score = fitness(model)
            scored_population.append((score, model))

            if score > best_score:
                best_score = score
                best_model = model

        # Select top 50%
        scored_population.sort(key=lambda x: x[0], reverse=True)
        survivors = [model for _, model in scored_population[:population_size // 2]]

        # Mutation
        new_population = survivors.copy()

        while len(new_population) < population_size:
            parent = random.choice(survivors)
            child = create_individual()  # fresh mutation
            new_population.append(child)

        population = new_population

    return best_model, best_score










# =====================================================
# SAFE TIME SERIES DETECTION
# =====================================================
def detect_time_series(df, target_column):
    for col in df.columns:
        if col == target_column:
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            if parsed.nunique() > len(parsed) * 0.5:
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    return True, col
        except:
            continue
    return False, None











# =====================================================
# GENETIC ALGORITHM FOR TABULAR
# =====================================================
def genetic_model_search(X_train, X_test, y_train, y_test, problem_type):

    population = []

    # Initial population
    for _ in range(6):
        n = np.random.randint(50, 200)
        if problem_type == "classification":
            model = RandomForestClassifier(n_estimators=n)
        else:
            model = RandomForestRegressor(n_estimators=n)
        population.append(model)

    best_model = None
    best_score = None
    best_metrics = None

    for generation in range(3):

        scored_population = []

        for model in population:

            model.fit(X_train, y_train)

            if problem_type == "classification":

                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                metrics = {
                    "accuracy": acc
                }

                score = acc
                better = best_score is None or score > best_score

            else:

                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                r2 = model.score(X_test, y_test)

                metrics = {
                    "mse": mse,
                    "r2": r2
                }

                score = mse
                better = best_score is None or score < best_score

            if better:
                best_score = score
                best_model = model
                best_metrics = metrics

            scored_population.append({
                "model": model,
                "score": score,
                "metrics": metrics
            })

        # Sort
        if problem_type == "classification":
            scored_population.sort(key=lambda x: x["score"], reverse=True)
        else:
            scored_population.sort(key=lambda x: x["score"])

        # Keep top 3
        survivors = scored_population[:3]

        # Create new generation
        new_population = [item["model"] for item in survivors]

        while len(new_population) < 6:

            parent = survivors[np.random.randint(0, len(survivors))]["model"]

            mutated_n = parent.n_estimators + np.random.randint(-20, 20)
            mutated_n = max(10, mutated_n)

            if problem_type == "classification":
                child = RandomForestClassifier(n_estimators=mutated_n, random_state=42)
            else:
                child = RandomForestRegressor(n_estimators=mutated_n, random_state=42)

            new_population.append(child)

        population = new_population

    # Prepare top models output
    top_models = []

    for item in survivors:
        model = item["model"]
        metrics = item["metrics"]

        model_info = {
            "model": type(model).__name__,
            **{k: round(v, 4) for k, v in metrics.items()}
        }

        top_models.append(model_info)

    return best_model, best_metrics, top_models






# =====================================================
# model_strength_summary
# =====================================================
def model_strength_summary(problem_type, metrics):

    if problem_type == "classification":

        acc = metrics["accuracy"]

        if acc > 0.9:
            level = "Excellent"
        elif acc > 0.8:
            level = "Strong"
        elif acc > 0.7:
            level = "Moderate"
        else:
            level = "Weak"

        return {
            "model_strength": level,
            "accuracy": round(acc,4)
        }

    else:

        r2 = metrics["r2"]

        if r2 > 0.9:
            level = "Excellent"
        elif r2 > 0.8:
            level = "Strong"
        elif r2 > 0.6:
            level = "Moderate"
        else:
            level = "Weak"

        return {
            "model_strength": level,
            "r2_score": round(r2,4)
        }
        
        
        
        





        
# =====================================================
# generate_explanation_text 
# =====================================================
def generate_explanation_text(problem_type, strength_summary):

    if problem_type == "classification":

        return (
            f"This classification model demonstrates {strength_summary['model_strength']} "
            f"performance with an accuracy of {strength_summary['accuracy']}. "
            "The model has learned patterns effectively from the dataset."
        )

    else:

        return (
            f"This regression model demonstrates {strength_summary['model_strength']} "
            f"performance with an R² score of {strength_summary['r2_score']}. "
            "The model captures variance in the data efficiently."
        )



       
        
        
        
        
                
        
# =====================================================
# CNN MODEL
# =====================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True
            
        # Replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
def generate_gradcam(model, image_tensor, target_class):

    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # 🔥 Hook last convolution block of ResNet18
    target_layer = model.model.layer4[-1]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    loss = output[:, target_class]

    model.zero_grad()
    loss.backward()

    grad = gradients[0]
    act = activations[0]

    weights = torch.mean(grad, dim=(2,3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()

    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()
        
    cam = cam.detach().cpu().numpy()

    handle_forward.remove()
    handle_backward.remove()

    return cam










# =====================================================
# IMAGE AUTO ORGANIZER
# =====================================================
def universal_image_organizer(root_folder):

    normalized_folder = os.path.join(root_folder, "__normalized__")

    # Remove old normalized folder if exists
    if os.path.exists(normalized_folder):
        shutil.rmtree(normalized_folder)

    os.makedirs(normalized_folder, exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    for current_root, dirs, files in os.walk(root_folder):

        # 🔥 Skip normalized folder itself
        if "__normalized__" in current_root:
            continue

        for file in files:
            if file.lower().endswith(image_extensions):

                full_path = os.path.join(current_root, file)

                # Determine class name from parent folder
                relative_path = os.path.relpath(current_root, root_folder)
                parts = relative_path.split(os.sep)

                parts = [p for p in parts if p not in (".", "")]

                if len(parts) >= 1:
                    class_name = parts[-1]
                else:
                    class_name = "unknown"

                target_class_folder = os.path.join(normalized_folder, class_name)
                os.makedirs(target_class_folder, exist_ok=True)

                destination = os.path.join(target_class_folder, file)

                # 🔥 Prevent SameFileError
                if os.path.abspath(full_path) != os.path.abspath(destination):
                    shutil.copy2(full_path, destination)

    # Remove everything except normalized folder
    for item in os.listdir(root_folder):
        if item == "__normalized__":
            continue

        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

    # Move normalized data up
    for class_name in os.listdir(normalized_folder):
        shutil.move(
            os.path.join(normalized_folder, class_name),
            os.path.join(root_folder, class_name)
        )

    shutil.rmtree(normalized_folder)










# =====================================================
# PREVIEW COLUMNS ENDPOINT
# =====================================================
@app.post("/preview-columns")
async def preview_columns(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file_path, nrows=5)
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file_path, nrows=5)
    else:
        return {"error": "Unsupported format"}

    return {
        "columns": list(df.columns),
        "preview": df.head().to_dict(orient="records")
    }




# =====================================================
# dataset_quality_score
# =====================================================
def dataset_quality_score(df):

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()

    missing_ratio = missing_cells / total_cells

    numeric_ratio = (
        len(df.select_dtypes(include=["int64","float64"]).columns)
        / max(1, len(df.columns))
    )

    score = 100

    # Penalize missing values
    score -= missing_ratio * 40

    # Penalize very small datasets
    if len(df) < 100:
        score -= 20
    elif len(df) < 500:
        score -= 10

    # Encourage numeric richness
    score += numeric_ratio * 10

    score = max(0, min(100, round(score,2)))

    return {
        "quality_score": score,
        "missing_ratio": round(missing_ratio,4),
        "numeric_feature_ratio": round(numeric_ratio,4)
    }
    
    
    
    
    
    
    
    

# =====================================================
# AUTO TRAIN
# =====================================================
@app.post("/train")
async def auto_train(file: UploadFile = File(...), target_column: str = Form(None)):

    filename = file.filename.lower()

    # ==========================================
    # IMAGE DATASET
    # ==========================================
    if filename.endswith(".zip"):

        zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        shutil.rmtree(IMAGE_DATASET_FOLDER, ignore_errors=True)
        os.makedirs(IMAGE_DATASET_FOLDER)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(IMAGE_DATASET_FOLDER)

        universal_image_organizer(IMAGE_DATASET_FOLDER)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2,0.2,0.2,0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

        dataset = datasets.ImageFolder(IMAGE_DATASET_FOLDER, transform=transform)

        if len(dataset.classes) < 2:
            return {"error": "Dataset must contain at least 2 classes"}

        if len(dataset) < 20:
            return {"error": "Dataset too small for training"}

        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = SimpleCNN(len(dataset.classes)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            list(model.model.layer4.parameters()) +
            list(model.model.fc.parameters()),
            lr=0.001
        )

        training_progress["value"] = 0
        total_epochs = 5

        for epoch in range(total_epochs):

            model.train()

            for images, labels in loader:

                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

            training_progress["value"] = int(((epoch + 1) / total_epochs) * 100)

        training_progress["value"] = 100

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": dataset.classes
        }, CNN_MODEL_PATH)

        return {
            "problem_type": "Image Classification",
            "classes": dataset.classes,
            "samples": len(dataset)
        }

    # ==========================================
    # TABULAR / TIME SERIES
    # ==========================================
    if filename.endswith(".csv") or filename.endswith(".xlsx"):

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if filename.endswith(".csv"):
            df = pd.read_csv(file_path, sep=None, engine="python")
        else:
            df = pd.read_excel(file_path)

        df.columns = df.columns.str.strip()

        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")

        numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # ==========================================
        # TIME SERIES DETECTION
        # ==========================================

        date_column = None

        for col in categorical_cols:

            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            try:

                converted = pd.to_datetime(df[col], errors="coerce")

                valid_ratio = converted.notna().sum() / max(len(df),1)

                unique_ratio = converted.nunique() / max(len(df),1)

                if (
                    valid_ratio > 0.8 and
                    unique_ratio > 0.5 and
                    any(k in col.lower() for k in ["date","time","year","month"])
                ):

                    date_column = col
                    df[col] = converted
                    break

            except:
                continue

        if date_column:

            df = df.sort_values(by=date_column)

            numeric_columns = []

            for col in df.columns:

                if col == date_column:
                    continue

                converted = pd.to_numeric(df[col], errors="coerce")

                if converted.notna().sum() > len(df) * 0.6:
                    df[col] = converted
                    numeric_columns.append(col)

            if len(numeric_columns) > 0:

                models = {}

                for col in numeric_columns:

                    ts_df = df[[date_column, col]].dropna().copy()

                    ts_df.columns = ["ds","y"]

                    if len(ts_df) < 10:
                        continue

                    model = Prophet()

                    model.fit(ts_df)

                    models[col] = model

                if len(models) > 0:

                    joblib.dump({
                        "models": models,
                        "problem_type": "time_series_multi",
                        "date_column": date_column,
                        "target_columns": list(models.keys())
                    }, MODEL_PATH)

                    return {
                        "problem_type": "Time-Series (Universal)",
                        "date_column": date_column,
                        "target_columns": list(models.keys()),
                        "rows": len(df),
                        "message": "Model trained successfully. Use /predict to forecast."
                    }

        # ==========================================
        # TABULAR AUTOML ENGINE
        # ==========================================

        if target_column is None:
            return {"error": "Provide target_column for tabular dataset"}

        if target_column not in df.columns:
            return {
                "error": "Invalid target column",
                "available_columns": list(df.columns)
            }

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Detect problem type
        problem_type = detect_problem_type(df, target_column)

        if problem_type == "classification" and y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # Numeric preprocessing
        numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()

        if len(numeric_cols) > 0:
            X[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(X[numeric_cols])
            X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

        # Categorical preprocessing
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        for col in categorical_cols:

            X[col] = X[col].astype(str)

            X[col] = X[col].fillna("missing")

            X[col] = LabelEncoder().fit_transform(X[col])

        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Select model pool
        if problem_type == "classification":
            models = CLASSIFICATION_MODELS
        else:
            models = REGRESSION_MODELS

        results = []
        best_model = None
        best_score = -999

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)

            results.append({
                "model": name,
                "score": float(score)
            })

            if score > best_score:
                best_score = score
                best_model = model

        top_models = sorted(results, key=lambda x: x["score"], reverse=True)

        joblib.dump({
            "model": best_model,
            "feature_columns": X.columns.tolist(),
            "problem_type": problem_type
        }, MODEL_PATH)

        if problem_type == "classification":

            return {
                "problem_type": "Classification",
                "rows": len(df),
                "target_column": target_column,
                "best_model": type(best_model).__name__,
                "accuracy": round(best_score,4),
                "top_models": top_models
            }

        else:

            mse = mean_squared_error(y_test, best_model.predict(X_test))

            return {
                "problem_type": "Regression",
                "rows": len(df),
                "target_column": target_column,
                "best_model": type(best_model).__name__,
                "mse": round(mse,4),
                "r2": round(best_score,4),
                "top_models": top_models
            }

    return {"error": "Unsupported dataset"}








# =====================================================
# RISK ANALYSIS
# =====================================================
def risk_analysis(prob_array):

    confidence = float(np.max(prob_array))

    if confidence > 0.85:
        risk = "Low Risk"
    elif confidence > 0.65:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return {
        "confidence": round(confidence, 4),
        "risk_level": risk
    }
    
    
    
    
    
    
    
    
    

# =====================================================
# TABULAR / TIME SERIES PREDICT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not os.path.exists(MODEL_PATH):
        return {"error": "No trained model found"}

    model_package = joblib.load(MODEL_PATH)
    problem_type = model_package.get("problem_type")

    # ==========================================
    # TIME SERIES PREDICTION
    # ==========================================
    if problem_type == "time_series_multi":

        models = model_package.get("models", {})
        forecast_output = {}

        for col, model in models.items():

            future = model.make_future_dataframe(periods=10)
            forecast = model.predict(future)

            future_only = forecast.tail(10)[["ds", "yhat"]]

            forecast_output[col] = future_only.to_dict(orient="records")

        return {
            "problem_type": "Time-Series (Universal)",
            "forecast_horizon": 10,
            "forecast": forecast_output
        }

    # ==========================================
    # TABULAR PREDICTION
    # ==========================================
    model = model_package.get("model")
    feature_columns = model_package.get("feature_columns")

    if model is None or feature_columns is None:
        return {"error": "Invalid tabular model package"}

    try:
        # Detect file type
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file, sep=None, engine="python")
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            return {"error": "Unsupported file format"}

        df.columns = df.columns.str.strip()

        # Validate required columns
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            return {
                "error": "Missing required columns",
                "missing_columns": missing_cols
            }

        df = df[feature_columns]

        # Numeric handling
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # Categorical handling
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])

        df = df.fillna(0)

        # ==========================================
        # CLASSIFICATION OUTPUT (With Risk Analysis)
        # ==========================================
        if problem_type == "classification":

            predictions = model.predict(df)

            probabilities = model.predict_proba(df) if hasattr(model, "predict_proba") else None

            results = []

            for i in range(len(predictions)):

                prediction = int(predictions[i])

                if probabilities is not None:

                    prob_array = probabilities[i]

                    # Use your risk analysis function
                    risk_info = risk_analysis(prob_array)

                    prob_dict = {
                        str(idx): round(float(prob), 4)
                        for idx, prob in enumerate(prob_array)
                    }

                    results.append({
                        "prediction": prediction,
                        "confidence": risk_info["confidence"],
                        "risk_level": risk_info["risk_level"],
                        "probabilities": prob_dict
                    })

                else:

                    results.append({
                        "prediction": prediction,
                        "confidence": None,
                        "risk_level": "Unknown",
                        "probabilities": None
                    })

            return {
                "problem_type": "Classification",
                "num_predictions": len(results),
                "predictions": results
            }

        # ==========================================
        # REGRESSION OUTPUT
        # ==========================================
        elif problem_type == "regression":

            predictions = model.predict(df)

            return {
                "problem_type": "Regression",
                "num_predictions": len(predictions),
                "predictions": [
                    round(float(p), 4) for p in predictions
                ]
            }

        else:
            return {"error": "Unknown problem type"}

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}










# =====================================================
# IMAGE PREDICT
# =====================================================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    if not os.path.exists(CNN_MODEL_PATH):
        return {"error":"No trained CNN model"}

    checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = SimpleCNN(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image=Image.open(file.file).convert("RGB")

    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

    img_tensor=transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output=model(img_tensor)
        probs=torch.softmax(output,dim=1)
        confidence,pred=torch.max(probs,1)

    return {
        "predicted_class":classes[pred.item()],
        "confidence":float(confidence.item())
    }


@app.post("/explain-image")
async def explain_image(file: UploadFile = File(...)):

    if not os.path.exists(CNN_MODEL_PATH):
        return {"error": "No trained CNN model"}
    
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = SimpleCNN(len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(file.file).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    cam = generate_gradcam(model, img_tensor, pred.item())

    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam).resize((224,224))
    cam = np.array(cam)

    original = np.array(image.resize((224,224)))

    heatmap = plt.cm.jet(cam)[:,:,:3]
    heatmap = np.uint8(255 * heatmap)

    superimposed = heatmap * 0.4 + original
    superimposed = np.uint8(superimposed)

    final_img = Image.fromarray(superimposed)

    img_io = io.BytesIO()
    final_img.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/png")










# =====================================================
# MODEL EXPLAIN
# =====================================================
@app.get("/model-explain")
def model_explain():

    if not os.path.exists(MODEL_PATH):
        return {"error": "No trained model found"}

    model_package = joblib.load(MODEL_PATH)

    problem_type = model_package.get("problem_type")

    # 🚫 Block time-series explanation
    if problem_type in ["time_series", "time_series_multi"]:
        return {
            "message": "Model explain not supported for Time-Series models"
        }

    # 🚫 Safety check
    if "model" not in model_package:
        return {
            "error": "No tabular model found to explain"
        }

    model = model_package["model"]
    feature_columns = model_package.get("feature_columns", [])

    # Feature Importance (Tree Models)
    if hasattr(model, "feature_importances_"):

        importance = dict(
            sorted(
                zip(feature_columns, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )
        )

        return {
            "model_type": type(model).__name__,
            "feature_importance": importance
        }

    # Coefficients (Linear Models)
    elif hasattr(model, "coef_"):

        coef = model.coef_

        if len(coef.shape) > 1:
            coef = coef[0]

        importance = dict(
            sorted(
                zip(feature_columns, coef),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        )

        return {
            "model_type": type(model).__name__,
            "coefficients": importance
        }

    else:
        return {
            "message": "This model type does not support explainability"
        }









# =====================================================
# SHAP
# =====================================================
# @app.post("/shap-explain")
# async def shap_explain(file: UploadFile = File(...)):

#     if not os.path.exists(MODEL_PATH):
#         return {"error":"No trained model"}

#     model_package=joblib.load(MODEL_PATH)

#     if model_package["problem_type"]=="time_series":
#         return {"message":"SHAP not supported for time-series"}

#     model=model_package["model"]
#     features=model_package["feature_columns"]

#     df=pd.read_csv(file.file)
#     df=df[features]

#     explainer=shap.TreeExplainer(model)
#     shap_values=explainer.shap_values(df)
#     shap_values=np.abs(np.array(shap_values))
#     importance=np.mean(shap_values,axis=0)

#     return dict(zip(features,importance.tolist()))









# =====================================================
# DOWNLOAD
# =====================================================
@app.get("/download-model")
def download_model():
    if os.path.exists(MODEL_PATH):
        return FileResponse(MODEL_PATH)
    return {"error":"No trained model"}







@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json(training_progress)
        await asyncio.sleep(0.5)
        
        
        
        
        
        
        

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})