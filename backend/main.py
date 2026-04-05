import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from datetime import timedelta
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import random
import shutil
import uuid
import zipfile

try:
    from dotenv import load_dotenv
except Exception:

    def load_dotenv(*args, **kwargs):
        return False


from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import joblib
from jose import JWTError, jwt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== DEPLOYMENT SETTINGS =====
LIGHTWEIGHT_DEPLOYMENT = True   # for Render free tier
MAX_LIGHTWEIGHT_ROWS = 10000    # limit dataset size

# ===== DEFAULT SETTINGS =====
PREVIEW_RESPONSE_ROWS = 5

# ===== FILE PATHS =====
UPLOAD_FOLDER = "uploads"
EXPERIMENTS_DIR = "experiments"
EXPERIMENTS_PATH = os.path.join(EXPERIMENTS_DIR, "experiments.json")

training_progress = {
    "progress": 0,
    "status": "Idle",
    "message": ""
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

if not os.path.exists(EXPERIMENTS_PATH):
    with open(EXPERIMENTS_PATH, "w") as f:
        f.write("[]")

SECRET_KEY = "mysecretkey123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

try:
    from pymongo.errors import PyMongoError
except Exception:
    PyMongoError = Exception
from pydantic import BaseModel, EmailStr
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    password = password[:72]   # 🔥 FIX
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    plain_password = plain_password[:72]   # 🔥 FIX
    return pwd_context.verify(plain_password, hashed_password)

                 
class UserSignup(BaseModel):
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


                    
from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    print("❌ MONGO_URL missing")
else:
    print("✅ MONGO_URL found")

client = MongoClient(MONGO_URL)
db = client["automl_db"]
users_collection = db["users"]
models_collection = db["models"]
subscriptions_collection = db["subscriptions"]
teams_collection = db["teams"]
usage_collection = db["usage"]

print("🚀 Starting FastAPI app...")

load_dotenv(Path(__file__).resolve().parent / ".env")

import os

print("MONGO_URL:", os.getenv("MONGO_URL"))

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                     
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("✅ App created")

                    
from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    print("❌ MONGO_URL missing")
else:
    print("✅ MONGO_URL found")

client = MongoClient(MONGO_URL)
db = client["automl_db"]
users_collection = db["users"]
models_collection = db["models"]
subscriptions_collection = db["subscriptions"]
teams_collection = db["teams"]
usage_collection = db["usage"]


def lightweight_feature_message(feature_name):
    return (
        f"{feature_name} is disabled in lightweight deployment mode to keep startup "
        "stable and memory usage low."
    )


@lru_cache(maxsize=1)
def get_openai_client():
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    return OpenAI(api_key=OPENAI_API_KEY)


@lru_cache(maxsize=1)
def get_model_library():
    from models.model_library import CLASSIFICATION_MODELS, REGRESSION_MODELS

    return CLASSIFICATION_MODELS, REGRESSION_MODELS


def analyze_training_dataset(X, y):
    from utils.automl_brain import analyze_dataset

    return analyze_dataset(X, y)


def recommend_training_models(dataset_info, problem_type):
    from utils.automl_brain import recommend_models

    return recommend_models(dataset_info, problem_type)


def run_auto_feature_engineering(df):
    from utils.feature_engineering import auto_feature_engineering

    return auto_feature_engineering(df)


def detect_training_problem_type(df, target_column):
    from utils.problem_detection import detect_problem_type

    return detect_problem_type(df, target_column)


def generate_training_pipeline_code(
    model_name, feature_columns, problem_type, target_column
):
    from utils.code_generator import generate_training_code

    return generate_training_code(
        model_name=model_name,
        feature_columns=feature_columns,
        problem_type=problem_type,
        target_column=target_column,
    )


def get_tune_random_forest():
    from utils.hyperparameter_tuning import tune_random_forest

    return tune_random_forest


@lru_cache(maxsize=1)
def get_torch_runtime():
    if LIGHTWEIGHT_DEPLOYMENT:
        raise RuntimeError(lightweight_feature_message("Deep learning features"))

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except Exception as exc:
        raise RuntimeError(
            "PyTorch dependencies are not installed. Add torch and torchvision to enable deep learning features."
        ) from exc

    return {
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "datasets": datasets,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "models": models,
        "transforms": transforms,
    }


@lru_cache(maxsize=1)
def get_prophet_class():
    if LIGHTWEIGHT_DEPLOYMENT:
        raise RuntimeError(lightweight_feature_message("Time-series training"))

    try:
        from prophet import Prophet
    except Exception as exc:
        raise RuntimeError(
            "Prophet is not installed. Add it back to enable time-series training."
        ) from exc

    return Prophet


@lru_cache(maxsize=1)
def get_shap_module():
    if LIGHTWEIGHT_DEPLOYMENT:
        raise RuntimeError(lightweight_feature_message("SHAP explanations"))

    try:
        import shap
    except Exception as exc:
        raise RuntimeError(
            "SHAP is not installed. Add shap back to enable SHAP explanations."
        ) from exc

    return shap


@lru_cache(maxsize=1)
def get_image_module():
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Pillow is not installed. Add pillow to enable image processing."
        ) from exc

    return Image


@lru_cache(maxsize=1)
def get_matplotlib_pyplot():
    mpl_config_dir = os.path.join(BASE_DIR, ".matplotlib")
    if not os.path.exists(mpl_config_dir):
        os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is not installed. Add it back to enable image explanation heatmaps."
        ) from exc

    return plt


def get_tab_transformer_class():
    if LIGHTWEIGHT_DEPLOYMENT:
        raise RuntimeError(lightweight_feature_message("Transformer models"))

    from models.transformer_models import TabTransformer

    return TabTransformer


def require_openai_client():
    client = get_openai_client()

    if not client:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the backend.",
        )
    return client


def extract_user_id_from_request(request: Request):
    auth_header = request.headers.get("Authorization", "").strip()

    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ", 1)[1].strip()

    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

    return payload.get("sub")


def load_experiment_logs():
    if not os.path.exists(EXPERIMENTS_PATH):
        return []

    with open(EXPERIMENTS_PATH, "r") as file_obj:
        try:
            data = json.load(file_obj)
        except json.JSONDecodeError:
            return []

    return data if isinstance(data, list) else []


def filter_experiments_for_user(experiments, user_id=None):
    if not user_id:
        return experiments

    return [item for item in experiments if item.get("user_id") == user_id]


def save_model_record(user_id, model_name, model_version, dataset_type, score=None):
    if models_collection is None:
        return

    if not user_id:
        return

    try:
        models_collection.insert_one(
            {
                "user_id": user_id,
                "model_name": model_name,
                "model_version": model_version,
                "dataset_type": dataset_type,
                "score": float(score) if score is not None else None,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except PyMongoError:
        return


def track_usage_event(user_id, action, metadata=None):
    if usage_collection is None:
        return

    if not user_id or not action:
        return

    document = {
        "user_id": user_id,
        "action": action,
        "time": datetime.utcnow().isoformat(),
    }

    if metadata:
        document["metadata"] = metadata

    try:
        usage_collection.insert_one(document)
    except PyMongoError:
        return


def is_admin(user_id):
    if users_collection is None:
        return False

    if not user_id:
        return False

    try:
        user = users_collection.find_one({"user_id": user_id}, {"role": 1})
    except PyMongoError:
        return False

    return (user or {}).get("role") == "admin"


def require_admin_user(request: Request):
    user_id = extract_user_id_from_request(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")

    return user_id


def update_progress(progress, status, message):
    global training_progress
    training_progress["progress"] = progress
    training_progress["status"] = status
    training_progress["message"] = message


def build_leaderboard_payload(
    best_model_name,
    model_version,
    models,
    dataset_type=None,
    problem_type=None,
):
    normalized_models = []

    for index, model_info in enumerate(models or []):
        normalized_models.append(
            {
                "rank": model_info.get("rank", index + 1),
                "model": model_info.get("model", "Unknown"),
                "score": (
                    float(model_info["score"])
                    if model_info.get("score") is not None
                    else None
                ),
                "time": (
                    float(model_info["time"])
                    if model_info.get("time") is not None
                    else None
                ),
                "metrics": model_info.get("metrics", {}),
            }
        )

    return {
        "best_model": best_model_name,
        "model_version": model_version,
        "dataset_type": dataset_type,
        "problem_type": problem_type,
        "total_models": len(normalized_models),
        "models": normalized_models,
    }


def save_leaderboard_snapshot(
    best_model_name,
    model_version,
    models,
    dataset_type=None,
    problem_type=None,
):
    payload = build_leaderboard_payload(
        best_model_name=best_model_name,
        model_version=model_version,
        models=models,
        dataset_type=dataset_type,
        problem_type=problem_type,
    )
    joblib.dump(payload, LEADERBOARD_PATH)
    return payload


def normalize_experiment_entry(entry):
    timestamp = entry.get("time") or entry.get("timestamp")
    leaderboard_models = build_leaderboard_payload(
        best_model_name=entry.get("best_model") or entry.get("model_name") or "N/A",
        model_version=entry.get("model_version", os.path.basename(MODEL_PATH)),
        models=entry.get("leaderboard", []),
        dataset_type=entry.get("dataset_type"),
        problem_type=entry.get("problem_type"),
    )["models"]
    best_model_name = (
        entry.get("best_model")
        or entry.get("model_name")
        or (leaderboard_models[0]["model"] if leaderboard_models else "N/A")
    )
    score = entry.get("score")
    if score is None:
        for model_info in leaderboard_models:
            if model_info.get("score") is not None:
                score = model_info["score"]
                break

    return {
        "time": timestamp,
        "timestamp": entry.get("timestamp", timestamp),
        "user_id": entry.get("user_id"),
        "best_model": best_model_name,
        "model_name": entry.get("model_name", best_model_name),
        "leaderboard": leaderboard_models,
        "model_version": entry.get("model_version"),
        "problem_type": entry.get("problem_type"),
        "dataset_type": entry.get("dataset_type"),
        "score": float(score) if score is not None else None,
        "rows": entry.get("rows"),
        "columns": entry.get("columns"),
        "total_models": entry.get("total_models", len(leaderboard_models)),
    }


def append_experiment_log(
    best_model_name,
    leaderboard_models,
    model_version,
    user_id=None,
    problem_type=None,
    dataset_type=None,
    rows=None,
    columns=None,
    score=None,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment = normalize_experiment_entry(
        {
            "time": timestamp,
            "timestamp": timestamp,
            "user_id": user_id,
            "best_model": best_model_name,
            "model_name": best_model_name,
            "leaderboard": leaderboard_models,
            "model_version": model_version,
            "problem_type": problem_type,
            "dataset_type": dataset_type,
            "score": score,
            "rows": rows,
            "columns": columns,
            "total_models": len(leaderboard_models or []),
        }
    )

    logs = load_experiment_logs()

    logs.append(experiment)

    with open(EXPERIMENTS_PATH, "w") as f:
        json.dump(logs, f, indent=4)

    return experiment


                                                       
         
                                                       
@app.post("/preview")
async def preview_dataset(file: UploadFile = File(...)):
    filename = file.filename if file.filename is not None else "unknown"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    filename_safe = file.filename if file.filename is not None else ""
    if filename_safe.endswith(".csv"):
        df = pd.read_csv(file_path, sep=None, engine="python")
    elif filename_safe.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        return {"error": "Unsupported file format"}

    df.columns = df.columns.str.strip()

                
    rows = len(df)
    columns = list(df.columns)

                                  
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

                                 
    date_column = None

    for col in categorical_cols:
                              
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        try:
                                               
            converted = pd.to_datetime(
                df[col], errors="coerce", infer_datetime_format=True
            )

            valid_ratio = converted.notna().sum() / max(len(df), 1)
            unique_ratio = converted.nunique() / max(len(df), 1)

                                        
            if (
                valid_ratio > 0.8
                and unique_ratio > 0.5
                and any(
                    keyword in col.lower()
                    for keyword in ["date", "time", "year", "month"]
                )
            ):
                date_column = col
                df[col] = converted
                break

        except Exception:
            continue

    preview_rows = (
        df.head(PREVIEW_RESPONSE_ROWS)
        .astype(object)
        .where(pd.notna(df.head(PREVIEW_RESPONSE_ROWS)), None)
        .to_dict(orient="records")
    )

    dataset_type = "time_series" if date_column and numeric_cols else "tabular"

    target_suggestions = numeric_cols[:5]

    return {
        "rows": rows,
        "columns": columns,
        "preview": preview_rows,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "detected_date_column": date_column,
        "suggested_target_columns": target_suggestions,
        "dataset_type": dataset_type,
    }


                                                       
                          
                                                       
def genetic_evolution(
    X_train, X_test, y_train, y_test, problem_type, generations=5, population_size=6
):

    def create_individual():
        if problem_type == "classification":
            return RandomForestClassifier(
                n_estimators=random.randint(50, 300),
                max_depth=random.choice([None, 5, 10, 20]),
                min_samples_split=random.randint(2, 10),
            )
        else:
            return RandomForestRegressor(
                n_estimators=random.randint(50, 300),
                max_depth=random.choice([None, 5, 10, 20]),
                min_samples_split=random.randint(2, 10),
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

                        
        scored_population.sort(key=lambda x: x[0], reverse=True)
        survivors = [model for _, model in scored_population[: population_size // 2]]

                  
        new_population = survivors.copy()

        while len(new_population) < population_size:
            child = create_individual()                  
            new_population.append(child)

        population = new_population

    return best_model, best_score


                                                       
                            
                                                       
def detect_time_series(df, target_column):
    for col in df.columns:
        if col == target_column:
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            parsed_series = pd.Series(parsed)
            if parsed_series.nunique() > len(parsed_series) * 0.5:
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    return True, col
        except Exception:
            continue
    return False, None


def auto_ml_recommendation(df, target_column):
    recommendations = {}

    if target_column not in df.columns:
        raise ValueError("Invalid target column")

    is_time_series, date_column = detect_time_series(df, target_column)

    if is_time_series:
        problem_type = "Time Series"
        recommended_models = [
            "Prophet",
            "XGBoost Regressor (Lag Features)",
            "RandomForestRegressor (Lag Features)",
        ]
        feature_engineering = [
            "Create lag features",
            "Add rolling statistics",
            "Extract calendar features from the date column",
        ]
        best_model_hint = (
            "Prophet is a strong starting point for clean business forecasting data."
        )

        if LIGHTWEIGHT_DEPLOYMENT:
            recommended_models = ["RandomForestRegressor (Lag Features)"]
            best_model_hint = lightweight_feature_message("Time-series training")
    else:
        target_series = df[target_column]

        if (
            pd.api.types.is_object_dtype(target_series)
            or pd.api.types.is_categorical_dtype(target_series)
            or target_series.nunique(dropna=True) < 20
        ):
            problem_type = "Classification"
            recommended_models = ["LogisticRegression", "RandomForest", "XGBoost"]
            feature_engineering = [
                "Normalize numeric features",
                "Encode categorical variables",
                "Remove highly correlated features",
            ]
            best_model_hint = (
                "RandomForest or XGBoost usually handle mixed tabular features well."
            )

            if LIGHTWEIGHT_DEPLOYMENT:
                recommended_models = ["LogisticRegression", "RandomForest"]
                best_model_hint = "RandomForest is a safer lightweight deployment choice for tabular classification."
        else:
            problem_type = "Regression"
            recommended_models = [
                "LinearRegression",
                "RandomForestRegressor",
                "XGBoostRegressor",
            ]
            feature_engineering = [
                "Scale skewed numeric features",
                "Encode categorical variables",
                "Remove highly correlated features",
            ]
            best_model_hint = (
                "XGBoostRegressor is often a strong baseline for tabular regression."
            )

            if LIGHTWEIGHT_DEPLOYMENT:
                recommended_models = [
                    "LinearRegression",
                    "RandomForestRegressor",
                ]
                best_model_hint = "RandomForestRegressor is a safer lightweight deployment choice for tabular regression."

    missing = int(df.isnull().sum().sum())
    issues = []

    if missing > 0:
        fix = "Handle missing values using imputation"
        issues.append("Missing values detected")
    else:
        fix = "No missing values detected"

    recommendations["problem_type"] = problem_type
    recommendations["recommended_models"] = recommended_models
    recommendations["best_model_hint"] = best_model_hint
    recommendations["missing_values"] = missing
    recommendations["fix"] = fix
    recommendations["feature_engineering"] = feature_engineering

    if date_column:
        recommendations["date_column"] = date_column

    if problem_type == "Classification":
        counts = df[target_column].value_counts(dropna=True)
        if len(counts) > 1 and counts.min() / max(counts.max(), 1) < 0.5:
            recommendations["imbalance"] = "Imbalanced dataset"
            issues.append("Class imbalance detected")
        else:
            recommendations["imbalance"] = "Balanced dataset"

    if issues:
        recommendations["issues"] = issues

    return recommendations


def train_single_model(name, model, X_train, X_test, y_train, y_test, problem_type):

    start = datetime.now()

    try:
        if name == "TabTransformer":
            torch_runtime = get_torch_runtime()
            torch = torch_runtime["torch"]
            nn = torch_runtime["nn"]
            device = torch_runtime["device"]

            X_torch = torch.tensor(X_train.values, dtype=torch.float32).to(device)
            y_torch = torch.tensor(y_train.values, dtype=torch.long).to(device)

            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(X_torch)
                loss = loss_fn(outputs, y_torch)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = model(
                    torch.tensor(X_test.values, dtype=torch.float32).to(device)
                )
                preds = torch.argmax(preds, dim=1).cpu().numpy()

            acc = accuracy_score(y_test, preds)

            score = acc
            metrics = {"accuracy": acc}

        else:
            params = auto_tune_model(name, model, X_train, y_train, problem_type)

            if params:
                model.set_params(**params)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
                metrics = {"accuracy": score}
            else:
                r2 = r2_score(y_test, preds)
                mse = mean_squared_error(y_test, preds)

                score = r2
                metrics = {"r2": r2, "mse": mse}

                if np.isnan(score):
                    score = -999

        end = datetime.now()
        duration = (end - start).total_seconds()

        return {
            "model": name,
            "score": float(score),
            "trained_model": model,
            "time": duration,
            "metrics": metrics,
        }

    except Exception as e:
        return {"model": name, "score": -999, "error": str(e), "trained_model": None}


def filter_models(models, X, y, problem_type):

    selected_models = {}

    num_rows = len(X)
    num_cols = X.shape[1]

    for name, model in models.items():
                                       
        if "SVM" in name and num_rows > 20000:
            continue

                                              
        if "KNN" in name and num_cols > 50:
            continue

                                                        
        if "LogisticRegression" in name and num_cols > 100:
            continue

                                                   
        if problem_type == "classification" and "LinearRegression" in name:
            continue

                                       
        if "SVR" in name and num_rows > 15000:
            continue

                      
        selected_models[name] = model

    return selected_models


                                                       
                                     
                                                       
def generate_ai_insights(df, problem_type, best_model_name, best_score):

    insights = []

                          
    if len(df) < 500:
        insights.append(
            "Dataset is small. Consider adding more data for better performance."
        )
    elif len(df) > 50000:
        if LIGHTWEIGHT_DEPLOYMENT:
            insights.append(
                "Large dataset detected. Use sampling or simpler models in lightweight deployment."
            )
        else:
            insights.append(
                "Large dataset detected. Consider using deep learning models."
            )

                     
    if df.shape[1] > 50:
        insights.append(
            "High number of features. Feature selection may improve performance."
        )

                    
    missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_ratio > 0.2:
        insights.append("High missing values detected. Data cleaning is recommended.")

                       
    if problem_type == "classification":
        if best_score > 0.9:
            insights.append(f"{best_model_name} is performing excellently.")
        elif best_score > 0.75:
            insights.append(
                f"{best_model_name} is performing well, but tuning may improve results."
            )
        else:
            insights.append(
                "Model performance is low. Try feature engineering or more data."
            )

    else:
        if best_score > 0.85:
            insights.append(f"{best_model_name} has strong predictive power.")
        elif best_score > 0.6:
            insights.append("Model is moderate. Consider tuning or better features.")
        else:
            insights.append(
                "Poor regression performance. Try different models or transformations."
            )

                        
    insights.append("Try ensemble models for better accuracy.")
    insights.append("Hyperparameter tuning can further improve performance.")

    return insights


                                                       
                            
                                                       
def auto_tune_model(name, model, X_train, y_train, problem_type):

    try:
                       
        if "RandomForest" in name:
            tune_random_forest = get_tune_random_forest()
            return tune_random_forest(X_train, y_train, problem_type)

                             
        if "LogisticRegression" in name:
            return {
                "C": random.choice([0.1, 1, 10]),
                "max_iter": random.choice([100, 200]),
            }

             
        if "SVM" in name:
            return {
                "C": random.choice([0.1, 1, 10]),
                "kernel": random.choice(["linear", "rbf"]),
            }

             
        if "KNN" in name:
            return {"n_neighbors": random.choice([3, 5, 7, 9])}

             
        if "SVR" in name:
            return {"C": random.choice([0.1, 1, 10])}

        return {}

    except:
        return {}


                                                       
                               
                                                       
def genetic_model_search(X_train, X_test, y_train, y_test, problem_type):
    classification_models, regression_models = get_model_library()

    if problem_type == "classification":
        models = classification_models
    else:
        models = regression_models

    best_model = None
    best_score = None
    best_metrics = None
    top_models = []

    for name, model in models.items():
        if "RandomForest" in name:
            tune_random_forest = get_tune_random_forest()
            best_params = tune_random_forest(X_train, y_train, problem_type)

            model.set_params(**best_params)

        model.fit(X_train, y_train)

        if problem_type == "classification":
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            metrics = {"accuracy": acc}

            score = acc
            better = best_score is None or score > best_score

        else:
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            metrics = {"mse": mse, "r2": r2}

            score = r2
            better = best_score is None or score > best_score

        if better:
            best_score = score
            best_model = model
            best_metrics = metrics

        model_info = {"model": name, **{k: round(v, 4) for k, v in metrics.items()}}

        top_models.append(model_info)

    return best_model, best_metrics, top_models


                                                       
                        
                                                       
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

        return {"model_strength": level, "accuracy": round(acc, 4)}

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

        return {"model_strength": level, "r2_score": round(r2, 4)}


                                                       
                           
                                                       
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


                                                       
           
                                                       
def build_simple_cnn(num_classes):
    torch_runtime = get_torch_runtime()
    nn = torch_runtime["nn"]
    vision_models = torch_runtime["models"]

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()

            self.model = vision_models.resnet18(
                weights=vision_models.ResNet18_Weights.DEFAULT
            )

            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.layer4.parameters():
                param.requires_grad = True

            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.model(x)

    return SimpleCNN(num_classes)


def generate_gradcam(model, image_tensor, target_class):
    torch = get_torch_runtime()["torch"]

    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

                                               
    target_layer = model.model.layer4[-1]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    loss = output[:, target_class]

    model.zero_grad()
    loss.backward()

    grad = gradients[0]
    act = activations[0]

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()

    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()

    cam = cam.detach().cpu().numpy()

    handle_forward.remove()
    handle_backward.remove()

    return cam


                                                       
                      
                                                       
def universal_image_organizer(root_folder):

    normalized_folder = os.path.join(root_folder, "__normalized__")

                                            
    if os.path.exists(normalized_folder):
        shutil.rmtree(normalized_folder)

    os.makedirs(normalized_folder, exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    for current_root, dirs, files in os.walk(root_folder):
                                       
        if "__normalized__" in current_root:
            continue

        for file in files:
            if (file or "").lower().endswith(image_extensions):
                full_path = os.path.join(current_root, file)

                                                         
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

                                         
                if os.path.abspath(full_path) != os.path.abspath(destination):
                    shutil.copy2(full_path, destination)

                                                
    for item in os.listdir(root_folder):
        if item == "__normalized__":
            continue

        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

                             
    for class_name in os.listdir(normalized_folder):
        shutil.move(
            os.path.join(normalized_folder, class_name),
            os.path.join(root_folder, class_name),
        )

    shutil.rmtree(normalized_folder)


                                                       
                          
                                                       
@app.post("/preview-columns")
async def preview_columns(file: UploadFile = File(...)):
    filename = file.filename if file.filename is not None else "unknown"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    filename_safe = file.filename if file.filename is not None else ""
    if filename_safe.endswith(".csv"):
        df = pd.read_csv(file_path, nrows=5)
    elif filename_safe.endswith(".xlsx"):
        df = pd.read_excel(file_path, nrows=5)
    else:
        return {"error": "Unsupported format"}

    return {"columns": list(df.columns), "preview": df.head().to_dict(orient="records")}


                                                       
                       
                                                       
def dataset_quality_score(df):

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()

    missing_ratio = missing_cells / total_cells

    numeric_ratio = len(df.select_dtypes(include=["int64", "float64"]).columns) / max(
        1, len(df.columns)
    )

    score = 100

                             
    score -= missing_ratio * 40

                                  
    if len(df) < 100:
        score -= 20
    elif len(df) < 500:
        score -= 10

                                
    score += numeric_ratio * 10

    score = max(0, min(100, round(score, 2)))

    return {
        "quality_score": score,
        "missing_ratio": round(missing_ratio, 4),
        "numeric_feature_ratio": round(numeric_ratio, 4),
    }


@app.post("/signup")
async def signup(user: UserSignup):
    try:
        if users_collection is None:
            return {"error": "Database not connected"}

                        
        existing = users_collection.find_one({"email": user.email})

        if existing:
            return {"error": "User already exists"}

                
        hashed_password = hash_password(user.password)
        user_id = str(uuid.uuid4())

        users_collection.insert_one(
            {
                "user_id": user_id,
                "email": user.email,
                "password": hashed_password,
                "role": "user",
                "name": "",
                "phone": "",
                "dob": "",
                "profile_pic": "",
                "created_at": datetime.utcnow().isoformat(),
            }
        )

        track_usage_event(user_id, "signup", {"email": user.email})

        return {"message": "Signup successful", "user_id": user_id}

    except Exception as e:
        print("❌ SIGNUP ERROR:", e)
        return {"error": str(e)}


@app.post("/login")
async def login(data: UserLogin):
    try:
        if users_collection is None:
            return {"error": "Database not connected"}

        user = users_collection.find_one({"email": data.email})

        if not user or not verify_password(data.password, user["password"]):
            return {"error": "Invalid credentials"}

        track_usage_event(user["user_id"], "login")

        access_token = create_access_token({"sub": user["email"]})

        return {
            "access_token": access_token,
            "token_type": "bearer"
        }

    except Exception as e:
        print("❌ LOGIN ERROR:", e)
        return {"error": str(e)}


@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    try:
        user = users_collection.find_one(
            {"user_id": user_id},
            {"_id": 0, "password": 0},
        )
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@app.post("/update-profile")
async def update_profile(data: dict, request: Request):
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    user_id = (data.get("user_id") or "").strip() or extract_user_id_from_request(
        request
    )

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    updates = {
        "name": data.get("name", ""),
        "phone": data.get("phone", ""),
        "dob": data.get("dob", ""),
        "profile_pic": data.get("profile_pic", ""),
    }

    try:
        result = users_collection.update_one({"user_id": user_id}, {"$set": updates})
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    track_usage_event(user_id, "update_profile")

    return {"message": "Profile updated"}


@app.post("/subscribe")
async def subscribe(data: dict, request: Request):
    user_id = (data.get("user_id") or "").strip() or extract_user_id_from_request(
        request
    )
    plan = (data.get("plan") or "free").strip().lower()

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    if plan not in {"free", "pro", "enterprise"}:
        raise HTTPException(status_code=400, detail="Invalid plan")

    try:
        subscriptions_collection.update_one(
            {"user_id": user_id},
            {"$set": {"plan": plan, "updated_at": datetime.utcnow().isoformat()}},
            upsert=True,
        )
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    track_usage_event(user_id, "subscribe", {"plan": plan})

    return {"message": f"Subscribed to {plan}", "plan": plan}


@app.get("/get-plan/{user_id}")
async def get_plan(user_id: str):
    try:
        subscription = subscriptions_collection.find_one(
            {"user_id": user_id}, {"_id": 0}
        )
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    if not subscription:
        return {"plan": "free"}

    return {"plan": subscription.get("plan", "free")}


@app.post("/create-team")
async def create_team(data: dict, request: Request):
    owner_id = (data.get("user_id") or "").strip() or extract_user_id_from_request(
        request
    )

    if not owner_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    team_id = str(uuid.uuid4())
    name = (data.get("name") or "My Team").strip() or "My Team"

    try:
        teams_collection.insert_one(
            {
                "team_id": team_id,
                "name": name,
                "owner": owner_id,
                "members": [owner_id],
                "created_at": datetime.utcnow().isoformat(),
            }
        )
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    track_usage_event(owner_id, "create_team", {"team_id": team_id})

    return {"team_id": team_id, "name": name}


@app.post("/invite")
async def invite(data: dict, request: Request):
    requester_id = (
        extract_user_id_from_request(request) or (data.get("owner_id") or "").strip()
    )
    team_id = (data.get("team_id") or "").strip()
    member_id = (data.get("member_user_id") or data.get("user_id") or "").strip()

    if not requester_id or not team_id or not member_id:
        raise HTTPException(status_code=400, detail="team_id and user_id are required")

    try:
        team = teams_collection.find_one({"team_id": team_id})
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    if team.get("owner") != requester_id and not is_admin(requester_id):
        raise HTTPException(
            status_code=403, detail="Only team owners or admins can invite users"
        )

    try:
        teams_collection.update_one(
            {"team_id": team_id},
            {"$addToSet": {"members": member_id}},
        )
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    track_usage_event(
        requester_id,
        "invite_to_team",
        {"team_id": team_id, "member_user_id": member_id},
    )

    return {"message": "User added"}


@app.get("/teams/{user_id}")
async def get_teams(user_id: str, request: Request):
    requester_id = extract_user_id_from_request(request)

    if requester_id and requester_id != user_id and not is_admin(requester_id):
        raise HTTPException(status_code=403, detail="Not allowed to view these teams")

    try:
        teams = list(
            teams_collection.find(
                {"members": user_id},
                {"_id": 0},
            )
        )
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    return {"teams": teams}


@app.get("/admin-stats")
async def admin_stats(request: Request):
    require_admin_user(request)

    try:
        total_users = users_collection.count_documents({})
        total_models = models_collection.count_documents({})
        total_teams = teams_collection.count_documents({})
        total_usage = usage_collection.count_documents({})
        total_subscriptions = subscriptions_collection.count_documents({})
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    return {
        "users": total_users,
        "models": total_models,
        "teams": total_teams,
        "usage_events": total_usage,
        "subscriptions": total_subscriptions,
    }


@app.post("/track-usage")
async def track_usage(data: dict, request: Request):
    user_id = (data.get("user_id") or "").strip() or extract_user_id_from_request(
        request
    )
    action = (data.get("action") or "").strip()

    if not user_id or not action:
        raise HTTPException(status_code=400, detail="user_id and action are required")

    metadata = data.get("metadata")
    track_usage_event(user_id, action, metadata)

    return {"message": "Tracked"}


                                                       
            
                                                       
@app.post("/train")
async def auto_train(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(None),
):
    filename = (file.filename if file.filename is not None else "").lower()
    user_id = extract_user_id_from_request(request)

                                                
                   
                                                
    if filename.endswith(".zip"):
        if LIGHTWEIGHT_DEPLOYMENT:
            return {"error": lightweight_feature_message("Image training")}

        try:
            torch_runtime = get_torch_runtime()
        except RuntimeError as exc:
            return {"error": str(exc)}

        torch = torch_runtime["torch"]
        nn = torch_runtime["nn"]
        DataLoader = torch_runtime["DataLoader"]
        datasets = torch_runtime["datasets"]
        device = torch_runtime["device"]
        transforms = torch_runtime["transforms"]

        zip_path = os.path.join(UPLOAD_FOLDER, file.filename or "unknown")
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        shutil.rmtree(IMAGE_DATASET_FOLDER, ignore_errors=True)
        os.makedirs(IMAGE_DATASET_FOLDER)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(IMAGE_DATASET_FOLDER)

        universal_image_organizer(IMAGE_DATASET_FOLDER)

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = datasets.ImageFolder(IMAGE_DATASET_FOLDER, transform=transform)

        if len(dataset.classes) < 2:
            return {"error": "Dataset must contain at least 2 classes"}

        if len(dataset) < 20:
            return {"error": "Dataset too small for training"}

        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = build_simple_cnn(len(dataset.classes)).to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            list(model.model.layer4.parameters()) + list(model.model.fc.parameters()),
            lr=0.001,
        )

        training_progress["progress"] = 0
        total_epochs = 5

        for epoch in range(total_epochs):
            model.train()

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

            training_progress["value"] = int(((epoch + 1) / total_epochs) * 100)

        training_progress["value"] = 100

        torch.save(
            {"model_state_dict": model.state_dict(), "classes": dataset.classes},
            CNN_MODEL_PATH,
        )

        joblib.dump(
            {
                "problem_type": "image_classification",
                "dataset_type": "image",
                "model_name": model.__class__.__name__,
                "classes": dataset.classes,
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            },
            MODEL_PATH,
        )

        leaderboard_data = save_leaderboard_snapshot(
            best_model_name=model.__class__.__name__,
            model_version=os.path.basename(MODEL_PATH),
            models=[{"model": model.__class__.__name__, "score": None, "time": None}],
            dataset_type="image",
            problem_type="image_classification",
        )

        experiment = append_experiment_log(
            best_model_name=model.__class__.__name__,
            leaderboard_models=leaderboard_data["models"],
            model_version=os.path.basename(MODEL_PATH),
            user_id=user_id,
            problem_type="image_classification",
            dataset_type="image",
            rows=len(dataset),
            columns=len(dataset.classes),
        )
        save_model_record(
            user_id=user_id,
            model_name=model.__class__.__name__,
            model_version=os.path.basename(MODEL_PATH),
            dataset_type="image",
        )
        track_usage_event(
            user_id, "train_image_model", {"model_name": model.__class__.__name__}
        )

        return {
            "dataset_type": "image",
            "problem_type": "Image Classification",
            "best_model": model.__class__.__name__,
            "classes": dataset.classes,
            "samples": len(dataset),
            "model_version": os.path.basename(MODEL_PATH),
            "leaderboard": leaderboard_data["models"],
            "experiment": experiment,
        }

                                                
                           
                                                
    if filename.endswith(".csv") or filename.endswith(".xlsx"):
        update_progress(5, "Loading dataset", "Dataset loading started...")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename or "unknown")

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, sep=None, engine="python")
            except Exception:
                return {"error": "Failed to read CSV file"}

        else:
            try:
                df = pd.read_excel(file_path)
            except Exception:
                return {"error": "Failed to read Excel file"}

        df.columns = df.columns.str.strip()
        update_progress(10, "Dataset Loaded", "Dataset successfully loaded")

        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")

        if LIGHTWEIGHT_DEPLOYMENT and len(df) > MAX_LIGHTWEIGHT_ROWS:
            return {
                "error": (
                    f"Dataset too large for lightweight deployment. Limit is "
                    f"{MAX_LIGHTWEIGHT_ROWS} rows."
                )
            }

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

                                                    
                               
                                                    

        date_column = None

        for col in categorical_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            try:
                converted = pd.to_datetime(df[col], errors="coerce", format="mixed")
                valid_ratio = pd.notna(converted).sum() / max(len(df), 1)
                unique_ratio = pd.Series(converted).nunique() / max(len(df), 1)

                if (
                    valid_ratio > 0.8
                    and unique_ratio > 0.5
                    and any(k in col.lower() for k in ["date", "time", "year", "month"])
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
                try:
                    Prophet = get_prophet_class()
                except RuntimeError as exc:
                    return {"error": str(exc)}

                models = {}

                for col in numeric_columns:
                    ts_df = df[[date_column, col]].dropna().copy()

                    ts_df.columns = ["ds", "y"]

                    if len(ts_df) < 10:
                        continue

                    model = Prophet()

                    model.fit(ts_df)

                    models[col] = model

                if len(models) > 0:
                    forecast_output = {}

                    for col, model in models.items():
                        future = model.make_future_dataframe(periods=10)
                        forecast = model.predict(future)
                        future_only = forecast.tail(10)[["ds", "yhat"]]
                        forecast_output[col] = future_only.to_dict(orient="records")

                    joblib.dump(
                        {
                            "models": models,
                            "problem_type": "time_series_multi",
                            "date_column": date_column,
                            "target_columns": list(models.keys()),
                        },
                        MODEL_PATH,
                    )

                    leaderboard_data = save_leaderboard_snapshot(
                        best_model_name="Prophet",
                        model_version=os.path.basename(MODEL_PATH),
                        models=[{"model": "Prophet", "score": None, "time": None}],
                        dataset_type="time_series",
                        problem_type="time_series_multi",
                    )

                    experiment = append_experiment_log(
                        best_model_name="Prophet",
                        leaderboard_models=leaderboard_data["models"],
                        model_version=os.path.basename(MODEL_PATH),
                        user_id=user_id,
                        problem_type="time_series_multi",
                        dataset_type="time_series",
                        rows=len(df),
                        columns=len(df.columns),
                    )
                    save_model_record(
                        user_id=user_id,
                        model_name="Prophet",
                        model_version=os.path.basename(MODEL_PATH),
                        dataset_type="time_series",
                    )
                    track_usage_event(
                        user_id, "train_time_series_model", {"model_name": "Prophet"}
                    )

                    return {
                        "dataset_type": "time_series",
                        "problem_type": "Time-Series (Universal)",
                        "best_model": "Prophet",
                        "date_column": date_column,
                        "target_columns": list(models.keys()),
                        "forecast": forecast_output,
                        "rows": len(df),
                        "message": "Model trained successfully. Use /predict to forecast.",
                        "model_version": os.path.basename(MODEL_PATH),
                        "leaderboard": leaderboard_data["models"],
                        "experiment": experiment,
                    }

                                                    
                               
                                                    

        if target_column is None:
            return {"error": "Provide target_column for tabular dataset"}

        if target_column not in df.columns:
            return {
                "error": "Invalid target column",
                "available_columns": list(df.columns),
            }

        X = df.drop(columns=[target_column])
        y = df[target_column]
        update_progress(15, "Feature Engineering", "Generating features...")
        try:
            X = run_auto_feature_engineering(X)
            update_progress(25, "Feature Engineering Done", "Features created")
        except Exception as e:
            return {"error": f"Feature engineering failed: {str(e)}"}

                             
        problem_type = detect_training_problem_type(df, target_column)

        if problem_type == "classification":
            y = LabelEncoder().fit_transform(y)

                               
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if len(numeric_cols) > 0:
            X[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(
                X[numeric_cols]
            )
            X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

                                   
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        for col in categorical_cols:
            X[col] = X[col].astype(str)

            X[col] = X[col].fillna("missing")

            X[col] = LabelEncoder().fit_transform(X[col])

        X = X.fillna(0)
        update_progress(30, "Splitting Data", "Preparing train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

                                    
                                
                                    

        dataset_info = analyze_training_dataset(X, y)

        recommended = recommend_training_models(dataset_info, problem_type)

        if LIGHTWEIGHT_DEPLOYMENT:
            recommended = [
                name for name in recommended if name not in LIGHTWEIGHT_BLOCKED_MODELS
            ]

        classification_models, regression_models = get_model_library()

        if problem_type == "classification":
            base_models = classification_models
        else:
            base_models = regression_models

        models = {}

        for name in recommended:
            if name in base_models:
                models[name] = base_models[name]

        models = filter_models(models, X, y, problem_type)

        update_progress(log=f"{len(models)} models selected after filtering")

                                        
        if "TabTransformer" in recommended and problem_type == "classification":
            try:
                TabTransformer = get_tab_transformer_class()
                models["TabTransformer"] = TabTransformer(
                    input_dim=X_train.shape[1], num_classes=len(np.unique(y_train))
                )
            except RuntimeError as exc:
                update_progress(log=str(exc))

        if not models:
            return {
                "error": "No lightweight-compatible models are available for this dataset."
            }

        results = []
        best_model = None
        best_score = -999
        update_progress(40, "Training Models", "Starting model training...")

        futures = []

        start_time = datetime.now()
        max_workers = 2 if LIGHTWEIGHT_DEPLOYMENT else 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for name, model in models.items():
                futures.append(
                    executor.submit(
                        train_single_model,
                        name,
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        problem_type,
                    )
                )

            for future in as_completed(futures):
                result = future.result()

                name = result["model"]
                score = result["score"]
                trained_model = result["trained_model"]

                update_progress(
                    status=f"Completed {name}",
                    log=f"{name} finished with score {round(score, 4)}",
                )

                results.append(
                    {
                        "model": name,
                        "score": score,
                        "time": result.get("time", 0),
                        "metrics": result.get("metrics", {}),
                    }
                )

                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / max(len(results), 1)
                remaining = len(models) - len(results)
                eta_seconds = int(avg_time * remaining)

                update_progress(eta=f"{eta_seconds} sec remaining")

                if score > best_score:
                    best_score = score
                    best_model = trained_model

        top_models = sorted(results, key=lambda x: x["score"], reverse=True)
        for i, model in enumerate(top_models):
            model["rank"] = i + 1
        best_model_name = top_models[0]["model"]
        update_progress(85, "Finalizing", "Selecting best model...")

                                  
        existing_models = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_v")]

        version = len(existing_models) + 1

        model_filename = f"model_v{version}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)

        leaderboard_data = save_leaderboard_snapshot(
            best_model_name=best_model_name,
            model_version=model_filename,
            models=top_models,
            dataset_type="tabular",
            problem_type=problem_type,
        )

        joblib.dump(
            {
                "model": best_model,
                "feature_columns": X.columns.tolist(),
                "problem_type": problem_type,
            },
            model_path,
        )

        joblib.dump(
            {
                "model": best_model,
                "feature_columns": X.columns.tolist(),
                "problem_type": problem_type,
            },
            MODEL_PATH,
        )

        update_progress(95, "Saving", "Saving model and reports...")

                                    
                             
                                    

        experiment = append_experiment_log(
            best_model_name=best_model_name,
            leaderboard_models=leaderboard_data["models"],
            model_version=model_filename,
            user_id=user_id,
            problem_type=problem_type,
            dataset_type="tabular",
            rows=len(df),
            columns=len(df.columns),
            score=float(best_score),
        )
        save_model_record(
            user_id=user_id,
            model_name=best_model_name,
            model_version=model_filename,
            dataset_type="tabular",
            score=float(best_score),
        )
        track_usage_event(
            user_id,
            "train_tabular_model",
            {
                "model_name": best_model_name,
                "problem_type": problem_type,
                "score": float(best_score),
            },
        )
                                    
                         
                                    
        quality = dataset_quality_score(df)

                                    
                        
                                    
        strength = model_strength_summary(
            problem_type,
            {
                "accuracy": best_score if problem_type == "classification" else None,
                "r2": best_score if problem_type == "regression" else None,
            },
        )

                                    
                          
                                    
        explanation = generate_explanation_text(problem_type, strength)

                                    
                     
                                    
        report = {
            "dataset_quality": quality,
            "model_strength": strength,
            "explanation": explanation,
        }

        joblib.dump(report, TRAINING_REPORT_PATH)

        ai_insights = generate_ai_insights(
            df, problem_type, type(best_model).__name__, best_score
        )

        generated_code = generate_training_pipeline_code(
            model_name=type(best_model).__name__,
            feature_columns=X.columns.tolist(),
            problem_type=problem_type,
            target_column=target_column,
        )

        with open(GENERATED_PIPELINE_PATH, "w") as f:
            f.write(generated_code)

        if problem_type == "classification":
            update_progress(
                100, "Completed", "Training completed successfully", eta="0 sec"
            )

            return {
                "dataset_type": "tabular",
                "problem_type": "Classification",
                "rows": len(df),
                "target_column": target_column,
                "best_model": best_model_name,
                "accuracy": round(best_score, 4),
                "top_models": top_models,
                "leaderboard": leaderboard_data["models"],
                "generated_code": generated_code,
                "model_version": model_filename,
                "ai_insights": ai_insights,
                "experiment": experiment,
            }

        else:
            mse = mean_squared_error(y_test, best_model.predict(X_test))
            update_progress(
                100, "Completed", "Training completed successfully", eta="0 sec"
            )

            return {
                "dataset_type": "tabular",
                "problem_type": "Regression",
                "rows": len(df),
                "target_column": target_column,
                "best_model": best_model_name,
                "mse": round(mse, 4),
                "r2": round(best_score, 4),
                "top_models": top_models,
                "leaderboard": leaderboard_data["models"],
                "generated_code": generated_code,
                "model_version": model_filename,
                "ai_insights": ai_insights,
                "experiment": experiment,
            }

    return {"error": "Unsupported dataset"}


                                                       
               
                                                       
def risk_analysis(prob_array):

    confidence = float(np.max(prob_array))

    if confidence > 0.85:
        risk = "Low Risk"
    elif confidence > 0.65:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return {"confidence": round(confidence, 4), "risk_level": risk}


                                                       
                               
                                                       
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH):
        return {"error": "No trained model found"}

    model_package = joblib.load(MODEL_PATH)
    problem_type = model_package.get("problem_type")

    if problem_type == "image_classification":
        return {"error": "Latest model is an image model. Use /predict-image instead."}

                                                
                            
                                                
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
            "forecast": forecast_output,
        }

                                                
                        
                                                
    model = model_package.get("model")
    feature_columns = model_package.get("feature_columns")

    if model is None or feature_columns is None:
        return {"error": "Invalid tabular model package"}

    try:
                          
        if (file.filename or "").endswith(".csv"):
            df = pd.read_csv(file.file, sep=None, engine="python")
        elif (file.filename or "").endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            return {"error": "Unsupported file format"}

        df.columns = df.columns.str.strip().str.replace(r"[^\w]+", "_", regex=True)

                                   
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            return {
                "error": "Missing required columns",
                "missing_columns": missing_cols,
            }

        df = df[feature_columns]

                          
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

                              
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])

        df = df.fillna(0)

                                                    
                                                    
                                                    
        if problem_type == "classification":
            predictions = model.predict(df)

            probabilities = (
                model.predict_proba(df) if hasattr(model, "predict_proba") else None
            )

            results = []

            for i in range(len(predictions)):
                prediction = int(predictions[i])

                if probabilities is not None:
                    prob_array = probabilities[i]

                                                     
                    risk_info = risk_analysis(prob_array)

                    prob_dict = {
                        str(idx): round(float(prob), 4)
                        for idx, prob in enumerate(prob_array)
                    }

                    results.append(
                        {
                            "prediction": prediction,
                            "confidence": risk_info["confidence"],
                            "risk_level": risk_info["risk_level"],
                            "probabilities": prob_dict,
                        }
                    )

                else:
                    results.append(
                        {
                            "prediction": prediction,
                            "confidence": None,
                            "risk_level": "Unknown",
                            "probabilities": None,
                        }
                    )

            return {
                "problem_type": "Classification",
                "num_predictions": len(results),
                "predictions": results,
            }

                                                    
                           
                                                    
        elif problem_type == "regression":
            predictions = model.predict(df)

            return {
                "problem_type": "Regression",
                "num_predictions": len(predictions),
                "predictions": [round(float(p), 4) for p in predictions],
            }

        else:
            return {"error": "Unknown problem type"}

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


                                                       
               
                                                       
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if LIGHTWEIGHT_DEPLOYMENT:
        raise HTTPException(
            status_code=503,
            detail=lightweight_feature_message("Image inference"),
        )

    if not os.path.exists(CNN_MODEL_PATH):
        return {"error": "No trained CNN model"}

    torch_runtime = get_torch_runtime()
    torch = torch_runtime["torch"]
    device = torch_runtime["device"]
    transforms = torch_runtime["transforms"]
    Image = get_image_module()

    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    model = build_simple_cnn(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(file.file).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    return {
        "predicted_class": classes[pred.item()],
        "confidence": float(confidence.item()),
    }


@app.post("/explain-image")
async def explain_image(file: UploadFile = File(...)):
    if LIGHTWEIGHT_DEPLOYMENT:
        raise HTTPException(
            status_code=503,
            detail=lightweight_feature_message("Image explanations"),
        )

    if not os.path.exists(CNN_MODEL_PATH):
        return {"error": "No trained CNN model"}

    torch_runtime = get_torch_runtime()
    torch = torch_runtime["torch"]
    device = torch_runtime["device"]
    transforms = torch_runtime["transforms"]
    Image = get_image_module()
    plt = get_matplotlib_pyplot()

    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    model = build_simple_cnn(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(file.file).convert("RGB")

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    cam = generate_gradcam(model, img_tensor, pred.item())

    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam).resize((224, 224))
    cam = np.array(cam)

    original = np.array(image.resize((224, 224)))

    heatmap = plt.cm.jet(cam)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)

    superimposed = heatmap * 0.4 + original
    superimposed = np.uint8(superimposed)

    final_img = Image.fromarray(superimposed)

    img_io = io.BytesIO()
    final_img.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/png")


                                                       
               
                                                       
@app.get("/model-explain")
def model_explain():

    if not os.path.exists(MODEL_PATH):
        return {"error": "No trained model found"}

    model_package = joblib.load(MODEL_PATH)

    problem_type = model_package.get("problem_type")

                                     
    if problem_type in ["time_series", "time_series_multi"]:
        return {"message": "Model explain not supported for Time-Series models"}

    if problem_type == "image_classification":
        return {"message": "Model explain not supported for image models"}

                    
    if "model" not in model_package:
        return {"error": "No tabular model found to explain"}

    model = model_package["model"]
    feature_columns = model_package.get("feature_columns", [])

                                      
    if hasattr(model, "feature_importances_"):
        importance = dict(
            sorted(
                zip(feature_columns, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        return {"model_type": type(model).__name__, "feature_importance": importance}

                                  
    elif hasattr(model, "coef_"):
        coef = model.coef_

        if len(coef.shape) > 1:
            coef = coef[0]

        importance = dict(
            sorted(zip(feature_columns, coef), key=lambda x: abs(x[1]), reverse=True)
        )

        return {"model_type": type(model).__name__, "coefficients": importance}

    else:
        return {"message": "This model type does not support explainability"}


                                                       
      
                                                       
@app.post("/shap-explain")
async def shap_explain(file: UploadFile = File(...)):

    if not os.path.exists(MODEL_PATH):
        return {"error": "No trained model found"}

    model_package = joblib.load(MODEL_PATH)

    if model_package["problem_type"] in ["time_series", "time_series_multi"]:
        return {"error": "SHAP not supported for time-series"}

    if model_package["problem_type"] == "image_classification":
        return {"error": "SHAP not supported for image models"}

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    try:
        shap = get_shap_module()

                         
        if (file.filename or "").endswith(".csv"):
            df = pd.read_csv(file.file, sep=None, engine="python")
        elif (file.filename or "").endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            return {"error": "Unsupported file format"}

        df = df[feature_columns]

                                         
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        df = df.fillna(0)

                          
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

                           
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        importance = np.abs(shap_values).mean(axis=0)

        feature_importance = dict(
            sorted(zip(feature_columns, importance), key=lambda x: x[1], reverse=True)
        )

                                        
        sample_values = (
            shap_values.values[0].tolist()
            if hasattr(shap_values, "values")
            else shap_values[0].tolist()
        )
        sample_explanation = dict(zip(feature_columns, sample_values))

        return {
            "feature_importance": feature_importance,
            "sample_explanation": sample_explanation,
        }

    except Exception as e:
        return {"error": str(e)}


                                                       
                              
                                                       
@app.get("/download-code/{format}")
def download_code(format: str):

    if not os.path.exists(GENERATED_PIPELINE_PATH):
        return {"error": "No generated code found. Train a model first."}

                                
                   
                                
    if format == "python":
        return FileResponse(GENERATED_PIPELINE_PATH, filename="pipeline.py")

                                
                      
                                
    elif format == "notebook":
        with open(GENERATED_PIPELINE_PATH, "r") as f:
            code_lines = f.read().split("\n")

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [line + "\n" for line in code_lines],
                    "outputs": [],
                    "execution_count": None,
                }
            ],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        notebook_path = GENERATED_NOTEBOOK_PATH

        with open(notebook_path, "w") as f:
            json.dump(notebook, f)

        return FileResponse(notebook_path)

                                
                      
                                
    elif format == "api":
        api_code = """
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
"""

        api_path = GENERATED_API_PATH

        with open(api_path, "w") as f:
            f.write(api_code)

        return FileResponse(api_path)

                                
                      
                                
    elif format == "requirements":
        requirements = """
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
"""

        req_path = GENERATED_REQUIREMENTS_PATH

        with open(req_path, "w") as f:
            f.write(requirements.strip())

        return FileResponse(req_path, filename="requirements.txt")

                                
                    
                                
    elif format == "docker":
        dockerfile = """
FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "generated_pipeline.py"]
"""

        with open(GENERATED_DOCKERFILE_PATH, "w") as f:
            f.write(dockerfile.strip())

                                   
        with open(GENERATED_REQUIREMENTS_PATH, "w") as f:
            f.write("pandas\nnumpy\nscikit-learn\njoblib\n")

        zip_path = DOCKER_PACKAGE_PATH

        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(GENERATED_PIPELINE_PATH, arcname="generated_pipeline.py")
            zipf.write(GENERATED_REQUIREMENTS_PATH, arcname="requirements.txt")
            zipf.write(GENERATED_DOCKERFILE_PATH, arcname="Dockerfile")

        return FileResponse(zip_path)

                                
                      
                                
    elif format == "project":
        zip_path = FULL_PROJECT_ZIP_PATH

        with zipfile.ZipFile(zip_path, "w") as zipf:
            if os.path.exists(GENERATED_PIPELINE_PATH):
                zipf.write(GENERATED_PIPELINE_PATH, arcname="generated_pipeline.py")

            if os.path.exists(MODEL_PATH):
                zipf.write(MODEL_PATH, arcname=os.path.basename(MODEL_PATH))

                              
            with open(GENERATED_REQUIREMENTS_PATH, "w") as f:
                f.write("pandas\nnumpy\nscikit-learn\njoblib\n")

            zipf.write(GENERATED_REQUIREMENTS_PATH, arcname="requirements.txt")

        return FileResponse(zip_path)

                                
                    
                                
    else:
        return {"error": "Format not supported"}


                                                       
             
                                                       
@app.get("/experiments")
def get_experiments(request: Request):
    user_id = extract_user_id_from_request(request)
    data = filter_experiments_for_user(load_experiment_logs(), user_id)
    normalized = [normalize_experiment_entry(item) for item in data]

    return {"total_experiments": len(normalized), "experiments": normalized[::-1]}


                                                       
               
                                                       
@app.get("/insights")
def get_insights(request: Request):
    user_id = extract_user_id_from_request(request)
    data = filter_experiments_for_user(load_experiment_logs(), user_id)

    if len(data) == 0:
        return {"error": "No experiments to analyze"}

    scored_experiments = [
        exp for exp in data if isinstance(exp.get("score"), (int, float))
    ]
    best_exp = (
        max(scored_experiments, key=lambda x: x["score"])
        if scored_experiments
        else data[-1]
    )
    scores = [exp["score"] for exp in scored_experiments]

    return {
        "best_model": best_exp["model_name"],
        "best_score": round(best_exp["score"], 4) if scores else None,
        "best_version": best_exp["model_version"],
        "total_experiments": len(data),
        "average_score": round(sum(scores) / len(scores), 4) if scores else None,
    }


                                                       
                            
                                                       
@app.get("/ai-insights")
def get_ai_insights(request: Request):
    user_id = extract_user_id_from_request(request)
    data = filter_experiments_for_user(load_experiment_logs(), user_id)

    if len(data) == 0:
        return {"error": "No experiments available"}

    latest = data[-1]

    insights = [
        f"Best model: {latest['model_name']}",
        f"Score: {round(latest['score'], 4) if isinstance(latest.get('score'), (int, float)) else 'N/A'}",
        f"Dataset rows: {latest.get('rows', 'N/A')}",
    ]

    return {"insights": insights}


                                                       
          
                                                       
@app.get("/download-model/{version}")
def download_model(version: str):

    safe_version = os.path.basename(version)
    model_path = os.path.join(MODEL_DIR, safe_version)

    if os.path.exists(model_path):
        return FileResponse(model_path)

    return {"error": "Model not found"}


@app.get("/download-model")
def download_best_model():
    if os.path.exists(MODEL_PATH):
        return FileResponse(MODEL_PATH, filename="best_model.pkl")
    return {"error": "No trained model"}


@app.get("/download-latest-model")
def download_latest_model():
    if os.path.exists(MODEL_PATH):
        return FileResponse(MODEL_PATH, filename="best_model.pkl")
    return {"error": "No trained model"}


@app.get("/leaderboard")
def get_leaderboard(request: Request):
    user_id = extract_user_id_from_request(request)
    user_experiments = filter_experiments_for_user(load_experiment_logs(), user_id)

    if user_experiments:
        latest_experiment = normalize_experiment_entry(user_experiments[-1])
        return build_leaderboard_payload(
            best_model_name=latest_experiment["best_model"],
            model_version=latest_experiment.get("model_version")
            or os.path.basename(MODEL_PATH),
            models=latest_experiment.get("leaderboard", []),
            dataset_type=latest_experiment.get("dataset_type"),
            problem_type=latest_experiment.get("problem_type"),
        )

    if not os.path.exists(LEADERBOARD_PATH):
        return {"error": "No leaderboard found. Train model first."}

    leaderboard = joblib.load(LEADERBOARD_PATH)

    if isinstance(leaderboard, list):
        best_model = leaderboard[0]["model"] if leaderboard else "N/A"
        return build_leaderboard_payload(
            best_model_name=best_model,
            model_version=os.path.basename(MODEL_PATH),
            models=leaderboard,
        )

    return leaderboard


@app.get("/training-report")
def get_training_report():

    if not os.path.exists(TRAINING_REPORT_PATH):
        return {"error": "No report found. Train model first."}

    report = joblib.load(TRAINING_REPORT_PATH)

    return report


@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(training_progress)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


@app.post("/analyze-dataset")
async def analyze_dataset_file(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file.file, sep=None, engine="python")
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)
    else:
        return {"error": "Unsupported file format"}

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }


@app.post("/auto-ml-insights")
async def auto_ml_insights(
    file: UploadFile = File(...), target_column: str = Form(...)
):
    filename = (file.filename or "").lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file.file, sep=None, engine="python")
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)
    else:
        return {"error": "Unsupported file format"}

    try:
        return auto_ml_recommendation(df, target_column)
    except ValueError as exc:
        return {"error": str(exc)}


@app.post("/generate-image")
async def generate_image(request: Request, data: dict = Body(...)):
    prompt = (data.get("prompt") or "").strip()
    user_id = extract_user_id_from_request(request)

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    client = require_openai_client()

    try:
        response = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size="1024x1024",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    image = response.data[0]
    image_url = getattr(image, "url", None)
    image_b64 = getattr(image, "b64_json", None)

    if image_url:
        track_usage_event(user_id, "generate_image", {"prompt_length": len(prompt)})
        return {"image_url": image_url}

    if image_b64:
        track_usage_event(user_id, "generate_image", {"prompt_length": len(prompt)})
        return {"image_url": f"data:image/png;base64,{image_b64}"}

    raise HTTPException(
        status_code=502, detail="Image generation returned no usable image output."
    )


@app.post("/adversarial-test")
async def adversarial_test(request: Request, file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    user_id = extract_user_id_from_request(request)

    if filename.endswith(".csv"):
        df = pd.read_csv(file.file, sep=None, engine="python")
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)
    else:
        return {"error": "Unsupported file format"}

    noisy_df = df.copy()
    numeric_columns = noisy_df.select_dtypes(include=["float", "int"]).columns.tolist()

    for column in numeric_columns:
        noise = np.random.normal(0, 0.1, len(noisy_df))
        noisy_df[column] = noisy_df[column] * (1 + noise)

    preview = (
        noisy_df.head(PREVIEW_RESPONSE_ROWS)
        .astype(object)
        .where(pd.notna(noisy_df.head(PREVIEW_RESPONSE_ROWS)), None)
        .to_dict(orient="records")
    )

    track_usage_event(user_id, "adversarial_test", {"rows": len(noisy_df)})

    return {
        "message": "Adversarial noise added",
        "rows": len(noisy_df),
        "numeric_columns": numeric_columns,
        "preview": preview,
    }


@app.post("/generate-code")
async def generate_code(request: Request, data: dict = Body(...)):
    task = (data.get("task") or "").strip()
    user_id = extract_user_id_from_request(request)

    if not task:
        raise HTTPException(status_code=400, detail="task is required")

    client = require_openai_client()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Generate clean ML Python code only. Avoid explanations unless asked.",
                },
                {"role": "user", "content": task},
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    generated_code = (response.choices[0].message.content or "").strip()

    if generated_code.startswith("```"):
        generated_code = generated_code.strip("`")
        generated_code = generated_code.replace("python\n", "", 1)

    track_usage_event(user_id, "generate_code", {"task_length": len(task)})

    return {"code": generated_code}


@app.post("/text-to-speech")
async def text_to_speech(request: Request, data: dict = Body(...)):
    text = (data.get("text") or "").strip()
    voice = (data.get("voice") or "coral").strip() or "coral"
    user_id = extract_user_id_from_request(request)

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    client = require_openai_client()

    try:
        audio_response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="mp3",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    audio_bytes = b""

    if hasattr(audio_response, "read"):
        audio_bytes = audio_response.read()
    elif hasattr(audio_response, "content"):
        audio_bytes = audio_response.content
    else:
        try:
            audio_bytes = bytes(audio_response)
        except Exception:
            audio_bytes = b""

    if not audio_bytes:
        raise HTTPException(
            status_code=502, detail="Text-to-speech returned no audio data."
        )

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    track_usage_event(
        user_id, "text_to_speech", {"voice": voice, "text_length": len(text)}
    )

    return {"audio_url": f"data:audio/mpeg;base64,{audio_b64}"}


@app.post("/video-ai")
async def video_ai(request: Request):
    track_usage_event(extract_user_id_from_request(request), "video_ai_placeholder")
    return {"message": "Video model coming soon"}


@app.post("/chat")
async def chat_ai(data: dict = Body(...)):
    message = data.get("message", "")
    dataset_info = data.get("dataset_info", "")
    client = get_openai_client()

    if client:
        prompt = f"""
User question: {message}

Dataset info:
{dataset_info}

Answer like a professional ML expert helping inside an AutoML dashboard.
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data scientist helping with datasets, models, training, and ML debugging.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return {"reply": response.choices[0].message.content}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    lowered = message.lower()

    if "accuracy" in lowered:
        reply = "Your model accuracy depends on data quality, features, class balance, and model choice. Try feature engineering or tuning."
    elif "best model" in lowered:
        reply = "The best model is usually the one with the highest leaderboard score on the latest run."
    elif "overfitting" in lowered:
        reply = "Overfitting happens when a model memorizes training data. Try regularization, simpler models, or more data."
    elif "dataset" in lowered:
        reply = "Dataset quality strongly impacts performance. Clean missing values, review target leakage, and remove noisy columns."
    else:
        reply = "I am your AI assistant. Ask me about models, training, datasets, or ML errors."

    return {"reply": reply}


@app.get("/")
def home():
    return {"status": "Backend is LIVE 🚀", "message": "AutoML API running"}
