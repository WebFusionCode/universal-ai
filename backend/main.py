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
from google import genai

try:
    from dotenv import load_dotenv
except Exception:

    def load_dotenv(*args, **kwargs):
        return False


from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import joblib
from jose import JWTError, jwt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ===== DEPLOYMENT SETTINGS =====
LIGHTWEIGHT_DEPLOYMENT = False   # for Render free tier
MAX_LIGHTWEIGHT_ROWS = 10000    # limit dataset size

# ===== DEFAULT SETTINGS =====
PREVIEW_RESPONSE_ROWS = 5

# ===== FILE PATHS =====
UPLOAD_FOLDER = "uploads"
EXPERIMENTS_DIR = "experiments"
EXPERIMENTS_PATH = os.path.join(EXPERIMENTS_DIR, "experiments.json")

# ===== MODEL PATHS =====
MODEL_DIR = "models"
TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_model.pkl")
TIME_SERIES_MODEL_PATH = os.path.join(MODEL_DIR, "time_series.pkl")
IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")
LAST_MODEL_TYPE_PATH = os.path.join(MODEL_DIR, "last_model.txt")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.pkl")
TRAINING_REPORT_PATH = os.path.join(MODEL_DIR, "training_report.pkl")

# ===== GENERATED FILES =====
GENERATED_PIPELINE_PATH = "generated_pipeline.py"
GENERATED_NOTEBOOK_PATH = "generated_notebook.ipynb"
GENERATED_API_PATH = "generated_api.py"
GENERATED_REQUIREMENTS_PATH = "requirements.txt"
GENERATED_DOCKERFILE_PATH = "Dockerfile"

# ===== IMAGE PROCESSING =====
IMAGE_DATASET_FOLDER = os.path.join(UPLOAD_FOLDER, "images")

# ===== PACKAGES =====
DOCKER_PACKAGE_PATH = "docker_package.zip"
FULL_PROJECT_ZIP_PATH = "full_project.zip"

# ===== LIGHTWEIGHT BLOCKED MODELS =====
LIGHTWEIGHT_BLOCKED_MODELS = ["TabTransformer", "Prophet", "CNN", "DeepLearning"]

# ===== OPENAI SETTINGS =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = "dall-e-3"

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

SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret")
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
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
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

load_dotenv(Path(__file__).resolve().parent / ".env")

MONGO_URL = os.getenv(
    "MONGO_URL",
    "mongodb+srv://webwithfusion_db_user:Harsh123@cluster0.fu0kdb2.mongodb.net/?appName=Cluster0",
)

if not MONGO_URL:
    print("❌ MONGO_URL missing")
else:
    print("✅ MONGO_URL found")

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    print(f"📡 Configuring Gemini with Key: {GEMINI_API_KEY[:6]}...{GEMINI_API_KEY[-4:]}")
    # New SDK Style (Modern V1)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1'})
    print("✅ Gemini AI Initialized (Modern V1 - Gemini 2.0)")
else:
    print("⚠️ GEMINI_API_KEY not found in .env")
    gemini_client = None

def safe_extract_text(response):
    """Safely extract text from Gemini response object."""
    try:
        if response.text:
            return response.text
        return "No text response generated."
    except Exception:
        return "AI response parsing failed."

def call_gemini_safely(prompt, original_query=""):
    """
    Robust AI caller with:
    1. Primary (2.0 Flash)
    2. 429 Retry (5s sleep)
    3. Fallback (1.5 Flash)
    4. Safe String Fallback
    """
    if not gemini_client:
        return "Gemini AI not configured."

    import time
    
    # Attempt 1: Gemini 2.0
    try:
        response = gemini_client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=prompt
        )
        return safe_extract_text(response)
    
    except Exception as e:
        err_msg = str(e).lower()
        print(f"⚠️ Gemini 2.0 Failed: {err_msg}")
        
        # Scenario: Rate Limit (429) -> Retry
        if "429" in err_msg:
            print("⏳ Rate limited. Retrying in 5 seconds...")
            time.sleep(5)
            try:
                response = gemini_client.models.generate_content(
                    model="models/gemini-2.0-flash",
                    contents=prompt
                )
                return safe_extract_text(response)
            except Exception as e2:
                print(f"⚠️ Retry Failed: {str(e2)}")

        # Scenario: Fallback to Gemini 2.0 Flash Lite
        print("💡 Attempting fallback to Gemini 2.0 Flash Lite...")
        try:
            response = gemini_client.models.generate_content(
                model="models/gemini-2.0-flash-lite",
                contents=prompt
            )
            return safe_extract_text(response)
        except Exception as e3:
            print(f"❌ Fallback Failed: {str(e3)}")
            
    # Final Fallback: Safe UI String
    return f"AI Assistant is currently calibrating. Your question was: '{original_query or '...'}'"


def ask_gemini(prompt):
    return call_gemini_safely(prompt, "Direct Insight Query")


def generate_model_summary(model_name, score, problem_type):
    if not gemini_client:
        return "AI summary not available"

    prompt = f"""
    Explain this ML model in simple student-friendly language:

    Model: {model_name}
    Score: {score}
    Type: {problem_type}

    Include:
    - What this model does
    - Why it is good
    - When to use it
    - Limitations
    """

    return call_gemini_safely(prompt, f"Explain {model_name}")

print("MONGO_URL:", MONGO_URL)

client = MongoClient(MONGO_URL)
db = client["automl_db"]
users_collection = db["users"]
models_collection = db["models"]
subscriptions_collection = db["subscriptions"]
teams_collection = db["teams"]
usage_collection = db["usage"]
experiments_collection = db["experiments"]

print("🚀 Starting FastAPI app...")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                     
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
print("✅ App created and uploads mounted")


def lightweight_feature_message(feature_name):
    return (
        f"{feature_name} is disabled in lightweight deployment mode to keep startup "
        "stable and memory usage low."
    )


@lru_cache(maxsize=1)
def get_openclient():
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    return OpenAI(api_key=OPENAI_API_KEY)


@lru_cache(maxsize=1)
def get_model_library():
    try:
        from models.model_library import CLASSIFICATION_MODELS, REGRESSION_MODELS  # type: ignore
    except ImportError:
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        CLASSIFICATION_MODELS = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=200),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
        }
        REGRESSION_MODELS = {
            "RandomForest": RandomForestRegressor(),
            "LinearRegression": LinearRegression(),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
        }

    return CLASSIFICATION_MODELS, REGRESSION_MODELS


def analyze_training_dataset(X, y):
    try:
        from utils.automl_brain import analyze_dataset  # type: ignore
        return analyze_dataset(X, y)
    except ImportError:
        return {"features": len(X.columns), "samples": len(X)}


def recommend_training_models(dataset_info, problem_type):
    try:
        from utils.automl_brain import recommend_models  # type: ignore
        return recommend_models(dataset_info, problem_type)
    except ImportError:
        return ["RandomForest"]


def run_auto_feature_engineering(df):
    try:
        from utils.feature_engineering import auto_feature_engineering  # type: ignore
        return auto_feature_engineering(df)
    except ImportError:
        return df


def detect_training_problem_type(df, target_column):
    try:
        from utils.problem_detection import detect_problem_type  # type: ignore
        return detect_problem_type(df, target_column)
    except ImportError:
        # Simple detection based on target type
        if df[target_column].dtype in ['int64', 'float64']:
            unique_vals = df[target_column].nunique()
            return "regression" if unique_vals > 10 else "classification"
        else:
            return "classification"


def generate_training_pipeline_code(model_name, feature_columns, problem_type, target_column):
    try:
        from utils.code_generator import generate_training_code  # type: ignore
        return generate_training_code(
            model_name=model_name,
            feature_columns=feature_columns,
            problem_type=problem_type,
            target_column=target_column,
        )
    except ImportError:
        return f"# Generated code for {model_name}\nprint('Training {model_name}')"


def get_tune_random_forest():
    try:
        from utils.hyperparameter_tuning import tune_random_forest  # type: ignore
        return tune_random_forest
    except ImportError:
        return lambda: {"n_estimators": 100, "max_depth": None}


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
        raise RuntimeError(f"Prophet failed to import: {str(exc)}")

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

    try:
        from models.transformer_models import TabTransformer  # type: ignore
    except ImportError:
        raise RuntimeError("TabTransformer not available")
    
    return TabTransformer


def require_openclient():
    client = get_openclient()

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


def update_progress(progress=None, status=None, message=None, **kwargs):
    global training_progress

    if progress is not None:
        training_progress["progress"] = progress

    if status is not None:
        training_progress["status"] = status

    if message is not None:
        training_progress["message"] = message

    # Optional extras (like log, eta etc.)
    for key, value in kwargs.items():
        training_progress[key] = value


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
        model_version=entry.get("model_version", "model.pkl"),
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
                    for keyword in ["date", "time", "year", "month", "day", "ds", "timestamp"]
                )
            ):
                date_column = col
                print("Detected date column:", date_column)
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
    print(f"Training {name} - X shape: {X_train.shape}, y unique: {np.unique(y_train) if problem_type == 'classification' else 'regression'}")

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

            from sklearn.metrics import mean_squared_error
            loss = 0.0

            if problem_type == "classification":
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
                score = accuracy_score(y_test, preds)
                try:
                    # Try to get probabilities for log_loss
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_test)
                        loss = log_loss(y_test, probs)
                    else:
                        loss = 0.0 # Fallback
                    
                    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
                    rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
                    f1   = f1_score(y_test, preds, average="weighted", zero_division=0)
                except Exception:
                    prec = rec = f1 = 0.0
                metrics = {
                    "accuracy": round(float(score), 4),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "f1": round(float(f1), 4),
                    "loss": round(float(loss), 4)
                }
            else:
                from sklearn.metrics import r2_score
                import math
                r2  = r2_score(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                rmse = math.sqrt(mse)
                mae  = float(np.mean(np.abs(np.array(y_test) - np.array(preds))))
                score = r2
                loss = mse
                metrics = {
                    "r2": round(float(r2), 4),
                    "mse": round(float(mse), 4),
                    "rmse": round(float(rmse), 4),
                    "mae": round(float(mae), 4),
                    "loss": round(float(mse), 4)
                }

                if np.isnan(score) or score is None:
                    score = 0.0
                score = float(score)

        end = datetime.now()
        duration = (end - start).total_seconds()

        return {
            "model": name,
            "score": float(score),
            "loss": float(loss),
            "trained_model": model,
            "time": duration,
            "metrics": metrics,
        }

    except Exception as e:
        print(f"❌ MODEL FAILED: {name} → {str(e)}")
        return {
            "model": name,
            "score": -999,
            "error": str(e),
            "trained_model": None
        }


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
    if users_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    existing = users_collection.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = hash_password(user.password)
    user_id = str(uuid.uuid4())

    users_collection.insert_one({
        "user_id": user_id,
        "email": user.email,
        "password": hashed_password,
        "role": "user"
    })

    return {
        "message": "Signup successful",
        "user_id": user_id
    }


@app.post("/login")
async def login(data: UserLogin):
    if users_collection is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    user = users_collection.find_one({"email": data.email})

    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Wrong password")

    access_token = create_access_token({"sub": user["email"]})

    return {
        "access_token": access_token,
        "user_id": user["user_id"],
        "token_type": "bearer"
    }


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


                                                       
            
                                                       

# ===== UNIVERSAL DATASET PIPELINE HELPERS =====

def is_image_zip(file):
    return file.filename.endswith(".zip")

def detect_dataset_type(df):
    info = {}
    if isinstance(df, str):  # zip path check
        return "image", {}

    info["rows"] = len(df)
    info["columns"] = list(df.columns)

    # Detect datetime columns
    date_cols = []
    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if converted.notna().sum() > len(df) * 0.7:
                date_cols.append(col)
        except:
            pass

    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Detect categorical
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # 🧠 DECISION LOGIC

    # TIME SERIES
    if len(date_cols) == 1 and len(numeric_cols) == 1:
        return "time_series", {
            "date_column": date_cols[0],
            "target_candidates": numeric_cols
        }

    # TABULAR
    if len(df.columns) >= 2:
        return "tabular", {
            "numeric": numeric_cols,
            "categorical": categorical_cols
        }

    return "unknown", {}

def auto_select_target(df):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(numeric_cols) > 0:
        return numeric_cols[-1]
    return df.columns[-1]

def handle_tabular(df, target_column):
    df.columns = df.columns.str.replace(r"[^\w]+", "_", regex=True)

    # Clean text
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(r"[^\x00-\x7F]+", "", regex=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode
    from sklearn.preprocessing import LabelEncoder
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y

def handle_time_series(df, date_col, target_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    ts_df = df[[date_col, target_col]].dropna()
    ts_df.columns = ["ds", "y"]

    return ts_df


@app.post("/train")
async def auto_train(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(None),
    dataset_type: str = Form(None),
    model_name: str = Form("auto"),
):
    filename = (file.filename if file.filename is not None else "").lower()
    user_id = extract_user_id_from_request(request)

    dataset_input = None
    file_path = os.path.join(UPLOAD_FOLDER, file.filename or "unknown")
    
    # Store once fix
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    if is_image_zip(file):
        dataset_input = "zip_placeholder"
    else:
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file_path, sep=None, engine="python")
            else:
                df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()
            df = df.dropna(axis=1, how="all")
            df = df.dropna(axis=0, how="all")
            dataset_input = df
        except Exception:
            return {"error": "Failed to read file"}

    detected_type, meta = detect_dataset_type(dataset_input)
    print("Detected:", detected_type)

    if detected_type == "unknown":
        return {
            "error": "Dataset type not recognized",
            "suggestion": "Ensure dataset has date column or target column"
        }

    if detected_type == "image":
        print("🔥 REAL CNN TRAINING STARTED")

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

        zip_path = file_path # Used the already saved file
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

        try:
            dataset = datasets.ImageFolder(IMAGE_DATASET_FOLDER, transform=transform)
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}

        if len(dataset.classes) < 2:
            return {"error": "Dataset must contain at least 2 classes"}

        if len(dataset) < 20:
            return {"error": "Dataset too small for training"}

        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # ===== MULTI-MODEL IMAGE TRAINING =====
        vision_models = torch_runtime["models"]

        # Inference-safe transform for validation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_dataset = datasets.ImageFolder(IMAGE_DATASET_FOLDER, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        num_classes = len(dataset.classes)

        def make_model(name):
            if name == "ResNet18":
                m = vision_models.resnet18(weights=vision_models.ResNet18_Weights.DEFAULT)
                m.fc = nn.Linear(m.fc.in_features, num_classes)
                return m
            elif name == "ViT":
                m = vision_models.vit_b_16(weights=vision_models.ViT_B_16_Weights.DEFAULT)
                m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
                return m
            elif name == "MobileNet":
                m = vision_models.mobilenet_v2(weights=vision_models.MobileNet_V2_Weights.DEFAULT)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
                return m
            else:  # SimpleCNN (resnet18 backbone frozen)
                return build_simple_cnn(num_classes)

        competitors = ["SimpleCNN", "ResNet18", "ViT", "MobileNet"]
        if model_name and model_name.lower() != "auto" and model_name in competitors:
            competitors = [model_name]
            print(f"🎯 Manual image model selected: {model_name}")
        img_results = []
        best_score = 0.0
        best_loss = float("inf")
        best_model_name = "SimpleCNN"
        best_model_state = None
        total_models = len(competitors)

        for m_idx, m_name in enumerate(competitors):
            print(f"\n🔥 Training {m_name} ({m_idx + 1}/{total_models})")
            update_progress(
                int(20 + (m_idx / total_models) * 60),
                f"Training {m_name}",
                f"Model {m_idx + 1} of {total_models}"
            )

            try:
                m = make_model(m_name).to(device)
                crit = nn.CrossEntropyLoss()

                # Only unfreeze last layer for pretrained models to speed up
                if m_name in ("ResNet18", "ViT"):
                    for p in m.parameters():
                        p.requires_grad = False
                    # Unfreeze classifier head
                    if m_name == "ResNet18":
                        for p in m.fc.parameters():
                            p.requires_grad = True
                    else:
                        for p in m.heads.parameters():
                            p.requires_grad = True

                opt = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, m.parameters()), lr=0.001
                )

                # Train 3 epochs
                m.train()
                total_loss = 0.0
                total_batches = 0
                for epoch in range(3):
                    for imgs, lbls in loader:
                        imgs, lbls = imgs.to(device), lbls.to(device)
                        opt.zero_grad()
                        out = m(imgs)
                        loss = crit(out, lbls)
                        loss.backward()
                        opt.step()
                        total_loss += loss.item()
                        total_batches += 1

                avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

                # Evaluate on val_loader (no augmentation)
                m.eval()
                correct = total = 0
                with torch.no_grad():
                    for imgs, lbls in val_loader:
                        imgs, lbls = imgs.to(device), lbls.to(device)
                        out = m(imgs)
                        _, pred = torch.max(out, 1)
                        total += lbls.size(0)
                        correct += (pred == lbls).sum().item()

                acc = correct / total if total > 0 else 0.0
                print(f"{m_name} Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}")
                img_results.append({"model": m_name, "score": float(acc), "loss": float(avg_loss), "time": None})

                if acc > best_score:
                    best_score = acc
                    best_loss = avg_loss
                    best_model_name = m_name
                    best_model_state = {k: v.cpu() for k, v in m.state_dict().items()}

            except Exception as e:
                print(f"❌ {m_name} failed: {e}")
                img_results.append({"model": m_name, "score": 0.0, "time": None})

        # Sort leaderboard highest first
        img_results_sorted = sorted(img_results, key=lambda x: x["score"], reverse=True)

        training_progress["value"] = 100

        # Save best model with its type so loader can reconstruct correctly
        os.makedirs(os.path.dirname(IMAGE_MODEL_PATH), exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_model_state,
                "classes": dataset.classes,
                "model_type": best_model_name,
            },
            IMAGE_MODEL_PATH,
        )
        print(f"\n✅ SAVED MODEL PATH: {os.path.abspath(IMAGE_MODEL_PATH)}")
        print(f"✅ BEST MODEL: {best_model_name}  |  ACCURACY: {best_score:.4f}")
        print(f"✅ FILE SIZE: {os.path.getsize(IMAGE_MODEL_PATH):,} bytes")

        leaderboard_data = save_leaderboard_snapshot(
            best_model_name=best_model_name,
            model_version=os.path.basename(IMAGE_MODEL_PATH),
            models=img_results_sorted,
            dataset_type="image",
            problem_type="image_classification",
        )

        save_model_record(
            user_id=user_id,
            model_name=best_model_name,
            model_version=os.path.basename(IMAGE_MODEL_PATH),
            dataset_type="image",
            score=float(best_score),
        )

        # Log experiment to MongoDB
        try:
            experiment_doc = {
                "best_model": best_model_name,
                "score": float(best_score),
                "loss": float(best_loss),
                "created_at": datetime.utcnow(),
                "timestamp": datetime.now().isoformat(),
                "dataset": file.filename or "uploaded_file",
                "problem_type": "image",
                "user_id": user_id,
                "target_column": "classification",
                "rows": len(dataset),
                "columns": 3, # RGB
                "metrics": {"accuracy": float(best_score), "loss": float(best_loss)},
                "all_models": [
                    {"model": m["model"], "score": m["score"], "loss": m.get("loss", 0.0)}
                    for m in img_results_sorted
                ],
            }
            experiments_collection.insert_one(experiment_doc)
        except Exception as e:
            print(f"Error logging experiment: {str(e)}")

        track_usage_event(
            user_id, "train_image_model", {"model_name": best_model_name}
        )

        # Track which model type was last trained
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(LAST_MODEL_TYPE_PATH, "w") as f:
            f.write("image")

        _bs = float(best_score)
        return {
            "status": "success",
            "dataset_type": "image",
            "problem_type": "Image Classification",
            "best_model": best_model_name,
            "score": _bs,
            "loss": float(best_loss),
            "classes": dataset.classes,
            "samples": len(dataset),
            "rows": len(dataset),
            "model_version": os.path.basename(IMAGE_MODEL_PATH),
            "metrics": {"main_metric": _bs, "accuracy": _bs, "loss": float(best_loss)},
            "leaderboard": img_results_sorted,
            "top_models": img_results_sorted[:3],
            "all_models": img_results_sorted,
        }

    elif detected_type == "time_series":
        try:
            Prophet = get_prophet_class()
        except RuntimeError as exc:
            return {"error": str(exc)}

        date_column = meta["date_column"]
        target_columns = [auto_select_target(df)]
        ts_rf_scores = {}   # track RF lag model scores per column

        models = {}
        for col in target_columns:
            ts_df = handle_time_series(df, date_column, col)
            if len(ts_df) < 10:
                continue

            # Split, fit and calculate MAE for scoring
            train_size = int(len(ts_df) * 0.8)
            train = ts_df[:train_size]
            test = ts_df[train_size:]

            model = Prophet()
            model.fit(train)

            future = model.make_future_dataframe(periods=len(test))
            forecast = model.predict(future)

            preds = forecast.tail(len(test))["yhat"].values
            actual = test["y"].values
            col_mae = mean_absolute_error(actual, preds)
            col_loss = mean_squared_error(actual, preds)

            # --- RandomForest lag model comparison ---
            try:
                rf_ts_df = ts_df.copy()
                rf_ts_df["lag1"] = rf_ts_df["y"].shift(1)
                rf_ts_df = rf_ts_df.dropna()
                Xlag = rf_ts_df[["lag1"]]
                ylag = rf_ts_df["y"]
                split = int(len(Xlag) * 0.8)
                from sklearn.ensemble import RandomForestRegressor as _RFTS
                from sklearn.model_selection import train_test_split as _tts
                Xl_tr, Xl_te, yl_tr, yl_te = Xlag[:split], Xlag[split:], ylag[:split], ylag[split:]
                rf_ts = _RFTS(n_estimators=50)
                rf_ts.fit(Xl_tr, yl_tr)
                rf_preds = rf_ts.predict(Xl_te)
                rf_col_mae = mean_absolute_error(yl_te, rf_preds) if len(rf_preds) else col_mae
                rf_col_loss = mean_squared_error(yl_te, rf_preds) if len(rf_preds) else col_loss
                ts_rf_scores[col] = {"mae": float(rf_col_mae), "loss": float(rf_col_loss)}
            except Exception:
                ts_rf_scores[col] = {"mae": col_mae, "loss": col_loss}

            # Store model and meta info
            models[col] = {
                "model": model,
                "mae": float(col_mae),
                "loss": float(col_loss)
            }

        if len(models) > 0:
            forecast_output = {}
            maes = []

            for col, model_info in models.items():
                model = model_info["model"]
                maes.append(model_info["mae"])
                
                future = model.make_future_dataframe(periods=10)
                forecast = model.predict(future)
                future_only = forecast.tail(10)[["ds", "yhat"]]
                forecast_output[col] = future_only.to_dict(orient="records")

            avg_mae = sum(m["mae"] for m in models.values()) / len(models)
            avg_loss = sum(m["loss"] for m in models.values()) / len(models)

            # Update models dict to only contain the Prophet objects for joblib
            joblib_models = {col: info["model"] for col, info in models.items()}

            joblib.dump(
                {
                    "models": joblib_models,
                    "problem_type": "time_series_multi",
                    "date_column": date_column,
                    "target_columns": list(models.keys()),
                },
                TIME_SERIES_MODEL_PATH,
            )

            avg_rf_mae = sum(s["mae"] for s in ts_rf_scores.values()) / len(ts_rf_scores) if ts_rf_scores else avg_mae
            avg_rf_loss = sum(s["loss"] for s in ts_rf_scores.values()) / len(ts_rf_scores) if ts_rf_scores else avg_loss

            ts_leaderboard = [
                {"model": "Prophet", "score": float(avg_mae), "loss": float(avg_loss), "time": None},
                {"model": "RandomForest (lag)", "score": float(avg_rf_mae), "loss": float(avg_rf_loss), "time": None},
            ]
            ts_leaderboard_sorted = sorted(ts_leaderboard, key=lambda x: x["score"])  # lower MAE = better
            ts_best = ts_leaderboard_sorted[0]["model"]
            best_ts_loss = ts_leaderboard_sorted[0]["loss"]

            leaderboard_data = save_leaderboard_snapshot(
                best_model_name=ts_best,
                model_version=os.path.basename(TIME_SERIES_MODEL_PATH),
                models=ts_leaderboard_sorted,
                dataset_type="time_series",
                problem_type="time_series_multi",
            )

            save_model_record(
                user_id=user_id,
                model_name=ts_best,
                model_version=os.path.basename(TIME_SERIES_MODEL_PATH),
                dataset_type="time_series",
                score=float(avg_mae),
            )
            track_usage_event(
                user_id, "train_time_series_model", {"model_name": ts_best}
            )

            # Log experiment to MongoDB
            try:
                experiment_doc = {
                    "best_model": ts_best,
                    "score": float(avg_mae),
                    "loss": float(best_ts_loss),
                    "created_at": datetime.utcnow(),
                    "timestamp": datetime.now().isoformat(),
                    "dataset": file.filename or "uploaded_file",
                    "problem_type": "time_series",
                    "user_id": user_id,
                    "target_column": str(meta["target_column"]) if "target_column" in meta else "multi",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "metrics": {"mae": float(avg_mae), "loss": float(best_ts_loss)},
                    "all_models": [
                        {"model": m["model"], "score": m["score"], "loss": m.get("loss", 0.0)}
                        for m in ts_leaderboard_sorted
                    ],
                }
                experiments_collection.insert_one(experiment_doc)
            except Exception as e:
                print(f"Error logging experiment: {str(e)}")

            # Track which model type was last trained
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(LAST_MODEL_TYPE_PATH, "w") as f:
                f.write("time_series")

            return {
                "status": "success",
                "dataset_type": "time_series",
                "problem_type": "Time-Series (Universal)",
                "best_model": ts_best,
                "score": float(avg_mae),
                "loss": float(best_ts_loss),
                "mae": float(avg_mae),
                "date_column": date_column,
                "target_columns": list(models.keys()),
                "forecast": forecast_output,
                "rows": len(df),
                "message": f"Model trained successfully with average MAE: {round(avg_mae, 4)}",
                "model_version": os.path.basename(TIME_SERIES_MODEL_PATH),
                "metrics": {
                    "main_metric": float(avg_mae),
                    "mae": float(avg_mae),
                    "loss": float(best_ts_loss)
                },
                "leaderboard": ts_leaderboard_sorted,
                "top_models": ts_leaderboard_sorted[:3],
                "all_models": ts_leaderboard_sorted
            }
        else:
            return {"error": "Failed to train time-series models on any column"}

    elif detected_type == "tabular":
        # Save original columns before any cleaning
        original_columns = list(df.columns)

        # Clean column names (replace special chars with underscores)
        df.columns = df.columns.str.replace(r"[^\w]+", "_", regex=True)

        # Build original → cleaned mapping
        column_mapping = dict(zip(original_columns, df.columns))

        # ===== TARGET COLUMN FIX =====

        # Clean input
        if target_column:
            target_column = target_column.strip()

        # Map original → cleaned
        if target_column:
            target_column = column_mapping.get(target_column, target_column)

        # Auto-select if missing or wrong
        if not target_column or target_column.lower() in ["string", "none", ""]:
            target_column = auto_select_target(df)
            print(f"✅ Auto-selected target column: {target_column}")

        # Try smart matching if still not found
        if target_column not in df.columns:
            cleaned_map = {c.lower(): c for c in df.columns}
            target_column = cleaned_map.get(target_column.lower(), target_column)

        # Final validation
        if target_column not in df.columns:
            return {
                "error": "Invalid target column",
                "available_columns": original_columns,
                "hint": "Use one of the available columns exactly"
            }

        print(f"🎯 Using target column: {target_column}")

        try:
            df = run_auto_feature_engineering(df)
            update_progress(25, "Feature Engineering Done", "Features created")
        except Exception as e:
            print("Skipping feature engineering:", str(e))

        problem_type = detect_training_problem_type(df, target_column)

        X, y = handle_tabular(df, target_column)

        if problem_type == "classification":
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)

        update_progress(30, "Splitting Data", "Preparing train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # SAVE SAMPLES FOR XAI
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            # Save small sample (max 50 rows) for SHAP/LIME
            X_sample = X_test.iloc[:50] if len(X_test) > 50 else X_test
            y_sample = y_test[:50] if len(y_test) > 50 else y_test
            
            joblib.dump({
                "X_test": X_sample,
                "y_test": y_sample,
                "feature_names": X.columns.tolist(),
                "target_name": target_column
            }, os.path.join(MODEL_DIR, "last_xai_sample.pkl"))
        except Exception as e:
            print(f"⚠️ Failed to save XAI sample: {e}")

        dataset_info = analyze_training_dataset(X, y)
        # Build full recommended list, then apply manual model_name filter
        if problem_type == "classification":
            recommended = ["RandomForest", "LogisticRegression", "SVM", "KNN", "DecisionTree"]
        else:
            recommended = ["RandomForest", "LinearRegression", "SVR", "KNN", "DecisionTree"]

        # Manual model selection: if user picked a specific model, run only that
        if model_name and model_name.lower() != "auto" and model_name in (
            list(get_model_library()[0].keys()) + list(get_model_library()[1].keys())
        ):
            recommended = [model_name]
            print(f"🎯 Manual model selected: {model_name}")

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

        update_progress(35, "Model Selection", f"{len(models)} models selected after filtering")

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
        best_loss = 0.0
        update_progress(40, "Training Models", "Starting model training...")

        futures = []

        start_time = datetime.now()
        max_workers = 2 if LIGHTWEIGHT_DEPLOYMENT else 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for name, model in models.items():
                import copy
                futures.append(
                    executor.submit(
                        train_single_model,
                        name,
                        copy.deepcopy(model),
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

                # Track best score. For regression (MAE/MSE) lower is better sometimes, but score here seems to be R2 for regression typically in your setup. If using MAE, invert logic. Assuming score is still R2.
                if score > best_score:
                    best_score = score
                    best_model = trained_model

        top_models = sorted(results, key=lambda x: x["score"], reverse=True)
        for i, m in enumerate(top_models):
            m["rank"] = i + 1
        top_3_models = top_models[:3]
        best_model_name = top_models[0]["model"]
        update_progress(85, "Finalizing", "Selecting best model...")

        # Guard: if all models failed, return early
        if best_model is None:
            return {
                "error": "All models failed to train",
                "hint": "Check dataset quality or target column"
            }

        # Log experiment to MongoDB
        try:
            best_metrics = top_models[0].get("metrics", {}) if top_models else {}
            best_loss = best_metrics.get("loss", 0.0)
            experiment_doc = {
                "best_model": best_model_name,
                "score": float(best_score),
                "loss": float(best_loss),
                "created_at": datetime.utcnow(),
                "timestamp": datetime.now().isoformat(),
                "dataset": file.filename or "uploaded_file",
                "problem_type": problem_type,
                "user_id": user_id,
                "target_column": target_column,
                "rows": len(df),
                "columns": len(df.columns),
                "metrics": best_metrics,
                "all_models": [
                    {"model": m["model"], "score": m["score"], "loss": m.get("metrics", {}).get("loss", 0.0)}
                    for m in top_models if m.get("score", -999) > -999
                ],
            }
            experiments_collection.insert_one(experiment_doc)
        except Exception as e:
            print(f"Error logging experiment: {str(e)}")

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

        # Safe model save — only write LAST_MODEL_TYPE_PATH on success
        try:
            model_payload = {
                "model": best_model,
                "feature_columns": X.columns.tolist(),
                "problem_type": problem_type,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test
            }
            joblib.dump(model_payload, model_path)
            joblib.dump(model_payload, TABULAR_MODEL_PATH)

            os.makedirs(MODEL_DIR, exist_ok=True)
            try:
                with open(LAST_MODEL_TYPE_PATH, "w") as f:
                    f.write("tabular")
                print("✅ Model type set to TABULAR")
            except Exception as e:
                print("❌ Failed to write model type:", str(e))

            update_progress(95, "Saving", "Saving model and reports...")

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

        except Exception as e:
            return {"error": f"Model saving failed: {str(e)}"}

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

            best_model_metrics = top_models[0].get("metrics", {}) if top_models else {}
            return {
                "status": "success",
                "dataset_type": "tabular",
                "problem_type": "Classification",
                "rows": len(df),
                "target_column": target_column,
                "best_model": str(best_model_name),
                "score": float(best_score) if best_score is not None else 0.0,
                "loss": float(best_loss),
                "accuracy": round(best_score, 4),
                "top_models": top_3_models,
                "all_models": top_models,
                "leaderboard": leaderboard_data["models"],
                "generated_code": generated_code,
                "model_version": model_filename,
                "ai_insights": ai_insights,
                "metrics": {
                    "main_metric": float(best_score) if best_score is not None else 0.0,
                    **best_model_metrics,
                }
            }
        else:
            # We must make sure test labels and predictions match
            try:
                preds = best_model.predict(X_test)
                import math
                mse  = mean_squared_error(y_test, preds)
                rmse = math.sqrt(mse)
                mae  = float(np.mean(np.abs(np.array(y_test) - np.array(preds))))
            except:
                mse = rmse = mae = 0.0
                
            update_progress(
                100, "Completed", "Training completed successfully", eta="0 sec"
            )
            best_model_metrics = top_models[0].get("metrics", {}) if top_models else {}
            return {
                "status": "success",
                "dataset_type": "tabular",
                "problem_type": "Regression",
                "rows": len(df),
                "target_column": target_column,
                "best_model": str(best_model_name),
                "score": float(best_score) if best_score is not None else 0.0,
                "loss": float(best_loss),
                "mse": round(mse, 4),
                "r2": round(best_score, 4),
                "top_models": top_3_models,
                "all_models": top_models,
                "leaderboard": leaderboard_data["models"],
                "generated_code": generated_code,
                "model_version": model_filename,
                "ai_insights": ai_insights,
                "metrics": {
                    "main_metric": float(best_score) if best_score is not None else 0.0,
                    **best_model_metrics,
                }
            }



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
    if not os.path.exists(LAST_MODEL_TYPE_PATH):
        return {"error": "No model trained yet"}

    with open(LAST_MODEL_TYPE_PATH, "r") as f:
        model_type = f.read().strip()

    print(f"🔥 LAST MODEL TYPE: {model_type}")

    if model_type == "image":
        if not os.path.exists(IMAGE_MODEL_PATH):
            return {"error": "No trained image model found. Train an image dataset first."}

        print(f"📥 LOADING MODEL FROM: {os.path.abspath(IMAGE_MODEL_PATH)}")
        print(f"📦 FILE SIZE: {os.path.getsize(IMAGE_MODEL_PATH):,} bytes")


        try:
            torch_runtime = get_torch_runtime()
            torch = torch_runtime["torch"]
            nn = torch_runtime["nn"]
            device = torch_runtime["device"]
            transforms = torch_runtime["transforms"]
            Image = get_image_module()
        except Exception as e:
            return {"error": f"PyTorch not available: {str(e)}"}

        try:
            checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
        except Exception as e:
            return {"error": f"Failed to load image model: {str(e)}"}

        classes = checkpoint.get("classes", [])
        if not classes:
            return {"error": "Model has no class labels. Retrain."}

        saved_model_type = checkpoint.get("model_type", "SimpleCNN")
        vision_models = torch_runtime["models"]
        print(f"🧠 Reconstructing architecture: {saved_model_type}")

        if saved_model_type == "ResNet18":
            model_obj = vision_models.resnet18()
            model_obj.fc = nn.Linear(model_obj.fc.in_features, len(classes))
        elif saved_model_type == "ViT":
            model_obj = vision_models.vit_b_16()
            model_obj.heads.head = nn.Linear(model_obj.heads.head.in_features, len(classes))
        else:
            model_obj = build_simple_cnn(len(classes))

        model_obj.load_state_dict(checkpoint["model_state_dict"])
        model_obj = model_obj.to(device)
        model_obj.eval()

        # Read the uploaded file as an image
        try:
            raw_bytes = await file.read()
            import io
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            return {"error": "Invalid image file. Upload a .jpg / .png / .jpeg"}

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model_obj(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        predicted_class = classes[pred.item()]
        conf_val = float(confidence.item())
        risk = "Low Risk" if conf_val > 0.85 else "Moderate Risk" if conf_val > 0.65 else "High Risk"

        return {
            "problem_type": "Image Classification",
            "num_predictions": 1,
            "predictions": [{
                "prediction": predicted_class,
                "confidence": round(conf_val, 4),
                "risk_level": risk,
            }],
            "predicted_class": predicted_class,
            "confidence": round(conf_val, 4),
            "risk_level": risk,
            "all_probabilities": {
                classes[i]: round(float(probs[0][i]), 4)
                for i in range(len(classes))
            },
        }

    elif model_type == "tabular":
        if not os.path.exists(TABULAR_MODEL_PATH):
            return {"error": "Tabular model file missing. Retrain."}
        model_package = joblib.load(TABULAR_MODEL_PATH)

    elif model_type == "time_series":
        if not os.path.exists(TIME_SERIES_MODEL_PATH):
            return {"error": "Time-series model file missing. Retrain."}
        model_package = joblib.load(TIME_SERIES_MODEL_PATH)

    else:
        return {"error": "Unknown model type"}

    problem_type = model_package.get("problem_type")

                                                
                            
                                                
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

    if not os.path.exists(IMAGE_MODEL_PATH):
        return {"error": "No trained CNN model"}

    torch_runtime = get_torch_runtime()
    torch = torch_runtime["torch"]
    nn = torch_runtime["nn"]
    device = torch_runtime["device"]
    transforms = torch_runtime["transforms"]
    vision_models = torch_runtime["models"]
    Image = get_image_module()

    checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]
    saved_model_type = checkpoint.get("model_type", "SimpleCNN")
    print(f"🧠 /predict-image reconstructing: {saved_model_type}")

    if saved_model_type == "ResNet18":
        model = vision_models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    elif saved_model_type == "ViT":
        model = vision_models.vit_b_16()
        model.heads.head = nn.Linear(model.heads.head.in_features, len(classes))
    else:
        model = build_simple_cnn(len(classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    image = Image.open(file.file).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
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

    if not os.path.exists(IMAGE_MODEL_PATH):
        return {"error": "No trained CNN model"}

    torch_runtime = get_torch_runtime()
    torch = torch_runtime["torch"]
    device = torch_runtime["device"]
    transforms = torch_runtime["transforms"]
    Image = get_image_module()
    plt = get_matplotlib_pyplot()

    checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
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

    matched_path = TABULAR_MODEL_PATH if os.path.exists(TABULAR_MODEL_PATH) else TIME_SERIES_MODEL_PATH
    if not os.path.exists(matched_path):
        return {"error": "No trained model found"}

    model_package = joblib.load(matched_path)

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

    matched_path = TABULAR_MODEL_PATH if os.path.exists(TABULAR_MODEL_PATH) else TIME_SERIES_MODEL_PATH
    if not os.path.exists(matched_path):
        return {"error": "No trained model found"}

    model_package = joblib.load(matched_path)

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

            matched_path = TABULAR_MODEL_PATH if os.path.exists(TABULAR_MODEL_PATH) else TIME_SERIES_MODEL_PATH
            if os.path.exists(matched_path):
                zipf.write(matched_path, arcname=os.path.basename(matched_path))

                              
            with open(GENERATED_REQUIREMENTS_PATH, "w") as f:
                f.write("pandas\nnumpy\nscikit-learn\njoblib\n")

            zipf.write(GENERATED_REQUIREMENTS_PATH, arcname="requirements.txt")

        return FileResponse(zip_path)

                                
                    
                                
    else:
        return {"error": "Format not supported"}


                                                       
             
                                                       
@app.get("/experiments")
def get_experiments(request: Request):
    user_id = extract_user_id_from_request(request)
    query = {}
    if user_id:
        query["user_id"] = user_id
    experiments = list(experiments_collection.find(query, {"_id": 0}).sort("created_at", -1))
    return experiments

@app.get("/leaderboard/{type}")
def get_leaderboard_by_type(type: str, request: Request):
    user_id = extract_user_id_from_request(request)
    query = {"problem_type": type}
    if user_id:
        query["user_id"] = user_id
    data = list(experiments_collection.find(query, {"_id": 0}))
    sorted_data = sorted(data, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_data[:10]


                                                       
               
                                                       
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


                                                       
                            
                                                       
@app.get("/dashboard")
def get_dashboard(request: Request):
    user_id = extract_user_id_from_request(request)
    query = {"user_id": user_id} if user_id else {}
    data = list(experiments_collection.find(query, {"_id": 0}))

    if not data:
        return {
            "total_experiments": 0,
            "best_score": None,
            "best_model": None,
            "recent": []
        }

    from datetime import datetime

    def safe_date(x):
        try:
            # Handle both ISO strings and potentially existing datetime objects
            ts = x.get("timestamp") or x.get("created_at")
            if isinstance(ts, datetime):
                return ts
            return datetime.fromisoformat(ts)
        except:
            return datetime.min

    # Best score based on max value
    best = max(data, key=lambda x: x.get("score", 0))
    # Best loss (min non-zero loss)
    losses = [x.get("loss", 999) for x in data if x.get("loss", 0) > 0]
    best_loss = min(losses) if losses else None
    
    recent = sorted(data, key=safe_date, reverse=True)[:5]

    return {
        "total_experiments": len(data),
        "best_score": best.get("score"),
        "best_loss": best_loss,
        "best_model": best.get("best_model") or best.get("model_name"),
        "recent": recent
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
    matched_path = TABULAR_MODEL_PATH if os.path.exists(TABULAR_MODEL_PATH) else TIME_SERIES_MODEL_PATH
    if os.path.exists(matched_path):
        return FileResponse(matched_path, filename=os.path.basename(matched_path))
    return {"error": "No trained model"}


@app.get("/download-latest-model")
def download_latest_model():
    matched_path = TABULAR_MODEL_PATH if os.path.exists(TABULAR_MODEL_PATH) else TIME_SERIES_MODEL_PATH
    if os.path.exists(matched_path):
        return FileResponse(matched_path, filename=os.path.basename(matched_path))
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
            or "model.pkl",
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
            model_version="model.pkl",
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

    # AI Summary
    dataset_info_str = f"Columns: {list(df.columns)}, Rows: {len(df)}, Types: {df.dtypes.astype(str).to_dict()}"
    ai_summary = ask_gemini(f"Analyze this dataset summary and give 3 key insights/recommendations for ML: {dataset_info_str}")

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "ai_summary": ai_summary
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

    client = require_openclient()

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

    client = require_openclient()

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

    client = require_openclient()

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
    try:
        message = data.get("message", "")
        dataset_info = data.get("dataset_info", "")

        if not message:
            return {"response": "Empty message"}

        if not gemini_client:
            return {"response": "AI not configured"}

        prompt = message
        if dataset_info:
            prompt = f"""
            You are an AI ML assistant for an AutoML platform.
            
            Dataset context:
            {dataset_info}
            
            User Question:
            {message}
            """

        reply = call_gemini_safely(prompt, message)
        return {"response": reply}

    except Exception as e:
        print("🔥 CHAT ERROR:", str(e))
        return {"response": f"AI is temporarily unavailable. (Err: {str(e)})"}





@app.get("/api/insights")
def insights():
    return {"status": "ok", "message": "Insights API active"}


@app.get("/")
def home():
    return {"status": "Backend is LIVE 🚀", "message": "AutoML API running"}

@app.post("/download-training-script")
def download_script():
    return FileResponse("generated_pipeline.py")
@app.post("/model-explain")
async def model_explain(request: Request):
    try:
        req_data = await request.json()
        explain_type = req_data.get("type", "feature_importance")
        
        # Load Model
        pkl_path = os.path.join(MODEL_DIR, "model_v1.pkl") # Simplified for now, should find latest
        # Try to find the actual latest model
        all_models = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_v") and f.endswith(".pkl")]
        if all_models:
            latest = sorted(all_models, key=lambda x: int(x.replace("model_v", "").replace(".pkl", "")))[-1]
            pkl_path = os.path.join(MODEL_DIR, latest)

        if not os.path.exists(pkl_path):
            return {"error": "No trained tabular model found"}

        model_data = joblib.load(pkl_path)
        model_obj = model_data.get("model")
        features = model_data.get("feature_columns")
        problem_type = model_data.get("problem_type")

        # Load Data from model payload
        X_train = model_data.get("X_train")
        X_test = model_data.get("X_test")
        y_test = model_data.get("y_test")

        if X_train is None or X_test is None:
            # Fallback to sample file if data not in model (backwards compatibility)
            sample_path = os.path.join(MODEL_DIR, "last_xai_sample.pkl")
            if os.path.exists(sample_path):
                sample_data = joblib.load(sample_path)
                X_train = sample_data.get("X_test") # Use test as proxy for train if missing
                X_test = sample_data.get("X_test")
                y_test = sample_data.get("y_test")
            else:
                return {"error": "No dataset split found in model. Please retrain."}

        summary = {
            "model_name": type(model_obj).__name__,
            "problem_type": problem_type,
            "features_count": len(features)
        }
        
        # Determine score for summary
        score_val = "N/A"
        try:
            from sklearn.metrics import accuracy_score, r2_score
            y_pred = model_obj.predict(X_test)
            score_val = float(accuracy_score(y_test, y_pred)) if problem_type == "classification" else float(r2_score(y_test, y_pred))
            score_val = f"{score_val:.4f}"
        except:
            pass

        summary["ai_summary"] = generate_model_summary(
            summary["model_name"], score_val, problem_type
        )

        response = {"summary": summary}

        if explain_type == "feature_importance":
            if hasattr(model_obj, "feature_importances_"):
                fi = model_obj.feature_importances_.tolist()
                response["feature_importance"] = [{"feature": f, "importance": i} for f, i in zip(features, fi)]
            elif hasattr(model_obj, "coef_"):
                import numpy as np
                coef = model_obj.coef_
                if len(coef.shape) > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                response["feature_importance"] = [{"feature": f, "importance": float(i)} for f, i in zip(features, coef)]
            else:
                response["error"] = "Model does not support direct feature importance"

        elif explain_type == "shap":
            import shap
            # Use X_train as background and explain first 50 of X_test
            # Ensure X_test is not larger than 50 for performance
            X_explain = X_test.iloc[:50] if len(X_test) > 50 else X_test
            explainer = shap.Explainer(model_obj, X_train)
            shap_values = explainer(X_explain)
            
            import numpy as np
            vals = np.abs(shap_values.values).mean(0)
            if len(vals.shape) > 1: vals = vals.mean(axis=1) # handle multiclass
            response["shap"] = [{"feature": f, "value": float(v)} for f, v in zip(features, vals)]

        elif explain_type == "lime":
            from lime.lime_tabular import LimeTabularExplainer
            explainer = LimeTabularExplainer(X_train.values, feature_names=features, class_names=['0', '1'], mode=problem_type)
            exp = explainer.explain_instance(X_test.iloc[0].values, model_obj.predict_proba if hasattr(model_obj, "predict_proba") else model_obj.predict)
            response["lime"] = [{"feature": f, "impact": float(v)} for f, v in exp.as_list()]

        elif explain_type == "metrics":
            from sklearn.metrics import confusion_matrix, roc_curve, mean_squared_error, r2_score
            y_pred = model_obj.predict(X_test)
            
            if problem_type == "classification":
                cm = confusion_matrix(y_test, y_pred).tolist()
                response["confusion_matrix"] = cm
                
                if hasattr(model_obj, "predict_proba"):
                    probs = model_obj.predict_proba(X_test)
                    if probs.shape[1] >= 2:
                        fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
                        response["roc"] = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]
            elif problem_type == "regression":
                mse = float(mean_squared_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                response["metrics"] = {"mse": mse, "r2": r2}
                response["residuals"] = (y_test - y_pred).tolist()

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/ask-model")
async def ask_model(data: dict = Body(...)):
    try:
        question = data.get("question", "")

        if not gemini_client:
            return {"answer": "AI not configured"}

        prompt = f"""
        You are an expert ML engineer.
        Answer clearly:

        Question: {question}
        """

        answer = call_gemini_safely(prompt, question)
        return {"answer": answer}

    except Exception as e:
        return {"answer": str(e)}

@app.post("/explain-image-advanced")
async def explain_image(file: UploadFile = File(...)):
    import shutil
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    if not os.path.exists(IMAGE_MODEL_PATH):
        return {"error": "No image model found"}

    # Save uploaded file temporarily
    os.makedirs("uploads", exist_ok=True)
    temp_path = f"uploads/temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        checkpoint = torch.load(IMAGE_MODEL_PATH, map_location="cpu")
        model_type = checkpoint.get("model_type", "SimpleCNN")
        num_classes = len(checkpoint.get("classes", []))

        # Reconstruct model
        if "ResNet18" in model_type:
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "MobileNet" in model_type:
            model = models.mobilenet_v2()
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        else:
            return {"error": f"Grad-CAM not supported for {model_type}"}

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Load and preprocess image
        image = cv2.imread(temp_path)
        image = cv2.resize(image, (224, 224))
        rgb_img = np.float32(image) / 255
        input_tensor = transforms.ToTensor()(image).unsqueeze(0)

        # Target layer
        target_layers = []
        for layer in reversed(list(model.modules())):
            if isinstance(layer, torch.nn.Conv2d):
                target_layers = [layer]
                break

        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        output_filename = f"gradcam_{file.filename}"
        output_path = os.path.join("uploads", output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        return {"gradcam": f"/uploads/{output_filename}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/explain-vit")
async def explain_vit(file: UploadFile = File(...)):
    # Simple simulated attention heatmap for ViT
    import shutil
    import cv2
    import os

    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    # Create fake heatmap
    heatmap = cv2.applyColorMap(cv2.resize(img, (224, 224)), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    out_name = f"vit_attn_{file.filename}"
    out_path = os.path.join("uploads", out_name)
    cv2.imwrite(out_path, overlay)

    return {"attention": f"/uploads/{out_name}"}
