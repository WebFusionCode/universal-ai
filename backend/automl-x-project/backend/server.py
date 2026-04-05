import asyncio
import json
import os
import random
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import numpy as np
import pandas as pd
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from passlib.context import CryptContext
from pymongo import MongoClient

# ===== CONFIG =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIGHTWEIGHT_DEPLOYMENT = True
MAX_LIGHTWEIGHT_ROWS = 10000
RUN_LOCAL_TRAINING = True
PREVIEW_RESPONSE_ROWS = 5
LIGHTWEIGHT_BLOCKED_MODELS = ["TabTransformer", "CatBoost"]

# ===== FILE PATHS =====
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
EXPERIMENTS_PATH = os.path.join(EXPERIMENTS_DIR, "experiments.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
LEADERBOARD_PATH = os.path.join(BASE_DIR, "leaderboard.pkl")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.pth")
TRAINING_REPORT_PATH = os.path.join(BASE_DIR, "training_report.pkl")
IMAGE_DATASET_FOLDER = os.path.join(BASE_DIR, "image_dataset")
GENERATED_PIPELINE_PATH = os.path.join(BASE_DIR, "generated_pipeline.py")
GENERATED_NOTEBOOK_PATH = os.path.join(BASE_DIR, "generated_notebook.ipynb")
GENERATED_API_PATH = os.path.join(BASE_DIR, "generated_api.py")
GENERATED_REQUIREMENTS_PATH = os.path.join(BASE_DIR, "generated_requirements.txt")
GENERATED_DOCKERFILE_PATH = os.path.join(BASE_DIR, "Dockerfile_generated")
DOCKER_PACKAGE_PATH = os.path.join(BASE_DIR, "docker_package.zip")
FULL_PROJECT_ZIP_PATH = os.path.join(BASE_DIR, "full_project.zip")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(EXPERIMENTS_PATH):
    with open(EXPERIMENTS_PATH, "w") as f:
        f.write("[]")

# ===== AUTH CONFIG =====
SECRET_KEY = os.getenv("SECRET_KEY", "automl-x-secret-key-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password[:72])

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password[:72], hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ===== MODELS =====
class UserSignup(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# ===== DATABASE =====
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "automl_db")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]
users_collection = db["users"]
models_collection = db["models"]
subscriptions_collection = db["subscriptions"]
teams_collection = db["teams"]
usage_collection = db["usage"]

# ===== LLM CONFIG =====
EMERGENT_LLM_KEY = os.getenv("EMERGENT_LLM_KEY", "")

# ===== TRAINING PROGRESS =====
training_progress = {"progress": 0, "status": "Idle", "message": "", "log": "", "eta": ""}

def update_progress(progress=None, status=None, message=None, log=None, eta=None):
    global training_progress
    if progress is not None:
        training_progress["progress"] = progress
    if status is not None:
        training_progress["status"] = status
    if message is not None:
        training_progress["message"] = message
    if log is not None:
        training_progress["log"] = log
    if eta is not None:
        training_progress["eta"] = eta

# ===== APP =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== LAZY IMPORTS =====
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

def generate_training_pipeline_code(model_name, feature_columns, problem_type, target_column):
    from utils.code_generator import generate_training_code
    return generate_training_code(model_name=model_name, feature_columns=feature_columns, problem_type=problem_type, target_column=target_column)

def get_tune_random_forest():
    from utils.hyperparameter_tuning import tune_random_forest
    return tune_random_forest

# ===== HELPERS =====
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
    with open(EXPERIMENTS_PATH, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []

def filter_experiments_for_user(experiments, user_id=None):
    if not user_id:
        return experiments
    return [item for item in experiments if item.get("user_id") == user_id]

def save_model_record(user_id, model_name, model_version, dataset_type, score=None):
    if not user_id:
        return
    try:
        models_collection.insert_one({
            "user_id": user_id, "model_name": model_name,
            "model_version": model_version, "dataset_type": dataset_type,
            "score": float(score) if score is not None else None,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception:
        pass

def track_usage_event(user_id, action, metadata=None):
    if not user_id or not action:
        return
    doc = {"user_id": user_id, "action": action, "time": datetime.utcnow().isoformat()}
    if metadata:
        doc["metadata"] = metadata
    try:
        usage_collection.insert_one(doc)
    except Exception:
        pass

def is_admin(user_id):
    if not user_id:
        return False
    try:
        user = users_collection.find_one({"user_id": user_id}, {"role": 1})
    except Exception:
        return False
    return (user or {}).get("role") == "admin"

def build_leaderboard_payload(best_model_name, model_version, models, dataset_type=None, problem_type=None):
    normalized = []
    for i, m in enumerate(models or []):
        normalized.append({
            "rank": m.get("rank", i + 1),
            "model": m.get("model", "Unknown"),
            "score": float(m["score"]) if m.get("score") is not None else None,
            "time": float(m["time"]) if m.get("time") is not None else None,
            "metrics": m.get("metrics", {}),
        })
    return {
        "best_model": best_model_name, "model_version": model_version,
        "dataset_type": dataset_type, "problem_type": problem_type,
        "total_models": len(normalized), "models": normalized,
    }

def save_leaderboard_snapshot(best_model_name, model_version, models, dataset_type=None, problem_type=None):
    payload = build_leaderboard_payload(best_model_name, model_version, models, dataset_type, problem_type)
    joblib.dump(payload, LEADERBOARD_PATH)
    return payload

def normalize_experiment_entry(entry):
    timestamp = entry.get("time") or entry.get("timestamp")
    lb_models = build_leaderboard_payload(
        best_model_name=entry.get("best_model") or entry.get("model_name") or "N/A",
        model_version=entry.get("model_version", os.path.basename(MODEL_PATH)),
        models=entry.get("leaderboard", []),
        dataset_type=entry.get("dataset_type"),
        problem_type=entry.get("problem_type"),
    )["models"]
    best_model_name = entry.get("best_model") or entry.get("model_name") or (lb_models[0]["model"] if lb_models else "N/A")
    score = entry.get("score")
    if score is None:
        for m in lb_models:
            if m.get("score") is not None:
                score = m["score"]
                break
    return {
        "time": timestamp, "timestamp": entry.get("timestamp", timestamp),
        "user_id": entry.get("user_id"), "best_model": best_model_name,
        "model_name": entry.get("model_name", best_model_name),
        "leaderboard": lb_models, "model_version": entry.get("model_version"),
        "problem_type": entry.get("problem_type"), "dataset_type": entry.get("dataset_type"),
        "score": float(score) if score is not None else None,
        "rows": entry.get("rows"), "columns": entry.get("columns"),
        "total_models": entry.get("total_models", len(lb_models)),
    }

def append_experiment_log(best_model_name, leaderboard_models, model_version, user_id=None, problem_type=None, dataset_type=None, rows=None, columns=None, score=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment = normalize_experiment_entry({
        "time": timestamp, "timestamp": timestamp, "user_id": user_id,
        "best_model": best_model_name, "model_name": best_model_name,
        "leaderboard": leaderboard_models, "model_version": model_version,
        "problem_type": problem_type, "dataset_type": dataset_type,
        "score": score, "rows": rows, "columns": columns,
        "total_models": len(leaderboard_models or []),
    })
    logs = load_experiment_logs()
    logs.append(experiment)
    with open(EXPERIMENTS_PATH, "w") as f:
        json.dump(logs, f, indent=4)
    return experiment

def dataset_quality_score(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_ratio = missing_cells / total_cells
    numeric_ratio = len(df.select_dtypes(include=["int64", "float64"]).columns) / max(1, len(df.columns))
    score = 100 - missing_ratio * 40
    if len(df) < 100:
        score -= 20
    elif len(df) < 500:
        score -= 10
    score += numeric_ratio * 10
    score = max(0, min(100, round(score, 2)))
    return {"quality_score": score, "missing_ratio": round(missing_ratio, 4), "numeric_feature_ratio": round(numeric_ratio, 4)}

def model_strength_summary(problem_type, metrics):
    if problem_type == "classification":
        acc = metrics.get("accuracy", 0) or 0
        level = "Excellent" if acc > 0.9 else "Strong" if acc > 0.8 else "Moderate" if acc > 0.7 else "Weak"
        return {"model_strength": level, "accuracy": round(acc, 4)}
    else:
        r2 = metrics.get("r2", 0) or 0
        level = "Excellent" if r2 > 0.9 else "Strong" if r2 > 0.8 else "Moderate" if r2 > 0.6 else "Weak"
        return {"model_strength": level, "r2_score": round(r2, 4)}

def generate_explanation_text(problem_type, strength):
    if problem_type == "classification":
        return f"This classification model demonstrates {strength['model_strength']} performance with an accuracy of {strength['accuracy']}."
    return f"This regression model demonstrates {strength['model_strength']} performance with an R2 score of {strength['r2_score']}."

def generate_ai_insights(df, problem_type, best_model_name, best_score):
    insights = []
    if len(df) < 500:
        insights.append("Dataset is small. Consider adding more data for better performance.")
    elif len(df) > 50000:
        insights.append("Large dataset detected. Consider using sampling or simpler models.")
    if df.shape[1] > 50:
        insights.append("High number of features. Feature selection may improve performance.")
    missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_ratio > 0.2:
        insights.append("High missing values detected. Data cleaning is recommended.")
    if problem_type == "classification":
        if best_score > 0.9:
            insights.append(f"{best_model_name} is performing excellently.")
        elif best_score > 0.75:
            insights.append(f"{best_model_name} is performing well, but tuning may improve results.")
        else:
            insights.append("Model performance is low. Try feature engineering or more data.")
    else:
        if best_score > 0.85:
            insights.append(f"{best_model_name} has strong predictive power.")
        elif best_score > 0.6:
            insights.append("Model is moderate. Consider tuning or better features.")
        else:
            insights.append("Poor regression performance. Try different models or transformations.")
    insights.append("Try ensemble models for better accuracy.")
    insights.append("Hyperparameter tuning can further improve performance.")
    return insights

def auto_tune_model(name, model, X_train, y_train, problem_type):
    try:
        if "RandomForest" in name:
            tune_rf = get_tune_random_forest()
            return tune_rf(X_train, y_train, problem_type)
        if "LogisticRegression" in name:
            return {"C": random.choice([0.1, 1, 10]), "max_iter": random.choice([100, 200])}
        if "SVM" in name:
            return {"C": random.choice([0.1, 1, 10]), "kernel": random.choice(["linear", "rbf"])}
        if "KNN" in name:
            return {"n_neighbors": random.choice([3, 5, 7, 9])}
        if "SVR" in name:
            return {"C": random.choice([0.1, 1, 10])}
        return {}
    except Exception:
        return {}

def train_single_model(name, model, X_train, X_test, y_train, y_test, problem_type):
    start = datetime.now()
    try:
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
        duration = (datetime.now() - start).total_seconds()
        return {"model": name, "score": float(score), "trained_model": model, "time": duration, "metrics": metrics}
    except Exception as e:
        return {"model": name, "score": -999, "error": str(e), "trained_model": None}

def filter_models(models, X, y, problem_type):
    selected = {}
    num_rows, num_cols = len(X), X.shape[1]
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
        selected[name] = model
    return selected

def auto_ml_recommendation(df, target_column):
    if target_column not in df.columns:
        raise ValueError("Invalid target column")
    target_series = df[target_column]
    if pd.api.types.is_object_dtype(target_series) or target_series.nunique(dropna=True) < 20:
        problem_type = "Classification"
        recommended_models = ["LogisticRegression", "RandomForest", "XGBoost"]
        feature_engineering = ["Normalize numeric features", "Encode categorical variables", "Remove highly correlated features"]
        best_model_hint = "RandomForest or XGBoost usually handle mixed tabular features well."
    else:
        problem_type = "Regression"
        recommended_models = ["LinearRegression", "RandomForestRegressor", "XGBoostRegressor"]
        feature_engineering = ["Scale skewed numeric features", "Encode categorical variables", "Remove highly correlated features"]
        best_model_hint = "XGBoostRegressor is often a strong baseline for tabular regression."
    missing = int(df.isnull().sum().sum())
    fix = "Handle missing values using imputation" if missing > 0 else "No missing values detected"
    recommendations = {
        "problem_type": problem_type, "recommended_models": recommended_models,
        "best_model_hint": best_model_hint, "missing_values": missing,
        "fix": fix, "feature_engineering": feature_engineering,
    }
    if problem_type == "Classification":
        counts = df[target_column].value_counts(dropna=True)
        if len(counts) > 1 and counts.min() / max(counts.max(), 1) < 0.5:
            recommendations["imbalance"] = "Imbalanced dataset"
        else:
            recommendations["imbalance"] = "Balanced dataset"
    return recommendations

# ===== API ROUTES =====

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "AutoML X API"}

@app.get("/api/")
def home():
    return {"status": "Backend is LIVE", "message": "AutoML X API running"}

# ===== AUTH =====
@app.post("/api/signup")
async def signup(user: UserSignup):
    try:
        existing = users_collection.find_one({"email": user.email})
        if existing:
            raise HTTPException(status_code=400, detail="User already exists")
        hashed_password = hash_password(user.password)
        user_id = str(uuid.uuid4())
        users_collection.insert_one({
            "user_id": user_id, "email": user.email, "password": hashed_password,
            "role": "user", "name": "", "phone": "", "dob": "", "profile_pic": "",
            "created_at": datetime.utcnow().isoformat(),
        })
        track_usage_event(user_id, "signup", {"email": user.email})
        token = create_access_token({"sub": user.email, "user_id": user_id})
        return {"message": "Signup successful", "user_id": user_id, "access_token": token, "token_type": "bearer", "email": user.email}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
async def login(data: UserLogin):
    try:
        user = users_collection.find_one({"email": data.email})
        if not user or not verify_password(data.password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        track_usage_event(user.get("user_id"), "login")
        token = create_access_token({"sub": user["email"], "user_id": user.get("user_id")})
        return {"access_token": token, "token_type": "bearer", "user_id": user.get("user_id"), "email": user["email"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    user = users_collection.find_one({"user_id": user_id}, {"_id": 0, "password": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/api/update-profile")
async def update_profile(data: dict, request: Request):
    user_id = (data.get("user_id") or "").strip() or extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    updates = {"name": data.get("name", ""), "phone": data.get("phone", ""), "dob": data.get("dob", ""), "profile_pic": data.get("profile_pic", "")}
    result = users_collection.update_one({"user_id": user_id}, {"$set": updates})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Profile updated"}

# ===== DATASET =====
@app.post("/api/preview")
async def preview_dataset(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    fn = filename.lower()
    if fn.endswith(".csv"):
        df = pd.read_csv(file_path, sep=None, engine="python")
    elif fn.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif fn.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, XLSX, or JSON.")
    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    preview_rows = df.head(PREVIEW_RESPONSE_ROWS).astype(object).where(pd.notna(df.head(PREVIEW_RESPONSE_ROWS)), None).to_dict(orient="records")
    dataset_type = "tabular"
    target_suggestions = numeric_cols[:5]
    return {
        "rows": len(df), "columns": list(df.columns), "preview": preview_rows,
        "numeric_columns": numeric_cols, "categorical_columns": categorical_cols,
        "suggested_target_columns": target_suggestions, "dataset_type": dataset_type,
    }

@app.post("/api/preview-columns")
async def preview_columns(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    fn = filename.lower()
    if fn.endswith(".csv"):
        df = pd.read_csv(file_path, nrows=5)
    elif fn.endswith(".xlsx"):
        df = pd.read_excel(file_path, nrows=5)
    elif fn.endswith(".json"):
        df = pd.read_json(file_path)
        df = df.head(5)
    else:
        return {"error": "Unsupported format"}
    return {"columns": list(df.columns), "preview": df.head().to_dict(orient="records")}

@app.post("/api/analyze-dataset")
async def analyze_dataset_file(file: UploadFile = File(...)):
    fn = (file.filename or "").lower()
    if fn.endswith(".csv"):
        df = pd.read_csv(file.file, sep=None, engine="python")
    elif fn.endswith(".xlsx"):
        df = pd.read_excel(file.file)
    elif fn.endswith(".json"):
        df = pd.read_json(file.file)
    else:
        return {"error": "Unsupported file format"}
    return {
        "rows": len(df), "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "data_types": df.dtypes.astype(str).to_dict(),
    }

@app.post("/api/auto-ml-insights")
async def auto_ml_insights(file: UploadFile = File(...), target_column: str = Form(...)):
    fn = (file.filename or "").lower()
    if fn.endswith(".csv"):
        df = pd.read_csv(file.file, sep=None, engine="python")
    elif fn.endswith(".xlsx"):
        df = pd.read_excel(file.file)
    elif fn.endswith(".json"):
        df = pd.read_json(file.file)
    else:
        return {"error": "Unsupported file format"}
    try:
        return auto_ml_recommendation(df, target_column)
    except ValueError as exc:
        return {"error": str(exc)}

# ===== TRAINING =====
@app.post("/api/train")
async def auto_train(request: Request, file: UploadFile = File(...), target_column: str = Form(None)):
    filename = (file.filename or "").lower()
    user_id = extract_user_id_from_request(request)

    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".json")):
        raise HTTPException(status_code=400, detail="Unsupported dataset format")

    update_progress(5, "Loading dataset", "Dataset loading started...")
    file_path = os.path.join(UPLOAD_FOLDER, file.filename or "unknown")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(file_path, sep=None, engine="python")
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to read CSV file")
    elif filename.endswith(".xlsx"):
        try:
            df = pd.read_excel(file_path)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to read Excel file")
    else:
        try:
            df = pd.read_json(file_path)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to read JSON file")

    df.columns = df.columns.str.strip()
    update_progress(10, "Dataset Loaded", "Dataset successfully loaded")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if LIGHTWEIGHT_DEPLOYMENT and len(df) > MAX_LIGHTWEIGHT_ROWS:
        raise HTTPException(status_code=400, detail=f"Dataset too large. Limit is {MAX_LIGHTWEIGHT_ROWS} rows.")

    if target_column is None:
        return {"error": "Provide target_column for tabular dataset", "available_columns": list(df.columns)}

    if target_column not in df.columns:
        return {"error": "Invalid target column", "available_columns": list(df.columns)}

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
    if numeric_cols:
        X[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(X[numeric_cols])
        X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype(str).fillna("missing")
        X[col] = LabelEncoder().fit_transform(X[col])
    X = X.fillna(0)

    update_progress(30, "Splitting Data", "Preparing train/test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dataset_info = analyze_training_dataset(X, y)
    recommended = recommend_training_models(dataset_info, problem_type)
    recommended = [n for n in recommended if n not in LIGHTWEIGHT_BLOCKED_MODELS]

    classification_models, regression_models = get_model_library()
    base_models = classification_models if problem_type == "classification" else regression_models
    models = {n: base_models[n] for n in recommended if n in base_models}
    models = filter_models(models, X, y, problem_type)

    if not models:
        return {"error": "No compatible models available for this dataset."}

    results = []
    best_model = None
    best_score = -999
    update_progress(40, "Training Models", "Starting model training...")

    start_time = datetime.now()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for name, model in models.items():
            futures.append(executor.submit(train_single_model, name, model, X_train, X_test, y_train, y_test, problem_type))
        for future in as_completed(futures):
            result = future.result()
            name, score = result["model"], result["score"]
            results.append({"model": name, "score": score, "time": result.get("time", 0), "metrics": result.get("metrics", {})})
            if score > best_score:
                best_score = score
                best_model = result["trained_model"]

    top_models = sorted(results, key=lambda x: x["score"], reverse=True)
    for i, m in enumerate(top_models):
        m["rank"] = i + 1
    best_model_name = top_models[0]["model"]
    update_progress(85, "Finalizing", "Selecting best model...")

    existing = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_v")]
    version = len(existing) + 1
    model_filename = f"model_v{version}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    leaderboard_data = save_leaderboard_snapshot(best_model_name, model_filename, top_models, "tabular", problem_type)
    joblib.dump({"model": best_model, "feature_columns": X.columns.tolist(), "problem_type": problem_type}, model_path)
    joblib.dump({"model": best_model, "feature_columns": X.columns.tolist(), "problem_type": problem_type}, MODEL_PATH)

    update_progress(95, "Saving", "Saving model and reports...")
    experiment = append_experiment_log(best_model_name, leaderboard_data["models"], model_filename, user_id, problem_type, "tabular", len(df), len(df.columns), float(best_score))
    save_model_record(user_id, best_model_name, model_filename, "tabular", float(best_score))

    quality = dataset_quality_score(df)
    strength = model_strength_summary(problem_type, {"accuracy": best_score if problem_type == "classification" else None, "r2": best_score if problem_type == "regression" else None})
    explanation = generate_explanation_text(problem_type, strength)
    report = {"dataset_quality": quality, "model_strength": strength, "explanation": explanation}
    joblib.dump(report, TRAINING_REPORT_PATH)

    ai_insights = generate_ai_insights(df, problem_type, type(best_model).__name__, best_score)
    generated_code = generate_training_pipeline_code(type(best_model).__name__, X.columns.tolist(), problem_type, target_column)
    with open(GENERATED_PIPELINE_PATH, "w") as f:
        f.write(generated_code)

    update_progress(100, "Completed", "Training completed successfully")

    result_data = {
        "dataset_type": "tabular", "problem_type": problem_type.title(),
        "rows": len(df), "target_column": target_column,
        "best_model": best_model_name,
        "top_models": top_models, "leaderboard": leaderboard_data["models"],
        "generated_code": generated_code, "model_version": model_filename,
        "ai_insights": ai_insights, "experiment": experiment,
    }
    if problem_type == "classification":
        result_data["accuracy"] = round(best_score, 4)
    else:
        result_data["r2"] = round(best_score, 4)
        result_data["mse"] = round(mean_squared_error(y_test, best_model.predict(X_test)), 4)
    return result_data

# ===== PREDICT =====
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH):
        return {"error": "No trained model found. Train a model first."}
    model_package = joblib.load(MODEL_PATH)
    problem_type = model_package.get("problem_type")
    model = model_package.get("model")
    feature_columns = model_package.get("feature_columns")
    if model is None or feature_columns is None:
        return {"error": "Invalid model package"}
    try:
        fn = (file.filename or "").lower()
        if fn.endswith(".csv"):
            df = pd.read_csv(file.file, sep=None, engine="python")
        elif fn.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        elif fn.endswith(".json"):
            df = pd.read_json(file.file)
        else:
            return {"error": "Unsupported file format"}
        df.columns = df.columns.str.strip().str.replace(r"[^\w]+", "_", regex=True)
        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            return {"error": "Missing required columns", "missing_columns": missing_cols}
        df = df[feature_columns]
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df = df.fillna(0)
        predictions = model.predict(df)
        if problem_type == "classification":
            results = [{"prediction": int(p)} for p in predictions]
            return {"problem_type": "Classification", "num_predictions": len(results), "predictions": results}
        else:
            return {"problem_type": "Regression", "num_predictions": len(predictions), "predictions": [round(float(p), 4) for p in predictions]}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ===== EXPERIMENTS =====
@app.get("/api/experiments")
def get_experiments(request: Request):
    user_id = extract_user_id_from_request(request)
    data = filter_experiments_for_user(load_experiment_logs(), user_id)
    normalized = [normalize_experiment_entry(item) for item in data]
    return {"total_experiments": len(normalized), "experiments": normalized[::-1]}

# ===== LEADERBOARD =====
@app.get("/api/leaderboard")
def get_leaderboard(request: Request):
    user_id = extract_user_id_from_request(request)
    user_experiments = filter_experiments_for_user(load_experiment_logs(), user_id)
    if user_experiments:
        latest = normalize_experiment_entry(user_experiments[-1])
        return build_leaderboard_payload(latest["best_model"], latest.get("model_version") or os.path.basename(MODEL_PATH), latest.get("leaderboard", []), latest.get("dataset_type"), latest.get("problem_type"))
    if not os.path.exists(LEADERBOARD_PATH):
        return {"error": "No leaderboard found. Train model first.", "models": []}
    return joblib.load(LEADERBOARD_PATH)

# ===== INSIGHTS =====
@app.get("/api/insights")
def get_insights(request: Request):
    user_id = extract_user_id_from_request(request)
    data = filter_experiments_for_user(load_experiment_logs(), user_id)
    if not data:
        return {"error": "No experiments to analyze"}
    scored = [e for e in data if isinstance(e.get("score"), (int, float))]
    best = max(scored, key=lambda x: x["score"]) if scored else data[-1]
    scores = [e["score"] for e in scored]
    return {
        "best_model": best.get("model_name", "N/A"),
        "best_score": round(best.get("score", 0), 4) if scores else None,
        "best_version": best.get("model_version"),
        "total_experiments": len(data),
        "average_score": round(sum(scores) / len(scores), 4) if scores else None,
    }

# ===== MODEL EXPLAIN =====
@app.get("/api/model-explain")
def model_explain():
    if not os.path.exists(MODEL_PATH):
        return {"error": "No trained model found"}
    model_package = joblib.load(MODEL_PATH)
    if "model" not in model_package:
        return {"error": "No tabular model found"}
    model = model_package["model"]
    feature_columns = model_package.get("feature_columns", [])
    if hasattr(model, "feature_importances_"):
        importance = dict(sorted(zip(feature_columns, model.feature_importances_), key=lambda x: x[1], reverse=True))
        return {"model_type": type(model).__name__, "feature_importance": importance}
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = coef[0]
        importance = dict(sorted(zip(feature_columns, coef), key=lambda x: abs(x[1]), reverse=True))
        return {"model_type": type(model).__name__, "coefficients": importance}
    return {"message": "This model type does not support explainability"}

# ===== DOWNLOAD =====
@app.get("/api/download-model")
def download_best_model():
    if os.path.exists(MODEL_PATH):
        return FileResponse(MODEL_PATH, filename="best_model.pkl")
    return {"error": "No trained model"}

@app.get("/api/download-model/{version}")
def download_model(version: str):
    safe = os.path.basename(version)
    path = os.path.join(MODEL_DIR, safe)
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "Model not found"}

@app.get("/api/download-code/{fmt}")
def download_code(fmt: str):
    if not os.path.exists(GENERATED_PIPELINE_PATH):
        return {"error": "No generated code found. Train a model first."}
    if fmt == "python":
        return FileResponse(GENERATED_PIPELINE_PATH, filename="pipeline.py")
    elif fmt == "notebook":
        with open(GENERATED_PIPELINE_PATH, "r") as f:
            code_lines = f.read().split("\n")
        notebook = {
            "cells": [{"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in code_lines], "outputs": [], "execution_count": None}],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5,
        }
        with open(GENERATED_NOTEBOOK_PATH, "w") as f:
            json.dump(notebook, f)
        return FileResponse(GENERATED_NOTEBOOK_PATH, filename="pipeline.ipynb")
    return {"error": "Format not supported"}

# ===== TRAINING REPORT =====
@app.get("/api/training-report")
def get_training_report():
    if not os.path.exists(TRAINING_REPORT_PATH):
        return {"error": "No report found. Train model first."}
    return joblib.load(TRAINING_REPORT_PATH)

# ===== TRAINING PROGRESS WS =====
@app.websocket("/api/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(training_progress)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return

# ===== AI CHAT =====
@app.post("/api/chat")
async def chat_ai(data: dict = Body(...)):
    message = data.get("message", "")
    dataset_info = data.get("dataset_info", "")

    if EMERGENT_LLM_KEY:
        try:
            from emergentintegrations.llm.chat import LlmChat, UserMessage
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"automl-chat-{uuid.uuid4().hex[:8]}",
                system_message="You are an expert data scientist and ML engineer helping users inside an AutoML dashboard. Provide concise, actionable advice about datasets, models, training strategies, feature engineering, and ML debugging."
            )
            chat.with_model("openai", "gpt-4.1-mini")
            prompt = f"User question: {message}\n\nDataset info:\n{dataset_info}\n\nAnswer like a professional ML expert helping inside an AutoML dashboard."
            user_msg = UserMessage(text=prompt)
            response = await chat.send_message(user_msg)
            return {"reply": response}
        except Exception as e:
            print(f"LLM Error: {e}")

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
        reply = "I am your AI assistant. Ask me about models, training, datasets, or ML errors and I'll help you improve your workflow."
    return {"reply": reply}

# ===== ADMIN =====
@app.get("/api/admin-stats")
async def admin_stats(request: Request):
    user_id = extract_user_id_from_request(request)
    if not user_id or not is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    return {
        "users": users_collection.count_documents({}),
        "models": models_collection.count_documents({}),
        "teams": teams_collection.count_documents({}),
        "usage_events": usage_collection.count_documents({}),
    }

print("AutoML X Backend started successfully!")
