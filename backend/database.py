import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017").strip()

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client["automl_db"]

users_collection = db["users"]
models_collection = db["models"]
subscriptions_collection = db["subscriptions"]
teams_collection = db["teams"]
usage_collection = db["usage"]
