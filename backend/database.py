import os

try:
    from dotenv import load_dotenv
except Exception:

    def load_dotenv(*args, **kwargs):
        return False


try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

from types import SimpleNamespace

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017").strip()


class NullCursor(list):
    def sort(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self


class NullCollection:
    def insert_one(self, *args, **kwargs):
        return None

    def find_one(self, *args, **kwargs):
        return None

    def update_one(self, *args, **kwargs):
        return SimpleNamespace(modified_count=0)

    def find(self, *args, **kwargs):
        return NullCursor()

    def count_documents(self, *args, **kwargs):
        return 0


if MongoClient is None:
    users_collection = NullCollection()
    models_collection = NullCollection()
    subscriptions_collection = NullCollection()
    teams_collection = NullCollection()
    usage_collection = NullCollection()
else:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["automl_db"]

    users_collection = db["users"]
    models_collection = db["models"]
    subscriptions_collection = db["subscriptions"]
    teams_collection = db["teams"]
    usage_collection = db["usage"]
