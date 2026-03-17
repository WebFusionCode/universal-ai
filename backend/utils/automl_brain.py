import numpy as np


def analyze_dataset(X, y):

    info = {}

    info["rows"] = len(X)
    info["columns"] = X.shape[1]
    info["missing_ratio"] = float(X.isna().sum().sum()) / (X.shape[0] * X.shape[1])
    info["unique_targets"] = len(np.unique(y))

    return info


def recommend_models(dataset_info, problem_type):

    models = []

    rows = dataset_info["rows"]
    cols = dataset_info["columns"]

    # SMALL DATASET
    if rows < 1000:
        models += ["LogisticRegression", "KNN"]

    # MEDIUM DATASET
    if 1000 <= rows < 10000:
        models += ["RandomForest"]

    # LARGE DATASET
    if rows >= 10000:
        models += ["LightGBM"]

    # HIGH DIMENSION
    if cols > 20:
        models += ["TabTransformer"]

    # ALWAYS GOOD MODELS
    models += ["RandomForest"]

    return list(set(models))