from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000),

    "RandomForest": RandomForestClassifier(
        n_estimators=200
    ),

    "GradientBoosting": GradientBoostingClassifier(),

    "SVM": SVC(
        probability=True
    ),

    "KNN": KNeighborsClassifier(),
}


REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=200
    ),

    "GradientBoosting": GradientBoostingRegressor(),

    "SVR": SVR(),

    "KNN": KNeighborsRegressor(),
}

if XGBClassifier is not None:
    CLASSIFICATION_MODELS["XGBoost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    )

if LGBMClassifier is not None:
    CLASSIFICATION_MODELS["LightGBM"] = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05
    )

if CatBoostClassifier is not None:
    CLASSIFICATION_MODELS["CatBoost"] = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0
    )

if XGBRegressor is not None:
    REGRESSION_MODELS["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )

if LGBMRegressor is not None:
    REGRESSION_MODELS["LightGBM"] = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05
    )

if CatBoostRegressor is not None:
    REGRESSION_MODELS["CatBoost"] = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0
    )
