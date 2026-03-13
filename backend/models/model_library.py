from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost
import lightgbm
import catboost
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

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

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05
    ),

    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0
    )
}


REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=200
    ),

    "GradientBoosting": GradientBoostingRegressor(),

    "SVR": SVR(),

    "KNN": KNeighborsRegressor(),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0
    )
}