import pandas as pd
import numpy as np


def auto_feature_engineering(df):

    df = df.copy()

    # ==========================
    # CLEAN COLUMN NAMES
    # ==========================

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
    )

    new_features = {}

    # ==========================
    # DATE FEATURES
    # ==========================

    for col in df.columns:

        try:
            converted = pd.to_datetime(df[col], errors="coerce", format="mixed")

            if converted.notna().sum() > len(df) * 0.7:

                new_features[col + "_year"] = converted.dt.year
                new_features[col + "_month"] = converted.dt.month
                new_features[col + "_day"] = converted.dt.day
                new_features[col + "_weekday"] = converted.dt.weekday

        except:
            pass

    # ==========================
    # NUMERIC FEATURES
    # ==========================

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:

        if df[col].nunique() > 10:

            new_features[col + "_log"] = np.log1p(np.abs(df[col]))
            new_features[col + "_square"] = df[col] ** 2

    # ==========================
    # INTERACTION FEATURES
    # ==========================

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for i in range(min(len(numeric_cols), 5)):
        for j in range(i + 1, min(len(numeric_cols), 5)):

            col1 = numeric_cols[i]
            col2 = numeric_cols[j]

            new_features[f"{col1}_x_{col2}"] = df[col1] * df[col2]

    # ==========================
    # CATEGORICAL FREQUENCY
    # ==========================

    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:

        freq = df[col].value_counts()
        new_features[col + "_freq"] = df[col].map(freq)

    # ==========================
    # ADD FEATURES AT ONCE
    # ==========================

    if new_features:
        df = pd.concat([df, pd.DataFrame(new_features)], axis=1)

    df = df.copy()

    return df