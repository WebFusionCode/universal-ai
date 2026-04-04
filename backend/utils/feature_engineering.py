import pandas as pd

import numpy as np

DATE_COLUMN_HINTS = ("date", "time", "year")


def auto_feature_engineering(df):

    df = df.copy()

    df.columns = df.columns.str.strip().str.replace(r"[^\w]+", "_", regex=True)

    new_features = {}

    for col in df.columns:
        if not any(hint in col.lower() for hint in DATE_COLUMN_HINTS):
            continue

        try:
            converted = pd.to_datetime(df[col], errors="coerce")

            if converted.notna().sum() > len(df) * 0.7:
                new_features[col + "_year"] = converted.dt.year

                new_features[col + "_month"] = converted.dt.month

                new_features[col + "_day"] = converted.dt.day

                new_features[col + "_weekday"] = converted.dt.weekday

        except:
            pass

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if df[col].nunique() > 10:
            new_features[col + "_log"] = np.log1p(np.abs(df[col]))

            new_features[col + "_square"] = df[col] ** 2

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for i in range(min(len(numeric_cols), 5)):
        for j in range(i + 1, min(len(numeric_cols), 5)):
            col1 = numeric_cols[i]

            col2 = numeric_cols[j]

            new_features[f"{col1}_x_{col2}"] = df[col1] * df[col2]

    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        freq = df[col].value_counts()

        new_features[col + "_freq"] = df[col].map(freq)

    if new_features:
        df = pd.concat([df, pd.DataFrame(new_features)], axis=1)

    df = df.copy()

    return df
