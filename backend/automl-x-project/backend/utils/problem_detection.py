import pandas as pd


def detect_problem_type(df, target_column):
    target_data = df[target_column]
    if pd.api.types.is_numeric_dtype(target_data):
        unique_values = target_data.nunique()
        if unique_values < 20:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"
