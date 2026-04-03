def generate_training_code(model_name, feature_columns, problem_type, target_column):
    code = f"""
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("your_dataset.csv")

target_column = "{target_column}"

X = df.drop(columns=[target_column])
y = df[target_column]

numeric_cols = X.select_dtypes(include=["int64","float64"]).columns

if len(numeric_cols) > 0:
    X[numeric_cols] = SimpleImputer(strategy="mean").fit_transform(X[numeric_cols])
    X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

categorical_cols = X.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    X[col] = X[col].astype(str)
    X[col] = LabelEncoder().fit_transform(X[col])

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
"""
    if problem_type == "classification":
        code += """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
"""
    else:
        code += """
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
"""
    code += """
model.fit(X_train, y_train)

joblib.dump(model, "trained_model.pkl")

print("Model trained and saved successfully!")
"""
    return code
