import os
import joblib
import pandas as pd

MODEL_PATH = os.path.basename("best_model.pkl")
model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
features = model_package.get("feature_columns", ['poisson_1', 'binomial_3', 'bernoulli_1', 'triangular_3', 'multivariatenormal_2_x_binned_target_enc', 'bernoulli_3_1_target_enc', 'hypergeometric_3_target_enc', 'cauchy_6', 'negativebinomial_1', 'beta_3', 'hypergeometric_4', 'multivariatenormal_2_y', 'triangular_1', 'negativebinomial_5_target_enc', 'binomial_5', 'exponential_2', 'chisquare_3', 'multinomial_3_x', 'poisson_2', 'uniform_4', 'beta_6', 'multivariatenormal_2_x_gamma_5', 'logistic_5', 'pareto_3', 'cauchy_3', 'multivariatenormal_2_y_binned', 'geometric_1_target_enc', 'negativebinomial_5', 'geometric_1', 'binomial_2_target_enc', 'negativebinomial_3_target_enc', 'multivariatenormal_2_x', 'gamma_5', 'binomial_2', 't_dist_5', 'multivariatenormal_2_x_binned', 'bernoulli_2', 'bernoulli_2_target_enc', 'weibull_5', 'negativebinomial_3', 'geometric_2_target_enc', 'multinomial_3_y', 'multivariatenormal_2_y_gamma_5', 'multinomial_1_x', 'binomial_3_target_enc', 'zipf_3', 'binomial_6', 'negativebinomial_1_target_enc', 'bernoulli_3_1', 'bernoulli_1_target_enc', 'gamma_5_2', 'numerical_target', 'multivariatenormal_2_y_binned_target_enc', 'binary_target'])


def preprocess(df):
    if features:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df[features]
    return df


def predict(data_input):
    if isinstance(data_input, str):
        df = pd.read_csv(data_input)
    elif isinstance(data_input, dict):
        df = pd.DataFrame([data_input])
    else:
        df = data_input

    if isinstance(df, pd.Series):
        df = df.to_frame().T

    df = preprocess(df)
    preds = model.predict(df)
    return preds.tolist()


if __name__ == "__main__":
    print("Model ready")
