from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def tune_random_forest(X, y, problem_type):
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(
            "Optuna is not installed. Add it back to enable hyperparameter tuning."
        ) from exc

    def objective(trial):

        n_estimators = trial.suggest_int("n_estimators", 50, 300)

        max_depth = trial.suggest_int("max_depth", 3, 20)

        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        if problem_type == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )

            score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()

        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )

            score = cross_val_score(model, X, y, cv=3, scoring="r2").mean()

        return score

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=20)

    return study.best_params
