import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess import engineer_features, get_feature_columns


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)


def train():
    raw = fetch_california_housing(as_frame=True)
    df = raw.frame
    df = engineer_features(df)
    features = get_feature_columns()

    X = df[features]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    mlflow.set_experiment("california-house-price")

    with mlflow.start_run():
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: objective(t, X_tr, y_tr, X_val, y_val),
            n_trials=30, show_progress_bar=True
        )

        best_params = {**study.best_params,
                       "objective": "regression", "metric": "rmse",
                       "verbosity": -1}
        mlflow.log_params(best_params)

        final_model = lgb.train(
            best_params,
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=500
        )

        preds = final_model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.lightgbm.log_model(final_model, "model")

        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")


if __name__ == "__main__":
    train()
