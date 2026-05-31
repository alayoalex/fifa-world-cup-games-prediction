"""Throwaway: confirm MLflow logs to our local SQLite store. Delete after"""

import mlflow

# Point MLflow at a local SQLite file (created on first use).
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("smoke-test")

with mlflow.start_run(run_name="hello-mlflow"):
    mlflow.log_param("learning_rate", 0.1)   # a hyperparameter
    mlflow.log_metric("log_loss", 0.42)      # a result
    print("Logged one run. Now lunch the UI to see it")
