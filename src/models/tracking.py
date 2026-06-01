"""
Central MLflow configuration for all Phase 3+ experiments.

Single source of truth for:
    - WHERE run are store (local SQLite store + artifact dir)
    - WHICH experiment they belong to
    - HOW run are named: {model}_{feature_set}_{cv}_{date}

Every traning script calls setup_mlflow() once at startup so the scripts
and `mlflow ui` always agree on the backend (the #1 "my runs vanished").
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import mlflow

PROJECT_DIR = Path(__file__).resolve().parents[2]
MLFLOW_DIR = PROJECT_DIR / "mlflow"
DB_PATH = MLFLOW_DIR / "mlflow.db"
ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"

TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
ARTIFACTS_URI = ARTIFACTS_DIR.as_uri()
EXPERIMENT_NAME = "wc2026-prediction"

def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACTS_URI)
    mlflow.set_experiment(experiment_name)
    return experiment_name

def make_run_name(model: str, feature_set: str = "v1", cv: str = "temporalcv") -> str:
    date = _dt.date.today().isoformat()
    return f"{model}_{feature_set}_{cv}_{date}"


if __name__ == "__main__":
    # Sanity check + the exact command to view these runs.
    setup_mlflow()
    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Artifacts URI: {ARTIFACTS_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Example run: {make_run_name('logreg')}")
    print("\nView the UI with:")
    print(f' mlflow ui --backend-store-uri "{TRACKING_URI}"')
