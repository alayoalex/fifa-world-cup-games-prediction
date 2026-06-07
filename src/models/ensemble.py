"""
Ensemble of logistic regression (3-way) and Poisson-derived outcome probabilities.

Blends both models with a configurable weight (default 0.55 logistic / 0.45 Poisson),
renormalizes per row, and evaluates with the same temporal CV as the single models.

Run evaluation:
    uv run python src/models/ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.paths import PROCESSED_DIR
from models.baseline import CLASSES, score
from models.logistic import FEATURES, build_pipeline as build_logreg, predict_proba_ordered
from models.poisson import build_pipeline as build_poisson, fit_models, outcome_probs, score_matrix, DEFAULT_RHO
from models.splits import DEFAULT_VAL_YEARS, temporal_folds
from models.tracking import make_run_name, setup_mlflow

DEFAULT_LOGREG_WEIGHT = 0.55


def blend_proba(logreg_proba: np.ndarray, poisson_proba: np.ndarray, w: float) -> np.ndarray:
    """Convex blend of two (n, 3) probability matrices in CLASSES order, row-renormalized."""
    w = float(np.clip(w, 0.0, 1.0))
    blended = w * logreg_proba + (1.0 - w) * poisson_proba
    row_sums = blended.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return blended / row_sums


def poisson_proba_matrix(home_model, away_model, X: pd.DataFrame, rho: float = DEFAULT_RHO) -> np.ndarray:
    lam_h = home_model.predict(X)
    lam_a = away_model.predict(X)
    return np.array([
        [outcome_probs(score_matrix(lh, la, rho=rho))[c] for c in CLASSES]
        for lh, la in zip(lam_h, lam_a)
    ])


def predict_ensemble_proba(
    df: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    logreg_weight: float = DEFAULT_LOGREG_WEIGHT,
    logreg_C: float = 1.0,
    poisson_alpha: float = 0.1,
    rho: float = DEFAULT_RHO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logreg, poisson, ensemble) probability arrays for fixtures."""
    played = df[df["played"]]
    X_train = played[FEATURES].astype(float)
    X_fix = fixtures[FEATURES].astype(float)

    logreg = build_logreg(logreg_C).fit(X_train, played["result"])
    home_m, away_m = fit_models(df, alpha=poisson_alpha)

    p_log = predict_proba_ordered(logreg, X_fix)
    p_poi = poisson_proba_matrix(home_m, away_m, X_fix, rho=rho)
    p_ens = blend_proba(p_log, p_poi, logreg_weight)
    return p_log, p_poi, p_ens


def run_ensemble(df: pd.DataFrame, logreg_weight: float = DEFAULT_LOGREG_WEIGHT) -> pd.DataFrame:
    """Temporal CV for the blended model, logged to MLflow."""
    X = df[FEATURES].astype(float)
    with mlflow.start_run(run_name=make_run_name("ensemble", feature_set="v1")):
        mlflow.set_tag("phase", "4-ensemble")
        mlflow.log_params({
            "model": "logreg_poisson_blend",
            "logreg_weight": logreg_weight,
            "poisson_weight": 1.0 - logreg_weight,
            "cv_strategy": "temporal_expanding",
            "val_years": DEFAULT_VAL_YEARS,
        })

        rows = []
        for i, fold in enumerate(temporal_folds(df)):
            y_val = df.loc[fold.val_idx, "result"]
            logreg = build_logreg().fit(X.loc[fold.train_idx], df.loc[fold.train_idx, "result"])
            home_m = build_poisson().fit(
                X.loc[fold.train_idx], df.loc[fold.train_idx, "home_score"].astype(float)
            )
            away_m = build_poisson().fit(
                X.loc[fold.train_idx], df.loc[fold.train_idx, "away_score"].astype(float)
            )

            p_log = predict_proba_ordered(logreg, X.loc[fold.val_idx])
            p_poi = poisson_proba_matrix(home_m, away_m, X.loc[fold.val_idx])
            p_ens = blend_proba(p_log, p_poi, logreg_weight)

            metrics = score(y_val, p_ens)
            rows.append({"fold": fold.name, **metrics})
            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=i)

        per_fold = pd.DataFrame(rows)
        for key in ("logloss", "brier", "accuracy"):
            mlflow.log_metric(f"mean_{key}", float(per_fold[key].mean()))
        mlflow.log_table(per_fold, artifact_file="per_fold_metrics.json")
        return per_fold


def main() -> None:
    setup_mlflow()
    df = pd.read_parquet(PROCESSED_DIR / "matches_features.parquet")
    per_fold = run_ensemble(df, logreg_weight=DEFAULT_LOGREG_WEIGHT)
    print(per_fold.to_string(index=False))
    means = per_fold[["logloss", "brier", "accuracy"]].mean().round(4)
    print("mean", means.to_dict())


if __name__ == "__main__":
    main()
