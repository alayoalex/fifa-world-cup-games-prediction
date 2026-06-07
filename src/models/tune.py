"""
Hyperparameter tuning with Optuna for the ensemble model.

Searches over 3 parameters simultaneously:
  - C          : LogisticRegression regularization (lower = more regularization)
  - alpha      : PoissonRegressor regularization
  - logreg_weight : ensemble blend weight for logistic vs Poisson

Optimal values are saved to data/processed/best_params.json and automatically
picked up by predict_all.py / ensemble.py.

Run:
    uv run python src/models/tune.py
    uv run python src/models/tune.py --trials 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.paths import PROCESSED_DIR
from models.baseline import CLASSES, score
from models.ensemble import blend_proba
from models.logistic import FEATURES, build_pipeline as build_logreg, predict_proba_ordered
from models.poisson import build_pipeline as build_poisson, outcome_probs, score_matrix, DEFAULT_RHO
from models.splits import temporal_folds

BEST_PARAMS_PATH = PROCESSED_DIR / "best_params.json"
DEFAULT_TRIALS = 80


def _poisson_proba(home_m, away_m, X: pd.DataFrame, rho: float = DEFAULT_RHO) -> np.ndarray:
    lam_h = home_m.predict(X)
    lam_a = away_m.predict(X)
    return np.array([
        [outcome_probs(score_matrix(lh, la, rho=rho))[c] for c in CLASSES]
        for lh, la in zip(lam_h, lam_a)
    ])


def objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    C = trial.suggest_float("C", 0.01, 20.0, log=True)
    alpha = trial.suggest_float("alpha", 0.001, 2.0, log=True)
    logreg_weight = trial.suggest_float("logreg_weight", 0.25, 0.80)
    rho = trial.suggest_float("rho", 0.0, 0.25)

    X = df[FEATURES].astype(float)
    fold_losses = []

    for fold in temporal_folds(df):
        X_tr, X_val = X.loc[fold.train_idx], X.loc[fold.val_idx]
        y_tr = df.loc[fold.train_idx, "result"]
        y_val = df.loc[fold.val_idx, "result"]
        y_tr_h = df.loc[fold.train_idx, "home_score"].astype(float)
        y_tr_a = df.loc[fold.train_idx, "away_score"].astype(float)

        logreg = build_logreg(C).fit(X_tr, y_tr)
        home_m = build_poisson(alpha).fit(X_tr, y_tr_h)
        away_m = build_poisson(alpha).fit(X_tr, y_tr_a)

        p_log = predict_proba_ordered(logreg, X_val)
        p_poi = _poisson_proba(home_m, away_m, X_val, rho=rho)
        p_ens = blend_proba(p_log, p_poi, logreg_weight)

        fold_losses.append(score(y_val, p_ens)["logloss"])

    return float(np.mean(fold_losses))


def tune(n_trials: int = DEFAULT_TRIALS) -> dict:
    df = pd.read_parquet(PROCESSED_DIR / "matches_features.parquet")
    played = df[df["played"]].copy()

    print(f"Tuning on {len(played):,} played matches, {n_trials} trials...")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, played), n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["logloss"] = round(study.best_value, 6)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")

    print(f"\nBest params (logloss={best['logloss']:.4f}):")
    for k, v in best.items():
        if k != "logloss":
            print(f"  {k:20s} = {v:.4f}")
    print(f"\nSaved -> {BEST_PARAMS_PATH}")
    return best


def load_best_params() -> dict:
    """Load saved params, or return defaults if not tuned yet."""
    defaults = {"C": 1.0, "alpha": 0.1, "logreg_weight": 0.55, "rho": DEFAULT_RHO}
    if BEST_PARAMS_PATH.exists():
        saved = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        return {**defaults, **saved}
    return defaults


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for ensemble")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args()
    tune(n_trials=args.trials)


if __name__ == "__main__":
    main()
