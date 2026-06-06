"""
Bivariate Poisson model for expected goals and exact scorelines.

Two independent Poisson regressions (home goals, away goals) on the same feature
set as logistic regression. Scoreline probabilities come from the outer product of
Poisson PMFs; H/D/A probabilities are aggregated from that matrix.

Run evaluation (temporal CV + MLflow):
    uv run python src/models/poisson.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.paths import PROCESSED_DIR
from models.baseline import CLASSES, score as score_3way
from models.logistic import FEATURES
from models.splits import DEFAULT_VAL_YEARS, temporal_folds
from models.tracking import make_run_name, setup_mlflow

MAX_GOALS = 6


def build_pipeline(alpha: float = 0.1) -> Pipeline:
    """Impute -> scale -> Poisson regression (predicts expected goals)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", PoissonRegressor(alpha=alpha, max_iter=2000)),
    ])


def score_matrix(lam_home: float, lam_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """(max_goals+1) x (max_goals+1) matrix of P(home=h, away=a)."""
    lam_home = max(float(lam_home), 1e-6)
    lam_away = max(float(lam_away), 1e-6)
    goals = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))


def outcome_probs(matrix: np.ndarray) -> dict[str, float]:
    """Aggregate score matrix into H / D / A probabilities."""
    p_h = float(np.tril(matrix, k=-1).sum())
    p_d = float(np.trace(matrix))
    p_a = float(np.triu(matrix, k=1).sum())
    total = p_h + p_d + p_a
    if total <= 0:
        return {c: 1.0 / 3 for c in CLASSES}
    return {CLASSES[0]: p_h / total, CLASSES[1]: p_d / total, CLASSES[2]: p_a / total}


def most_likely_score(matrix: np.ndarray) -> tuple[int, int, float]:
    """Return (home_goals, away_goals, probability) for the modal scoreline."""
    idx = np.unravel_index(matrix.argmax(), matrix.shape)
    return int(idx[0]), int(idx[1]), float(matrix[idx])


def top_scorelines(matrix: np.ndarray, n: int = 3) -> list[tuple[int, int, float]]:
    """Top-n scorelines by probability."""
    flat = matrix.ravel()
    order = np.argsort(flat)[::-1][:n]
    rows = []
    width = matrix.shape[1]
    for k in order:
        h, a = divmod(int(k), width)
        rows.append((h, a, float(flat[k])))
    return rows


def format_score(home_goals: int, away_goals: int) -> str:
    """Excel-safe score string (avoids '2-0' being parsed as a date)."""
    return f"{home_goals}x{away_goals}"


def format_top_scores(lines: list[tuple[int, int, float]]) -> str:
    return "|".join(f"{format_score(h, a)}:{p:.3f}" for h, a, p in lines)


def result_from_goals(home_goals: int, away_goals: int) -> str:
    """Map a scoreline to H / D / A."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def predict_match(home_model: Pipeline, away_model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Predict lambdas, modal scoreline, and 3-way probs for one or more fixtures."""
    lam_h = home_model.predict(X)
    lam_a = away_model.predict(X)

    rows = []
    for lh, la in zip(lam_h, lam_a):
        mat = score_matrix(lh, la)
        probs = outcome_probs(mat)
        h, a, p_score = most_likely_score(mat)
        top = top_scorelines(mat, n=3)
        score_result = result_from_goals(h, a)
        rows.append({
            "pred_home_goals": h,
            "pred_away_goals": a,
            "predicted_score": format_score(h, a),
            "lambda_home": round(float(lh), 3),
            "lambda_away": round(float(la), 3),
            "p_score": round(p_score, 4),
            "score_result": score_result,
            "predicted_result": score_result,
            "outcome_pick": max(probs, key=probs.get),
            "p_H": round(probs["H"], 4),
            "p_D": round(probs["D"], 4),
            "p_A": round(probs["A"], 4),
            "top_scores": format_top_scores(top),
        })
    return pd.DataFrame(rows)


def _fold_metrics(y_home, y_away, lam_h, lam_a, result) -> dict:
    proba = np.array([
        [outcome_probs(score_matrix(lh, la))[c] for c in CLASSES]
        for lh, la in zip(lam_h, lam_a)
    ])
    mae_h = mean_absolute_error(y_home, lam_h)
    mae_a = mean_absolute_error(y_away, lam_a)
    dev_h = mean_poisson_deviance(y_home, lam_h)
    dev_a = mean_poisson_deviance(y_away, lam_a)
    three = score_3way(result, proba)
    return {
        "mae_home": float(mae_h),
        "mae_away": float(mae_a),
        "poisson_dev_home": float(dev_h),
        "poisson_dev_away": float(dev_a),
        **three,
    }


def fit_models(df: pd.DataFrame, alpha: float = 0.1) -> tuple[Pipeline, Pipeline]:
    """Train home/away Poisson models on all played matches."""
    played = df[df["played"]]
    X = played[FEATURES].astype(float)
    home = build_pipeline(alpha).fit(X, played["home_score"].astype(float))
    away = build_pipeline(alpha).fit(X, played["away_score"].astype(float))
    return home, away


def run_poisson(df: pd.DataFrame, alpha: float = 0.1, feature_set: str = "v1") -> pd.DataFrame:
    """Temporal CV evaluation logged to MLflow."""
    X = df[FEATURES].astype(float)
    with mlflow.start_run(run_name=make_run_name("poisson", feature_set=feature_set)):
        mlflow.set_tag("phase", "4-model")
        mlflow.log_params({
            "model": "poisson_bivariate",
            "alpha": alpha,
            "max_goals": MAX_GOALS,
            "n_features": len(FEATURES),
            "features": ",".join(FEATURES),
            "cv_strategy": "temporal_expanding",
            "val_years": DEFAULT_VAL_YEARS,
            "feature_set": feature_set,
        })

        rows = []
        for i, fold in enumerate(temporal_folds(df)):
            y_h = df.loc[fold.val_idx, "home_score"].astype(float)
            y_a = df.loc[fold.val_idx, "away_score"].astype(float)
            result = df.loc[fold.val_idx, "result"]

            home_m = build_pipeline(alpha).fit(
                X.loc[fold.train_idx], df.loc[fold.train_idx, "home_score"].astype(float)
            )
            away_m = build_pipeline(alpha).fit(
                X.loc[fold.train_idx], df.loc[fold.train_idx, "away_score"].astype(float)
            )
            lam_h = home_m.predict(X.loc[fold.val_idx])
            lam_a = away_m.predict(X.loc[fold.val_idx])
            metrics = _fold_metrics(y_h, y_a, lam_h, lam_a, result)
            rows.append({"fold": fold.name, **metrics})
            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=i)

        per_fold = pd.DataFrame(rows)
        for key in per_fold.columns:
            if key == "fold":
                continue
            mlflow.log_metric(f"mean_{key}", float(per_fold[key].mean()))
        mlflow.log_table(per_fold, artifact_file="per_fold_metrics.json")

        played = df[df["played"]]
        home_final, away_final = fit_models(df, alpha=alpha)
        mlflow.sklearn.log_model(
            home_final, artifact_path="model_home",
            registered_model_name="wc2026-poisson-home",
            input_example=X.loc[played.index].head(3),
        )
        mlflow.sklearn.log_model(
            away_final, artifact_path="model_away",
            registered_model_name="wc2026-poisson-away",
            input_example=X.loc[played.index].head(3),
        )
        return per_fold


def main() -> None:
    setup_mlflow()
    df = pd.read_parquet(PROCESSED_DIR / "matches_features.parquet")
    per_fold = run_poisson(df, alpha=0.1, feature_set="v1")
    print(per_fold.to_string(index=False))
    summary = per_fold.drop(columns="fold").mean().round(4)
    print("mean", summary.to_dict())


if __name__ == "__main__":
    main()
