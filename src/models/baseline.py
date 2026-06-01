"""
Phase 4 baselines, logged to MLflow.

Two zero-training strategies, scored on identical temporal folds:
  - baseline-majority : predict the training-fold class base rates (H/D/A).
  - baseline-elo      : turn the Elo expected score into 3-way probabilities
                        using the training-fold draw rate (no model fitted).

Both are the numbers every Phase-4 model must beat.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # puth src/ on path
import mlflow

from etl.paths import PROCESSED_DIR
from models.splits import temporal_folds, DEFAULT_VAL_YEARS
from models.tracking import setup_mlflow, make_run_name

CLASSES = ["H", "D", "A"]   # fixed column order for every probability array

# --- metrics ---
def multiclass_brier(y_true, proba) -> float:
    """Mean over samples of sum_k (p_k - y_k)^2 -- the multiclass Brier score."""
    onehot = pd.get_dummies(pd.Categorical(y_true, categories=CLASSES)).to_numpy()
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def score(y_true, proba) -> dict:
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    # sklearn's log_loss IGNORES the order of `labels` and assumes y_prob columns
    # are in lexicographic class order. Our columns are CLASSES=[H,D,A], so reorder
    # BOTH columns and labels to sorted [A,D,H] -- otherwise it silently maps the
    # wrong probability to each class (and warns).
    order = np.argsort(CLASSES)                 # [H,D,A] -> [2,1,0] == [A,D,H]
    sorted_labels = list(np.array(CLASSES)[order])
    return {
        "logloss": float(log_loss(y_true, proba, labels=sorted_labels)),
        "brier": multiclass_brier(y_true, proba),
        "accuracy": float(accuracy_score(y_true, pred)),
    }

# --- the two strategies (each returns an (n_val, 3) probability array) ---
def predict_majority(train, val) -> np.ndarray:
    rates = train["result"].value_counts(normalize=True)
    p = np.array([rates.get(c, 0.0) for c in CLASSES])
    return np.tile(p, (len(val), 1))


def predict_elo(train, val) -> np.ndarray:
    d = float((train["result"] == "D").mean())   # training-fold draw rate
    e = val["elo_exp_home"].fillna(0.5).to_numpy()  # venue-aware Elo expectation
    prob = np.column_stack([e - 0.5*d, np.full_like(e, d), 1.0 - e - 0.5*d])
    proba = np.clip(prob, 1e-6, None)
    return proba / proba.sum(axis=1, keepdims=True)  # renormalize -> valid probs


# --- the logged run harness ------------------------------------------------
def run_baseline(df, name, predict_fn, feature_set) -> pd.DataFrame:
    with mlflow.start_run(run_name=make_run_name(name, feature_set=feature_set)):
        mlflow.set_tag("phase", "4-baseline")
        mlflow.log_param("strategy", name)
        mlflow.log_param("cv_strategy", "temporal_expanding")
        mlflow.log_param("val_years", DEFAULT_VAL_YEARS)
        mlflow.log_param("feature_set", feature_set)

        rows = []
        for i, fold in enumerate(temporal_folds(df)):
            train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
            metrics = score(val["result"], predict_fn(train, val))
            rows.append({"fold": fold.name, **metrics})
            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=i)    # per-fold curve

        per_fold = pd.DataFrame(rows)
        for key in ("logloss", "brier", "accuracy"):
            mlflow.log_metric(f"mean_{key}", float(per_fold[key].mean()))  # overall summary
        mlflow.log_table(per_fold, artifact_file="per_fold_metrics.json")
        return per_fold


def main():
    setup_mlflow()
    df = pd.read_parquet(PROCESSED_DIR / "matches_features.parquet")
    for name, fn, feature_set in [
        ("baseline-majority", predict_majority, "v1"),
        ("baseline-elo", predict_elo, "v1"),
    ]:
        per_fold = run_baseline(df, name, fn, feature_set)
        print(f"\n=== {name} ===")
        print(per_fold.to_string(index=False))
        means = per_fold[["logloss", "brier", "accuracy"]].mean().round(4)
        print("mean", means.to_dict())

if __name__ == "__main__":
    main()
