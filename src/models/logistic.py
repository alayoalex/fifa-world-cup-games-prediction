"""
Phase 4 -- first real model: multinomial logistic regression, logged to MLflow.

Same temporal folds and metric schema as the baselines, so runs compare directly.
Two new MLflow capabilities here:
  - log matplotlib figures (calibration curve + confusion matrix)
  - log the fitted model to the Model Registry (our W&B-Artifacts replacement)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render figures to file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put src/ on path
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402

from etl.paths import PROCESSED_DIR
from models.baseline import CLASSES, score
from models.splits import DEFAULT_VAL_YEARS, temporal_folds
from models.tracking import make_run_name, setup_mlflow

FEATURES = [
    "elo_diff",
    "elo_exp_home",
    "form_pts_home",
    "form_pts_away",
    "form_gf_home",
    "form_ga_home",
    "form_gf_away",
    "form_ga_away",
    "fifa_rank_diff",
    "h2h_home_winrate",
    "rest_days_diff",''
    "mv_log_ratio",
    "neutral",
    "same_confederation",
]

def build_pipeline(C: float = 1.0) -> Pipeline:
    """Impute -> scale -> multinomial logistic regression.

    Fit INSIDE each fold (on train only): the imputer's medians and the scaler's
    mean/std are learned from the past alone -> no validation leakage.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000)),     # lbfgs -> multinomial
    ])


def predict_proba_ordered(model, X) -> np.ndarray:
    """predict_proba columns remapped to CLASSES=[H,D,A].

    sklearn orders predict_proba by sorted classes_ (=[A,D,H]); we remap so the
    columns mean what our metrics assume. Same label-order trap as log_loss.
    """
    proba = model.predict_proba(X)
    col = {c:i for i,c in enumerate(model.classes_)}
    return proba[:, [col[c] for c in CLASSES]]


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion (row-normalized = recall)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", 
                    color="white" if cm[i,j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_calibration(y_true, proba):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot([0,1], [0,1], "k--", lw=1, label="perfectly")
    yt = np.asarray(y_true)
    for k, cls in enumerate(CLASSES):
        frac, mean_pred = calibration_curve(
            (yt == cls).astype(int), proba[:,k], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=f"P({cls})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (one-vs-rest)")
    ax.legend()
    fig.tight_layout()
    return fig


def run_logreg(df, C: float = 1.0, feature_set: str = "v1") -> pd.DataFrame:
    X = df[FEATURES].astype(float)  # nullable booleans -> float for the imputer
    y = df["result"]
    with mlflow.start_run(run_name=make_run_name("logreg", feature_set=feature_set)):
        mlflow.set_tag("phase", "4-model")
        mlflow.log_params({
            "model": "logreg", "C": C, "solver": "lbfgs",
            "n_features": len(FEATURES), "features": ",".join(FEATURES),
            "imputation": "median", "scaling": "standard",
            "cv_strategy": "temporal_expanding", "val_years": DEFAULT_VAL_YEARS,
            "feature_set": feature_set,
        })

        rows, oof_y, oof_p = [], [], []
        for i, fold in enumerate(temporal_folds(df)):
            pipe = build_pipeline(C).fit(X.loc[fold.train_idx], y.loc[fold.train_idx])
            proba = predict_proba_ordered(pipe, X.loc[fold.val_idx])
            yv = y.loc[fold.val_idx]
            metrics = score(yv, proba)
            rows.append({"fold": fold.name, **metrics})
            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=i)    # per-fold curve
            oof_y.append(yv.to_numpy())
            oof_p.append(proba)

        per_fold = pd.DataFrame(rows)
        for key in ("logloss", "brier", "accuracy"):
            mlflow.log_metric(f"mean_{key}", float(per_fold[key].mean()))  # overall summary
        mlflow.log_table(per_fold, artifact_file="per_fold_metrics.json")

        # pooled out-of-fold predictions -> honest diagnostic figures
        y_oof = np.concatenate(oof_y)
        p_oof = np.concatenate(oof_p)
        pred_oof = np.array(CLASSES)[p_oof.argmax(axis=1)]
        mlflow.log_figure(plot_confusion(y_oof, pred_oof), "confusion_matrix.png")
        mlflow.log_figure(plot_calibration(y_oof, p_oof), "calibration.png")

        # deployable model: trained on ALL played matches -> Model Registry
        played = df[df["played"]]
        final = build_pipeline(C).fit(X.loc[played.index], y.loc[played.index])
        mlflow.sklearn.log_model(
            final, artifact_path="model",
            registered_model_name="wc2026-logreg",
            input_example=X.loc[played.index].head(3),
        )
        return per_fold


def main():
    setup_mlflow()
    df = pd.read_parquet(PROCESSED_DIR / "matches_features.parquet")
    per_fold = run_logreg(df, C=1.0, feature_set="v1")
    print(per_fold.to_string(index=False))
    means = per_fold[["logloss", "brier", "accuracy"]].mean().round(4)
    print("mean", means.to_dict())


if __name__ == "__main__":
    main()
