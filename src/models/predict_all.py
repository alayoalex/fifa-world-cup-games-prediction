"""
Unified local prediction pipeline — logistic + Poisson + ensemble in one CSV.

This is the recommended production entry point for stable, powerful predictions.

Run (after make_dataset.py):
    uv run python src/models/predict_all.py
    uv run python src/models/predict_all.py --custom
    uv run python src/etl/refresh_tournament.py   # download + rebuild + predict_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.custom_fixtures import CUSTOM_TOURNAMENT
from etl.paths import PROCESSED_DIR
from models.baseline import CLASSES
from models.ensemble import DEFAULT_LOGREG_WEIGHT, predict_ensemble_proba
from models.logistic import FEATURES, build_pipeline as build_logreg, predict_proba_ordered
from models.poisson import fit_models, predict_match
from models.predict_fixtures import _pick_custom_fixtures, _pick_fixtures
from models.tune import load_best_params

WC_OUTPUT = PROCESSED_DIR / "wc2026_predictions_full.csv"
CUSTOM_OUTPUT = PROCESSED_DIR / "custom_predictions_full.csv"


def _proba_frame(prefix: str, proba: np.ndarray, pick_col: str) -> pd.DataFrame:
    data = {f"{prefix}_p_{c}": proba[:, i] for i, c in enumerate(CLASSES)}
    data[pick_col] = [CLASSES[i] for i in proba.argmax(axis=1)]
    data[f"{prefix}_confidence"] = proba.max(axis=1).round(4)
    return pd.DataFrame(data)


def predict_full(
    df: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    logreg_weight: float = DEFAULT_LOGREG_WEIGHT,
    logreg_C: float = 1.0,
    poisson_alpha: float = 0.1,
    rho: float = 0.10,
) -> pd.DataFrame:
    """Build the full prediction table for a fixture set."""
    meta = fixtures[
        ["date", "home_team", "away_team", "city", "country", "neutral", "tournament"]
    ].reset_index(drop=True)

    p_log, p_poi, p_ens = predict_ensemble_proba(
        df, fixtures,
        logreg_weight=logreg_weight,
        logreg_C=logreg_C,
        poisson_alpha=poisson_alpha,
        rho=rho,
    )

    logreg_df = _proba_frame("logreg", p_log, "logreg_pick")
    poisson_df = _proba_frame("poisson", p_poi, "poisson_pick")
    ensemble_df = _proba_frame("ensemble", p_ens, "ensemble_pick")

    home_m, away_m = fit_models(df, alpha=poisson_alpha)
    scores = predict_match(home_m, away_m, fixtures[FEATURES].astype(float), rho=rho)

    return pd.concat([meta, scores, logreg_df, poisson_df, ensemble_df], axis=1)


def _save_and_print(preds: pd.DataFrame, output: Path, label: str) -> pd.DataFrame:
    output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)
    print(f"[predict_all] {len(preds)} {label} fixtures -> {output}")
    show_cols = [
        "date", "home_team", "away_team",
        "pred_home_goals", "pred_away_goals", "predicted_score", "score_result",
        "ensemble_pick", "ensemble_p_H", "ensemble_p_D", "ensemble_p_A",
    ]
    show = preds[[c for c in show_cols if c in preds.columns]]
    print(show.to_string(index=False))
    return preds


def predict_wc(
    *,
    year: int | None = 2026,
    tournament_only: bool = True,
    output: Path = WC_OUTPUT,
    logreg_weight: float = DEFAULT_LOGREG_WEIGHT,
) -> pd.DataFrame:
    store = PROCESSED_DIR / "matches_features.parquet"
    if not store.exists():
        raise FileNotFoundError(f"Missing {store}. Run make_dataset.py first.")

    df = pd.read_parquet(store)
    fixtures = _pick_fixtures(df, year=year, tournament_only=tournament_only)
    preds = predict_full(df, fixtures, logreg_weight=logreg_weight)
    return _save_and_print(preds, output, "WC")


def predict_custom(
    output: Path = CUSTOM_OUTPUT,
    logreg_weight: float = DEFAULT_LOGREG_WEIGHT,
) -> pd.DataFrame:
    store = PROCESSED_DIR / "matches_features.parquet"
    if not store.exists():
        raise FileNotFoundError(f"Missing {store}. Run add_fixture.py --predict first.")

    df = pd.read_parquet(store)
    fixtures = _pick_custom_fixtures(df)
    preds = predict_full(df, fixtures, logreg_weight=logreg_weight)
    return _save_and_print(preds, output, "custom")


def predict_everything(logreg_weight: float = DEFAULT_LOGREG_WEIGHT) -> None:
    """Score WC fixtures and custom fixtures (if any) in one run."""
    predict_wc(logreg_weight=logreg_weight)
    store = PROCESSED_DIR / "matches_features.parquet"
    df = pd.read_parquet(store)
    custom = df.loc[~df["played"] & (df["tournament"] == CUSTOM_TOURNAMENT)]
    if not custom.empty:
        predict_custom(logreg_weight=logreg_weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified predictions (logreg + Poisson + ensemble)")
    parser.add_argument("--custom", action="store_true", help="custom fixtures only")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--all-unplayed", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--logreg-weight", type=float, default=None)
    args = parser.parse_args()

    # Use tuned params if available, otherwise defaults
    params = load_best_params()
    logreg_weight = args.logreg_weight if args.logreg_weight is not None else params.get("logreg_weight", DEFAULT_LOGREG_WEIGHT)
    logreg_C = params.get("C", 1.0)
    poisson_alpha = params.get("alpha", 0.1)
    rho = params.get("rho", 0.10)

    if args.custom:
        predict_custom(
            output=args.output or CUSTOM_OUTPUT,
            logreg_weight=logreg_weight,
        )
    else:
        predict_wc(
            year=None if args.all_unplayed else args.year,
            tournament_only=not args.all_unplayed,
            output=args.output or WC_OUTPUT,
            logreg_weight=logreg_weight,
        )


if __name__ == "__main__":
    main()
