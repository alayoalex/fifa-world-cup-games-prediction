"""
Predict exact scorelines for unplayed fixtures using the bivariate Poisson model.

Outputs expected goals (lambda), most likely score, top-3 scorelines, and H/D/A
probabilities derived from the score matrix.

Run (after make_dataset.py):
    uv run python src/models/predict_scores.py
    uv run python src/models/predict_scores.py --custom
    uv run python src/models/predict_scores.py --output data/processed/my_scores.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.paths import PROCESSED_DIR
from models.poisson import fit_models, predict_match
from models.predict_fixtures import _pick_custom_fixtures, _pick_fixtures


WC_SCORES_OUTPUT = PROCESSED_DIR / "wc2026_score_predictions.csv"
CUSTOM_SCORES_OUTPUT = PROCESSED_DIR / "custom_score_predictions.csv"


def predict_scorelines(
    df: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    alpha: float = 0.1,
) -> pd.DataFrame:
    from models.logistic import FEATURES

    home_model, away_model = fit_models(df, alpha=alpha)
    X = fixtures[FEATURES].astype(float)
    scored = predict_match(home_model, away_model, X)
    meta = fixtures[
        ["date", "home_team", "away_team", "city", "country", "neutral", "tournament"]
    ].reset_index(drop=True)
    return pd.concat([meta, scored], axis=1)


def main(
    *,
    year: int | None = 2026,
    tournament_only: bool = True,
    output: Path = WC_SCORES_OUTPUT,
    alpha: float = 0.1,
) -> pd.DataFrame:
    feature_store = PROCESSED_DIR / "matches_features.parquet"
    if not feature_store.exists():
        raise FileNotFoundError(
            f"Missing {feature_store}. Run: uv run python src/etl/make_dataset.py"
        )

    df = pd.read_parquet(feature_store)
    fixtures = _pick_fixtures(df, year=year, tournament_only=tournament_only)
    preds = predict_scorelines(df, fixtures, alpha=alpha)

    output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)

    print(f"[predict_scores] {len(preds)} fixtures scored")
    print(f"[predict_scores] -> {output}")
    show = preds[
        ["date", "home_team", "away_team", "predicted_score", "lambda_home",
         "lambda_away", "p_score", "predicted_result"]
    ]
    print(show.to_string(index=False))
    return preds


def predict_custom_scores(output: Path | None = None, alpha: float = 0.1) -> pd.DataFrame:
    feature_store = PROCESSED_DIR / "matches_features.parquet"
    if not feature_store.exists():
        raise FileNotFoundError(
            f"Missing {feature_store}. Run add_fixture.py with --predict first."
        )

    df = pd.read_parquet(feature_store)
    fixtures = _pick_custom_fixtures(df)
    preds = predict_scorelines(df, fixtures, alpha=alpha)

    out_path = output or CUSTOM_SCORES_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)

    print(f"[predict_custom_scores] {len(preds)} fixtures scored")
    print(f"[predict_custom_scores] -> {out_path}")
    show = preds[
        ["date", "home_team", "away_team", "predicted_score", "lambda_home",
         "lambda_away", "p_score", "predicted_result"]
    ]
    print(show.to_string(index=False))
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poisson scoreline predictions (local CSV)")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--all-unplayed", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    if args.custom:
        predict_custom_scores(output=args.output, alpha=args.alpha)
    else:
        main(
            year=None if args.all_unplayed else args.year,
            tournament_only=not args.all_unplayed,
            output=args.output or WC_SCORES_OUTPUT,
            alpha=args.alpha,
        )
