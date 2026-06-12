"""
Generate local WC fixture predictions — no network, no external services.

Trains on all played matches in the local feature store, scores unplayed
fixtures (2026 World Cup by default), and writes a CSV you can open in Excel,
a notebook, or any local tool.

Run (after make_dataset.py):
    uv run python src/models/predict_fixtures.py
    uv run python src/models/predict_fixtures.py --year 2026 --tournament-only
    uv run python src/models/predict_fixtures.py --output data/processed/my_preds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.custom_fixtures import CUSTOM_TOURNAMENT
from etl.paths import PROCESSED_DIR
from models.baseline import CLASSES
from models.logistic import FEATURES, build_pipeline, predict_proba_ordered


DEFAULT_OUTPUT = PROCESSED_DIR / "wc2026_predictions.csv"
CUSTOM_OUTPUT = PROCESSED_DIR / "custom_predictions.csv"


def _pick_fixtures(
    df: pd.DataFrame,
    *,
    year: int | None,
    tournament_only: bool,
) -> pd.DataFrame:
    mask = ~df["played"]
    if year is not None:
        mask &= df["date"].dt.year == year
    if tournament_only:
        mask &= df["is_world_cup"].fillna(False)
    fixtures = df.loc[mask].copy()
    if fixtures.empty:
        raise ValueError(
            "No unplayed fixtures matched the filters. "
            "Run `uv run python src/etl/make_dataset.py` first."
        )
    return fixtures.sort_values("date")


def _pick_custom_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    fixtures = df.loc[~df["played"] & (df["tournament"] == CUSTOM_TOURNAMENT)].copy()
    if fixtures.empty:
        raise ValueError(
            "No custom fixtures in the feature store. "
            "Add one with: uv run python src/etl/add_fixture.py add --home X --away Y --predict"
        )
    return fixtures.sort_values("date")


def predict_fixtures(
    df: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    C: float = 1.0,
) -> pd.DataFrame:
    played = df[df["played"]]
    X_train = df.loc[played.index, FEATURES].astype(float)
    y_train = df.loc[played.index, "result"]
    X_fix = fixtures[FEATURES].astype(float)

    model = build_pipeline(C).fit(X_train, y_train)
    proba = predict_proba_ordered(model, X_fix)

    out = fixtures[
        ["date", "home_team", "away_team", "city", "country", "neutral", "tournament"]
    ].copy()
    for i, cls in enumerate(CLASSES):
        out[f"p_{cls}"] = proba[:, i]
    out["predicted"] = [CLASSES[i] for i in proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(4)
    return out


def main(
    year: int | None = 2026,
    tournament_only: bool = True,
    output: Path = DEFAULT_OUTPUT,
    C: float = 1.0,
) -> pd.DataFrame:
    feature_store = PROCESSED_DIR / "matches_features.parquet"
    if not feature_store.exists():
        raise FileNotFoundError(
            f"Missing {feature_store}. Run: uv run python src/etl/make_dataset.py"
        )

    df = pd.read_parquet(feature_store)
    fixtures = _pick_fixtures(df, year=year, tournament_only=tournament_only)
    preds = predict_fixtures(df, fixtures, C=C)

    output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)

    print(f"[predict_fixtures] {len(preds)} fixtures scored")
    print(f"[predict_fixtures] -> {output}")
    print(preds[["date", "home_team", "away_team", "predicted", "p_H", "p_D", "p_A"]].to_string(index=False))
    return preds


def predict_custom(output: Path | None = None, C: float = 1.0) -> pd.DataFrame:
    """Score user-defined hypothetical fixtures from the local feature store."""
    feature_store = PROCESSED_DIR / "matches_features.parquet"
    if not feature_store.exists():
        raise FileNotFoundError(
            f"Missing {feature_store}. Run add_fixture.py with --predict first."
        )

    df = pd.read_parquet(feature_store)
    fixtures = _pick_custom_fixtures(df)
    preds = predict_fixtures(df, fixtures, C=C)

    out_path = output or CUSTOM_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)

    print(f"[predict_custom] {len(preds)} custom fixtures scored")
    print(f"[predict_custom] -> {out_path}")
    print(preds[["date", "home_team", "away_team", "predicted", "p_H", "p_D", "p_A"]].to_string(index=False))
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local WC fixture predictions (CSV output)")
    parser.add_argument("--year", type=int, default=2026, help="fixture year (default: 2026)")
    parser.add_argument(
        "--all-unplayed", action="store_true",
        help="include every unplayed match, not only World Cup fixtures",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output CSV path")
    parser.add_argument("--C", type=float, default=1.0, help="logistic regression regularization")
    parser.add_argument(
        "--custom", action="store_true",
        help="score custom hypothetical fixtures only",
    )
    args = parser.parse_args()
    if args.custom:
        predict_custom(output=args.output, C=args.C)
    else:
        main(
            year=None if args.all_unplayed else args.year,
            tournament_only=not args.all_unplayed,
            output=args.output,
            C=args.C,
        )
