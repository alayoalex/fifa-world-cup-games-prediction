"""
Record real match results during the tournament and re-predict remaining fixtures.

Two modes:
  1. Manual entry  — record a single result from CLI or from the Streamlit UI
  2. Auto-refresh  — download latest martj42 results and rebuild everything

The manual path writes to data/interim/wc2026_live_results.parquet, which is
merged into the feature store during build_features. This avoids re-downloading
all ~49k rows from GitHub just to add one result.

Usage:
    # Record a result manually
    uv run python src/etl/record_result.py record --home Mexico --away "South Africa" --score 2-0 --date 2026-06-11

    # Auto-download all new results from martj42 and rebuild
    uv run python src/etl/record_result.py refresh

    # List all manually recorded results
    uv run python src/etl/record_result.py list
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from etl.paths import INTERIM_DIR, PROCESSED_DIR
from etl.team_names import standardize_name

LIVE_RESULTS_PATH = INTERIM_DIR / "wc2026_live_results.parquet"
FEATURE_STORE = PROCESSED_DIR / "matches_features.parquet"

_EMPTY_SCHEMA = {
    "match_id": pd.Series(dtype="int64"),
    "date": pd.Series(dtype="datetime64[ns]"),
    "home_team": pd.Series(dtype="str"),
    "away_team": pd.Series(dtype="str"),
    "home_score": pd.Series(dtype="float64"),
    "away_score": pd.Series(dtype="float64"),
    "tournament": pd.Series(dtype="str"),
    "recorded_at": pd.Series(dtype="datetime64[ns]"),
}


def _load_live() -> pd.DataFrame:
    if LIVE_RESULTS_PATH.exists():
        return pd.read_parquet(LIVE_RESULTS_PATH)
    return pd.DataFrame(_EMPTY_SCHEMA)


def _save_live(df: pd.DataFrame) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LIVE_RESULTS_PATH, index=False)


def _result_label(home: float, away: float) -> str:
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def record_result(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    match_date: str | date,
    tournament: str = "FIFA World Cup",
    *,
    rebuild: bool = True,
    predict: bool = True,
) -> dict:
    """Record a single match result and optionally rebuild features + predictions."""
    home = standardize_name(home_team)
    away = standardize_name(away_team)
    match_date = pd.Timestamp(match_date)

    live = _load_live()

    # Check for duplicate
    dupe = live[
        (live["home_team"] == home) &
        (live["away_team"] == away) &
        (live["date"] == match_date)
    ]
    if not dupe.empty:
        print(f"[record_result] Result already recorded: {home} vs {away} on {match_date.date()}")
        return {"status": "duplicate", "home": home, "away": away}

    # Find the match_id from the feature store to update it
    match_id = None
    if FEATURE_STORE.exists():
        store = pd.read_parquet(FEATURE_STORE, columns=["match_id", "date", "home_team", "away_team", "played"])
        candidates = store[
            (store["home_team"] == home) &
            (store["away_team"] == away) &
            (store["date"] == match_date)
        ]
        if not candidates.empty:
            match_id = int(candidates.iloc[0]["match_id"])

    new_row = pd.DataFrame([{
        "match_id": match_id,
        "date": match_date,
        "home_team": home,
        "away_team": away,
        "home_score": float(home_score),
        "away_score": float(away_score),
        "tournament": tournament,
        "recorded_at": pd.Timestamp.now(),
    }])

    updated = pd.concat([live, new_row], ignore_index=True)
    _save_live(updated)

    result = _result_label(home_score, away_score)
    print(f"[record_result] Recorded: {home} {home_score}-{away_score} {away}  ({result})  [{match_date.date()}]")

    if rebuild:
        _rebuild_and_predict(predict=predict)

    return {
        "status": "ok",
        "home": home,
        "away": away,
        "home_score": home_score,
        "away_score": away_score,
        "result": result,
        "match_id": match_id,
    }


def _rebuild_and_predict(predict: bool = True) -> None:
    """Apply live results to feature store and re-generate predictions."""
    from features.build_features import build as build_features

    print("[record_result] Applying live results to feature store...")
    _apply_live_results_to_store()

    print("[record_result] Rebuilding features...")
    build_features()

    if predict:
        from models.predict_all import predict_everything
        print("[record_result] Re-generating predictions...")
        predict_everything()


def _apply_live_results_to_store() -> None:
    """Patch the canonical matches.parquet with live results so build_features sees them."""
    live = _load_live()
    if live.empty:
        return

    matches_path = INTERIM_DIR / "matches.parquet"
    if not matches_path.exists():
        print("[record_result] Warning: matches.parquet not found, skipping patch")
        return

    matches = pd.read_parquet(matches_path)

    updated = 0
    for _, row in live.iterrows():
        mask = (
            (matches["home_team"] == row["home_team"]) &
            (matches["away_team"] == row["away_team"]) &
            (matches["date"] == row["date"])
        )
        if mask.sum() == 0:
            print(f"[record_result] Warning: match not found in table: {row['home_team']} vs {row['away_team']} {row['date'].date()}")
            continue
        matches.loc[mask, "home_score"] = row["home_score"]
        matches.loc[mask, "away_score"] = row["away_score"]
        matches.loc[mask, "played"] = True
        matches.loc[mask, "result"] = _result_label(row["home_score"], row["away_score"])
        updated += 1

    matches.to_parquet(matches_path, index=False)
    print(f"[record_result] Patched {updated} match(es) in matches.parquet")


def list_results() -> pd.DataFrame:
    live = _load_live()
    if live.empty:
        print("No live results recorded yet.")
        return live
    show = live[["date", "home_team", "home_score", "away_score", "away_team", "recorded_at"]].copy()
    show["score"] = show["home_score"].astype(int).astype(str) + "-" + show["away_score"].astype(int).astype(str)
    show["result"] = show.apply(lambda r: _result_label(r["home_score"], r["away_score"]), axis=1)
    print(show[["date", "home_team", "score", "away_team", "result", "recorded_at"]].to_string(index=False))
    return live


def clear_results(*, rebuild: bool = True) -> None:
    """Remove all manually recorded results (use with caution)."""
    if LIVE_RESULTS_PATH.exists():
        LIVE_RESULTS_PATH.unlink()
        print("[record_result] Cleared all live results.")
    if rebuild:
        _rebuild_and_predict(predict=True)


def _parse_score(score_str: str) -> tuple[int, int]:
    """Parse '2-0', '2:0', or '2x0' into (home, away)."""
    for sep in ("-", ":", "x"):
        if sep in score_str:
            parts = score_str.split(sep)
            return int(parts[0].strip()), int(parts[1].strip())
    raise ValueError(f"Cannot parse score '{score_str}'. Use format '2-0', '2:0', or '2x0'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record WC 2026 match results and re-predict")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # record subcommand
    rec = sub.add_parser("record", help="Record a match result manually")
    rec.add_argument("--home", required=True, help="Home team name")
    rec.add_argument("--away", required=True, help="Away team name")
    rec.add_argument("--score", required=True, help="Score in format '2-0'")
    rec.add_argument("--date", required=True, help="Match date YYYY-MM-DD")
    rec.add_argument("--tournament", default="FIFA World Cup")
    rec.add_argument("--no-predict", action="store_true", help="Skip re-prediction after recording")
    rec.add_argument("--no-rebuild", action="store_true", help="Skip feature rebuild (fast, for testing)")

    # refresh subcommand
    ref = sub.add_parser("refresh", help="Download latest results from martj42 and rebuild")
    ref.add_argument("--skip-scrape", action="store_true", default=True)

    # list subcommand
    sub.add_parser("list", help="Show all manually recorded results")

    # clear subcommand
    clr = sub.add_parser("clear", help="Remove all manually recorded live results")
    clr.add_argument("--no-rebuild", action="store_true")

    args = parser.parse_args()

    if args.cmd == "record":
        home_score, away_score = _parse_score(args.score)
        record_result(
            home_team=args.home,
            away_team=args.away,
            home_score=home_score,
            away_score=away_score,
            match_date=args.date,
            tournament=args.tournament,
            rebuild=not args.no_rebuild,
            predict=not args.no_predict,
        )

    elif args.cmd == "refresh":
        from etl.refresh_tournament import main as refresh_main
        refresh_main(skip_download=False, skip_scrape=True, force=False)

    elif args.cmd == "list":
        list_results()

    elif args.cmd == "clear":
        clear_results(rebuild=not args.no_rebuild)


if __name__ == "__main__":
    main()
