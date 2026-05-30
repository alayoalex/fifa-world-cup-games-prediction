"""
Source: FIFA men's world ranking.

Cleans the local ``fifa_mens_rank.csv`` (1992-2024, year + semester granularity) into a
tidy, name-standardized table with an ``effective_date`` so it can be as-of joined to
matches. Includes a best-effort ``refresh()`` for post-2024 rankings that degrades
gracefully (FIFA's official endpoint is bot-protected and not relied upon).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from etl.paths import FIFA_RANK_CSV, INTERIM_DIR  # noqa: E402
from etl.team_names import standardize_series  # noqa: E402

OUT = INTERIM_DIR / "fifa_ranking.parquet"


def _effective_date(year: int, semester: int) -> pd.Timestamp:
    """Approximate date a (year, semester) ranking comes into effect.

    Semester 1 -> Jan 1, semester 2 -> Jul 1. Used for as-of joins: a match uses the most
    recent ranking whose effective_date is on or before the match date.
    """
    month = 1 if semester == 1 else 7
    return pd.Timestamp(year=year, month=month, day=1)


def clean() -> pd.DataFrame:
    """Clean the local FIFA ranking CSV -> tidy parquet in data/interim."""
    df = pd.read_csv(FIFA_RANK_CSV)
    df = df.rename(columns={
        "date": "year",
        "total.points": "points",
        "previous.points": "previous_points",
        "diff.points": "diff_points",
    })
    df["team"] = standardize_series(df["team"])
    df["effective_date"] = [
        _effective_date(y, s) for y, s in zip(df["year"], df["semester"])
    ]
    df = df[[
        "team", "year", "semester", "effective_date",
        "rank", "points", "previous_points", "diff_points",
    ]].sort_values(["effective_date", "rank"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"[fifa_ranking] {len(df):,} rows, "
          f"{df['effective_date'].min().date()} -> {df['effective_date'].max().date()} "
          f"-> {OUT.name}")
    return df


def refresh() -> None:
    """Best-effort fetch of post-2024 rankings. Degrades gracefully if unavailable.

    FIFA's ranking endpoint is protected (returns an HTML challenge, not JSON), so we do
    not depend on it. Wire a reliable source here later; the local data through 2024-S2 is
    sufficient for model training.
    """
    print("[fifa_ranking] refresh: no reliable free post-2024 source wired; "
          "using local data through 2024-S2 (sufficient for training).")


if __name__ == "__main__":
    clean()
