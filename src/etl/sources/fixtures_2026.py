"""
Source: 2026 FIFA World Cup fixtures.

The martj42 backbone already contains the scheduled 2026 WC fixtures as future-dated rows
(``tournament == "FIFA World Cup"`` with null scores). This extracts and standardizes them
into the fixture list to predict. Knockout fixtures that depend on group results are not
yet determined and will appear in martj42 as the tournament unfolds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from etl.paths import INTERIM_DIR, INTL_RESULTS_DIR  # noqa: E402
from etl.team_names import standardize_series  # noqa: E402

OUT = INTERIM_DIR / "fixtures_2026.parquet"


def clean() -> pd.DataFrame:
    """Extract unplayed 2026 World Cup fixtures from the martj42 corpus."""
    df = pd.read_csv(INTL_RESULTS_DIR / "results.csv", parse_dates=["date"])
    fx = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"].dt.year == 2026)
        & (df["home_score"].isna())
    ].copy()

    out = pd.DataFrame({
        "date": fx["date"],
        "home_team": standardize_series(fx["home_team"]),
        "away_team": standardize_series(fx["away_team"]),
        "city": fx["city"],
        "country": fx["country"],
        "neutral": fx["neutral"].astype("boolean"),
    }).sort_values("date").reset_index(drop=True)

    out.to_parquet(OUT, index=False)
    teams = sorted(set(out["home_team"]) | set(out["away_team"]))
    print(f"[fixtures_2026] {len(out)} fixtures, {len(teams)} teams "
          f"({out['date'].min().date()} -> {out['date'].max().date()}) -> {OUT.name}")
    return out


if __name__ == "__main__":
    clean()
