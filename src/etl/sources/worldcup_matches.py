"""
Source: jfjelstul/worldcup — World Cup match metadata.

The martj42 backbone tags WC matches but lacks stage/group/knockout detail. This cleans
jfjelstul's men's WC matches (name-standardized) into an interim table that can enrich the
canonical match table with tournament-stage features and serve as a validation cross-check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from etl.paths import INTERIM_DIR, WORLDCUP_CSV_DIR  # noqa: E402
from etl.team_names import standardize_series  # noqa: E402

OUT = INTERIM_DIR / "worldcup_matches.parquet"
MATCHES_URL = (
    "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv"
)


def download(force: bool = False) -> Path:
    """Download jfjelstul World Cup matches.csv into ``data/raw/worldcup/data-csv/``."""
    dest = WORLDCUP_CSV_DIR / "matches.csv"
    if dest.exists() and not force:
        print(f"[worldcup_matches] cached  {dest}")
        return dest

    print(f"[worldcup_matches] fetching {MATCHES_URL}")
    df = pd.read_csv(MATCHES_URL)
    WORLDCUP_CSV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"[worldcup_matches] saved   {dest} ({len(df):,} rows)")
    return dest


def clean() -> pd.DataFrame:
    """Clean jfjelstul men's World Cup matches -> tidy parquet."""
    if not (WORLDCUP_CSV_DIR / "matches.csv").exists():
        download()
    df = pd.read_csv(WORLDCUP_CSV_DIR / "matches.csv", parse_dates=["match_date"])
    # Men's tournaments only (women's tournament_ids share the WC-YYYY scheme but for
    # odd years 1991/1995/... — keep the men's editions, which fall on the canonical years).
    mens_years = {1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978,
                    1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022}
    df["year"] = df["match_date"].dt.year
    df = df[df["year"].isin(mens_years)].copy()

    out = pd.DataFrame({
        "date": df["match_date"],
        "year": df["year"],
        "home_team": standardize_series(df["home_team_name"]),
        "away_team": standardize_series(df["away_team_name"]),
        "home_score": df["home_team_score"],
        "away_score": df["away_team_score"],
        "stage": df["stage_name"],
        "group_name": df["group_name"],
        "knockout_stage": df["knockout_stage"].astype("boolean"),
        "extra_time": df["extra_time"].astype("boolean"),
        "penalty_shootout": df["penalty_shootout"].astype("boolean"),
    }).sort_values("date").reset_index(drop=True)

    out.to_parquet(OUT, index=False)
    print(f"[worldcup_matches] {len(out):,} men's WC matches "
            f"({out['year'].min()}-{out['year'].max()}) -> {OUT.name}")
    return out


if __name__ == "__main__":
    clean()
