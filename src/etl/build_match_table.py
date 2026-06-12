"""
Build the canonical long-history match table from the martj42 backbone.

One row per international match (1872 -> scheduled 2026 fixtures), with standardized team
names, a home-perspective 3-way result, a neutral-venue flag, and competition typing. This
is the single table every feature is computed from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etl import custom_fixtures  # noqa: E402
from etl.paths import INTERIM_DIR, INTL_RESULTS_DIR  # noqa: E402
from etl.team_names import standardize_series  # noqa: E402

OUT = INTERIM_DIR / "matches.parquet"

# Major continental championships (finals tournaments) — used for competition typing.
_CONTINENTAL = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "Gold Cup", "CONCACAF Championship", "Oceania Nations Cup",
}


def _competition_type(t: str) -> str:
    if t == "Friendly":
        return "friendly"
    if t == "FIFA World Cup":
        return "world_cup"
    if t == "FIFA World Cup qualification":
        return "world_cup_qual"
    if t in _CONTINENTAL:
        return "continental"
    if "qualification" in t:
        return "continental_qual"
    return "other"


def build() -> pd.DataFrame:
    df = pd.read_csv(INTL_RESULTS_DIR / "results.csv", parse_dates=["date"])
    df["home_team"] = standardize_series(df["home_team"])
    df["away_team"] = standardize_series(df["away_team"])

    df["played"] = df["home_score"].notna() & df["away_score"].notna()

    # Home-perspective 3-way result for played matches (based on 90-min/full-time score;
    # shootout outcomes are deliberately left as draws — the match itself was a draw).
    cond_h = df["home_score"] > df["away_score"]
    cond_a = df["home_score"] < df["away_score"]
    df["result"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df["played"] & cond_h, "result"] = "H"
    df.loc[df["played"] & cond_a, "result"] = "A"
    df.loc[df["played"] & ~cond_h & ~cond_a, "result"] = "D"

    df["competition_type"] = df["tournament"].map(_competition_type)
    df["is_world_cup"] = df["tournament"] == "FIFA World Cup"
    df["neutral"] = df["neutral"].astype("boolean")

    cols = [
        "date", "home_team", "away_team",
        "home_score", "away_score", "result", "played",
        "neutral", "tournament", "competition_type", "is_world_cup",
        "city", "country",
    ]
    out = df[cols]

    custom = custom_fixtures.to_match_rows()
    if not custom.empty:
        out = pd.concat([out, custom], ignore_index=True)

    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    out.insert(0, "match_id", out.index + 1)

    out.to_parquet(OUT, index=False)

    played = out["played"].sum()
    n_custom = int((out["tournament"] == custom_fixtures.CUSTOM_TOURNAMENT).sum())
    print(f"[matches] {len(out):,} matches "
          f"({out['date'].min().date()} -> {out['date'].max().date()}), "
          f"{played:,} played, {len(out) - played} future fixtures "
          f"({n_custom} custom) -> {OUT.name}")
    print("  competition_type:", out["competition_type"].value_counts().to_dict())
    return out


if __name__ == "__main__":
    build()
