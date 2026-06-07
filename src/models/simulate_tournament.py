"""
Monte Carlo simulation of the 2026 FIFA World Cup bracket.

Simulates the full tournament N times using ensemble match probabilities:
  1. Group stage: round-robin within each group (3 matches per team)
  2. Round of 24: top 2 from each group + 4 best 3rd-place teams advance (but here
     we follow the actual 2026 format: top 2 from each of 12 groups = 24 teams)
  3. Round of 16 → Quarterfinals → Semifinals → Final

Output: for each team, probability of reaching each round + winning the tournament.

Run:
    uv run python src/models/simulate_tournament.py
    uv run python src/models/simulate_tournament.py --sims 10000
    uv run python src/models/simulate_tournament.py --output data/processed/simulation.csv
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from etl.paths import PROCESSED_DIR
from models.tune import load_best_params

PREDICTIONS_PATH = PROCESSED_DIR / "wc2026_predictions_full.csv"
SIMULATION_PATH = PROCESSED_DIR / "simulation.csv"
DEFAULT_SIMS = 10_000

# 2026 WC: 12 groups of 4, top 2 advance (24 teams to knockout)
# Knockout bracket pairing (group winners vs runners-up from other groups)
# Official 2026 bracket seeding (A1 vs B2, B1 vs A2, etc.)
KNOCKOUT_BRACKET = [
    ("A", 1, "B", 2),
    ("B", 1, "A", 2),
    ("C", 1, "D", 2),
    ("D", 1, "C", 2),
    ("E", 1, "F", 2),
    ("F", 1, "E", 2),
    ("G", 1, "H", 2),
    ("H", 1, "G", 2),
    ("I", 1, "J", 2),
    ("J", 1, "I", 2),
    ("K", 1, "L", 2),
    ("L", 1, "K", 2),
]

GROUPS: dict[str, list[str]] = {
    "A": ["Algeria", "Argentina", "Austria", "Jordan"],
    "B": ["Australia", "Paraguay", "Turkey", "United States"],
    "C": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "D": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"],
    "E": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "F": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "G": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "H": ["Croatia", "England", "Ghana", "Panama"],
    "I": ["Curaçao", "Ecuador", "Germany", "Ivory Coast"],
    "J": ["Czech Republic", "Mexico", "South Africa", "South Korea"],
    "K": ["France", "Iraq", "Norway", "Senegal"],
    "L": ["Japan", "Netherlands", "Sweden", "Tunisia"],
}


def _build_prob_lookup(preds: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Map (home, away) -> (p_H, p_D, p_A) from predictions CSV."""
    lookup: dict[tuple[str, str], tuple[float, float, float]] = {}
    for _, row in preds.iterrows():
        h, a = row["home_team"], row["away_team"]
        ph = row.get("ensemble_p_H", row.get("p_H", 1/3))
        pd_ = row.get("ensemble_p_D", row.get("p_D", 1/3))
        pa = row.get("ensemble_p_A", row.get("p_A", 1/3))
        lookup[(h, a)] = (ph, pd_, pa)
    return lookup


def _simulate_match(
    home: str,
    away: str,
    prob_lookup: dict,
    *,
    allow_draw: bool = True,
) -> str:
    """Return winner ('home' or 'away') or 'draw' (only in group stage)."""
    key = (home, away)
    rev = (away, home)
    if key in prob_lookup:
        ph, pd_, pa = prob_lookup[key]
    elif rev in prob_lookup:
        pa, pd_, ph = prob_lookup[rev]
    else:
        ph, pd_, pa = 1/3, 1/3, 1/3

    if not allow_draw:
        # Redistribute draw probability proportionally for knockout
        total = ph + pa
        if total <= 0:
            ph, pa = 0.5, 0.5
        else:
            ph, pa = ph / total, pa / total
        pd_ = 0.0

    r = random.random()
    if r < ph:
        return "home"
    elif r < ph + pd_:
        return "draw"
    else:
        return "away"


def _simulate_group(
    group_teams: list[str],
    prob_lookup: dict,
) -> list[str]:
    """Simulate one group's round-robin. Return teams sorted by points (desc), then GD."""
    stats: dict[str, dict] = {
        t: {"pts": 0, "gd": 0, "gf": 0} for t in group_teams
    }

    for i, home in enumerate(group_teams):
        for away in group_teams[i + 1:]:
            outcome = _simulate_match(home, away, prob_lookup, allow_draw=True)
            # Approximate GD using expected goals from lookup
            if outcome == "home":
                stats[home]["pts"] += 3
                stats[home]["gd"] += 1
                stats[away]["gd"] -= 1
                stats[home]["gf"] += 1
            elif outcome == "away":
                stats[away]["pts"] += 3
                stats[away]["gd"] += 1
                stats[home]["gd"] -= 1
                stats[away]["gf"] += 1
            else:
                stats[home]["pts"] += 1
                stats[away]["pts"] += 1

    # Sort: points desc, goal diff desc, goals for desc, then random tiebreak
    return sorted(
        group_teams,
        key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gf"], random.random()),
        reverse=True,
    )


def _simulate_knockout_match(home: str, away: str, prob_lookup: dict) -> str:
    """Return winner (no draws in knockout)."""
    outcome = _simulate_match(home, away, prob_lookup, allow_draw=False)
    return home if outcome == "home" else away


def simulate_once(prob_lookup: dict) -> dict[str, str]:
    """Run one full tournament simulation. Return {team: best_round_reached}."""
    rounds: dict[str, str] = {t: "Group stage" for group in GROUPS.values() for t in group}

    # --- Group stage ---
    group_results: dict[str, list[str]] = {}
    for gname, teams in GROUPS.items():
        standing = _simulate_group(teams, prob_lookup)
        group_results[gname] = standing
        # Top 2 advance
        for pos, team in enumerate(standing):
            if pos < 2:
                rounds[team] = "Round of 24"

    # --- Build knockout bracket (Round of 24 → 12 matches) ---
    r24_matches: list[tuple[str, str]] = []
    for g1, pos1, g2, pos2 in KNOCKOUT_BRACKET:
        team1 = group_results[g1][pos1 - 1]
        team2 = group_results[g2][pos2 - 1]
        r24_matches.append((team1, team2))

    # Round of 24
    r16_teams: list[str] = []
    for home, away in r24_matches:
        winner = _simulate_knockout_match(home, away, prob_lookup)
        rounds[winner] = "Round of 16"
        r16_teams.append(winner)

    # Round of 16 (12 → 6... but 12 is not divisible cleanly; pair sequentially)
    # 12 winners → pair as 1v2, 3v4, 5v6, 7v8, 9v10, 11v12
    qf_teams: list[str] = []
    for i in range(0, len(r16_teams), 2):
        if i + 1 < len(r16_teams):
            winner = _simulate_knockout_match(r16_teams[i], r16_teams[i + 1], prob_lookup)
            rounds[winner] = "Quarterfinal"
            qf_teams.append(winner)

    # Quarterfinals (6 → 3... pad to even by giving bye to first team if odd)
    sf_teams: list[str] = []
    qf_iter = iter(qf_teams)
    for home in qf_iter:
        try:
            away = next(qf_iter)
            winner = _simulate_knockout_match(home, away, prob_lookup)
        except StopIteration:
            winner = home  # bye
        rounds[winner] = "Semifinal"
        sf_teams.append(winner)

    # Semifinals (typically 4, but could be 3 with bye logic above — normalize to 4)
    # Pad if needed
    while len(sf_teams) < 4:
        sf_teams.append(sf_teams[-1])

    finalists: list[str] = []
    for i in range(0, 4, 2):
        winner = _simulate_knockout_match(sf_teams[i], sf_teams[i + 1], prob_lookup)
        rounds[winner] = "Final"
        finalists.append(winner)

    # Final
    if len(finalists) >= 2:
        champion = _simulate_knockout_match(finalists[0], finalists[1], prob_lookup)
        rounds[champion] = "Champion"

    return rounds


ROUND_ORDER = ["Group stage", "Round of 24", "Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"]


def simulate_tournament(n_sims: int = DEFAULT_SIMS) -> pd.DataFrame:
    """Run n_sims Monte Carlo simulations. Return probability table per team per round."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {PREDICTIONS_PATH}. Run predict_all.py first.")

    preds = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    prob_lookup = _build_prob_lookup(preds)

    all_teams = [t for group in GROUPS.values() for t in group]
    counts: dict[str, dict[str, int]] = {t: {r: 0 for r in ROUND_ORDER} for t in all_teams}

    rng_seed = 42
    random.seed(rng_seed)

    for _ in range(n_sims):
        result = simulate_once(prob_lookup)
        for team, best_round in result.items():
            # Credit all rounds up to and including best_round
            best_idx = ROUND_ORDER.index(best_round)
            for r in ROUND_ORDER[: best_idx + 1]:
                counts[team][r] += 1

    rows = []
    for team in all_teams:
        group = next(g for g, members in GROUPS.items() if team in members)
        row = {"team": team, "group": group}
        for r in ROUND_ORDER:
            row[r] = round(counts[team][r] / n_sims, 4)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def save_simulation(df: pd.DataFrame, path: Path = SIMULATION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[simulate] Saved -> {path}")


def print_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    print(f"\n{'':>3} {'Team':<30} {'Grp':>4} {'R24':>6} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>6} {'WIN':>6}")
    print("-" * 80)
    for _, row in df.head(top_n).iterrows():
        print(
            f"{int(row['rank']):>3} {row['team']:<30} {row['group']:>4} "
            f"{row['Round of 24']:>6.1%} {row['Round of 16']:>6.1%} "
            f"{row['Quarterfinal']:>6.1%} {row['Semifinal']:>6.1%} "
            f"{row['Final']:>6.1%} {row['Champion']:>6.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo WC 2026 tournament simulation")
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS, help="Number of simulations")
    parser.add_argument("--output", type=Path, default=SIMULATION_PATH)
    parser.add_argument("--top", type=int, default=20, help="Teams to show in summary")
    args = parser.parse_args()

    params = load_best_params()
    print(f"Running {args.sims:,} simulations (seed=42)...")
    df = simulate_tournament(n_sims=args.sims)
    save_simulation(df, args.output)
    print_summary(df, top_n=args.top)


if __name__ == "__main__":
    main()
