"""
Leakage-safe feature engineering for the match-prediction model.

For every match we compute each team's *pre-match* state using ONLY matches strictly
before that match's date. A single chronological pass maintains the running state (Elo,
recent form, head-to-head, rest days); FIFA rank is an as-of join and market value is a
current-snapshot join.

Output (data/processed/):
  * matches_features.parquet  — model-ready feature store (played matches + 2026 fixtures)
  * elo_history.parquet       — per-team pre-match Elo timeline
  * data_dictionary.md        — column documentation
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etl.paths import INTERIM_DIR, PROCESSED_DIR  # noqa: E402
from etl.team_names import build_confederation_map  # noqa: E402

# --- Elo configuration (World Football Elo, eloratings.net conventions) ---------------
ELO_INIT = 1500.0
HOME_ADVANTAGE = 100.0
K_BY_TYPE = {
    "world_cup": 60.0,
    "continental": 50.0,
    "world_cup_qual": 40.0,
    "continental_qual": 40.0,
    "other": 30.0,
    "friendly": 20.0,
}

# --- Recent-form configuration --------------------------------------------------------
FORM_WINDOW = 10     # matches
FORM_DECAY = 0.85    # exponential weight per step back in time


def _goal_diff_multiplier(gd: int) -> float:
    """World Football Elo goal-difference weighting."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _weighted_form(history: deque) -> tuple[float, float, float]:
    """Exp-decayed (goals_for, goals_against, points) over a team's recent matches.

    ``history`` holds (gf, ga, pts) with the most recent match last. Returns NaNs when the
    team has no prior matches.
    """
    if not history:
        return (np.nan, np.nan, np.nan)
    recs = list(history)                       # oldest -> newest
    n = len(recs)
    # weight: most recent (last) gets decay**0 = 1
    weights = np.array([FORM_DECAY ** (n - 1 - i) for i in range(n)])
    wsum = weights.sum()
    gf = np.dot(weights, [r[0] for r in recs]) / wsum
    ga = np.dot(weights, [r[1] for r in recs]) / wsum
    pts = np.dot(weights, [r[2] for r in recs]) / wsum
    return (gf, ga, pts)


def _fifa_rank_asof(matches: pd.DataFrame) -> pd.DataFrame:
    """As-of join: each team's most recent FIFA rank effective on/before the match date."""
    ranks = pd.read_parquet(INTERIM_DIR / "fifa_ranking.parquet")
    ranks = ranks[["team", "effective_date", "rank", "points"]].sort_values(
        "effective_date"
    )

    def _join(side: str) -> pd.DataFrame:
        left = (matches[["match_id", "date", f"{side}_team"]]
                .rename(columns={f"{side}_team": "team"})
                .sort_values("date"))
        merged = pd.merge_asof(
            left, ranks, left_on="date", right_on="effective_date",
            by="team", direction="backward",
        )
        return merged.set_index("match_id")[["rank", "points"]].rename(
            columns={"rank": f"{side}_fifa_rank", "points": f"{side}_fifa_points"}
        )

    return _join("home").join(_join("away"))


def build() -> pd.DataFrame:
    matches = pd.read_parquet(INTERIM_DIR / "matches.parquet").sort_values(
        "date", kind="stable"
    ).reset_index(drop=True)

    # ---- running state -------------------------------------------------------------
    elo: dict[str, float] = defaultdict(lambda: ELO_INIT)
    last_date: dict[str, pd.Timestamp] = {}
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    h2h: dict[tuple[str, str], list] = defaultdict(list)  # key=sorted(teams)

    rows = []
    elo_timeline = []
    for m in matches.itertuples(index=False):
        h, a = m.home_team, m.away_team
        neutral = bool(m.neutral) if m.neutral is not None else False
        eh, ea = elo[h], elo[a]

        # --- pre-match features ---
        dr = eh - ea + (0.0 if neutral else HOME_ADVANTAGE)
        we_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

        gf_h, ga_h, pts_h = _weighted_form(form[h])
        gf_a, ga_a, pts_a = _weighted_form(form[a])

        key = (h, a) if h < a else (a, h)
        prior = h2h[key]
        if prior:
            # orient each prior meeting to the current home team
            hp = [(r[1] if h == r[0] else r[2],   # current-home goals
                   r[2] if h == r[0] else r[1])    # current-away goals
                  for r in prior]
            h2h_n = len(hp)
            h2h_home_winrate = np.mean([1.0 if x > y else 0.5 if x == y else 0.0
                                        for x, y in hp])
            h2h_goaldiff = np.mean([x - y for x, y in hp])
        else:
            h2h_n, h2h_home_winrate, h2h_goaldiff = 0, np.nan, np.nan

        rest_h = (m.date - last_date[h]).days if h in last_date else np.nan
        rest_a = (m.date - last_date[a]).days if a in last_date else np.nan

        rows.append({
            "match_id": m.match_id,
            "elo_home_pre": eh,
            "elo_away_pre": ea,
            "elo_diff": eh - ea,
            "elo_exp_home": we_home,
            "form_gf_home": gf_h,
            "form_ga_home": ga_h,
            "form_pts_home": pts_h,
            "form_gf_away": gf_a,
            "form_ga_away": ga_a,
            "form_pts_away": pts_a,
            "h2h_matches": h2h_n,
            "h2h_home_winrate": h2h_home_winrate,
            "h2h_goaldiff_home": h2h_goaldiff,
            "rest_days_home": rest_h,
            "rest_days_away": rest_a,
            "rest_days_diff": (rest_h - rest_a) if (h in last_date and a in last_date) else np.nan,
        })
        elo_timeline.append({"match_id": m.match_id, "date": m.date, "team": h, "elo_pre": eh})
        elo_timeline.append({"match_id": m.match_id, "date": m.date, "team": a, "elo_pre": ea})

        # --- post-match state update (only for played matches) ---
        if not m.played:
            continue
        gd = int(m.home_score - m.away_score)
        w_home = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        k = K_BY_TYPE.get(m.competition_type, 30.0)
        delta = k * _goal_diff_multiplier(gd) * (w_home - we_home)
        elo[h] = eh + delta
        elo[a] = ea - delta

        form[h].append((m.home_score, m.away_score, w_home * 1.0 if gd != 0 else 0.5))
        form[a].append((m.away_score, m.home_score, 1.0 - w_home if gd != 0 else 0.5))
        # store points as 3/1/0? keep simple win-units (1/0.5/0) for form_pts
        last_date[h] = m.date
        last_date[a] = m.date
        h2h[key].append((h, m.home_score, m.away_score))

    feat = pd.DataFrame(rows)
    out = matches.merge(feat, on="match_id", how="left")

    # ---- FIFA rank (as-of) ----------------------------------------------------------
    rk = _fifa_rank_asof(matches)
    out = out.merge(rk, left_on="match_id", right_index=True, how="left")
    out["fifa_rank_diff"] = out["away_fifa_rank"] - out["home_fifa_rank"]  # +ve: home better

    # ---- confederation --------------------------------------------------------------
    conf = build_confederation_map(matches).set_index("team")["confederation"]
    out["conf_home"] = out["home_team"].map(conf)
    out["conf_away"] = out["away_team"].map(conf)
    out["same_confederation"] = (out["conf_home"] == out["conf_away"])

    # ---- market value (current snapshot) --------------------------------------------
    mv = pd.read_parquet(INTERIM_DIR / "market_value.parquet").set_index("team")[
        "market_value_eur"
    ]
    out["mv_home_eur"] = out["home_team"].map(mv)
    out["mv_away_eur"] = out["away_team"].map(mv)
    out["mv_log_ratio"] = np.log(out["mv_home_eur"] / out["mv_away_eur"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_DIR / "matches_features.parquet", index=False)
    pd.DataFrame(elo_timeline).to_parquet(PROCESSED_DIR / "elo_history.parquet", index=False)
    _write_data_dictionary(out)

    played = out["played"].sum()
    print(f"[features] {len(out):,} rows x {out.shape[1]} cols "
          f"({played:,} played, {len(out)-played} fixtures) "
          f"-> matches_features.parquet")
    print(f"[features] market value present on {out['mv_home_eur'].notna().sum():,} home sides; "
          f"FIFA rank present on {out['home_fifa_rank'].notna().sum():,}")
    return out


def _write_data_dictionary(df: pd.DataFrame) -> None:
    descriptions = {
        "match_id": "Chronological match identifier",
        "date": "Match date",
        "home_team": "Home (or first-listed, for neutral) team — canonical name",
        "away_team": "Away (or second-listed) team — canonical name",
        "home_score": "Home goals (full time); null for unplayed fixtures",
        "away_score": "Away goals (full time); null for unplayed fixtures",
        "result": "TARGET (3-way): H=home win, D=draw, A=away win",
        "played": "Whether the match has been played",
        "neutral": "Whether the match is at a neutral venue (all WC matches are)",
        "tournament": "Raw competition name (martj42)",
        "competition_type": "friendly/world_cup/world_cup_qual/continental/continental_qual/other",
        "is_world_cup": "FIFA World Cup finals match",
        "city": "Host city", "country": "Host country",
        "elo_home_pre": "Home team Elo before the match (World Football Elo)",
        "elo_away_pre": "Away team Elo before the match",
        "elo_diff": "elo_home_pre - elo_away_pre",
        "elo_exp_home": "Elo-expected score for home (incl. home advantage unless neutral)",
        "form_gf_home": "Home exp-decayed avg goals scored, last %d matches" % FORM_WINDOW,
        "form_ga_home": "Home exp-decayed avg goals conceded, last %d" % FORM_WINDOW,
        "form_pts_home": "Home exp-decayed avg result units (W=1/D=0.5/L=0), last %d" % FORM_WINDOW,
        "form_gf_away": "Away exp-decayed avg goals scored, last %d" % FORM_WINDOW,
        "form_ga_away": "Away exp-decayed avg goals conceded, last %d" % FORM_WINDOW,
        "form_pts_away": "Away exp-decayed avg result units, last %d" % FORM_WINDOW,
        "h2h_matches": "Number of prior meetings between the two teams",
        "h2h_home_winrate": "Current-home win-rate in prior meetings (W=1/D=0.5/L=0)",
        "h2h_goaldiff_home": "Avg goal difference (current-home perspective) in prior meetings",
        "rest_days_home": "Days since home team's previous match",
        "rest_days_away": "Days since away team's previous match",
        "rest_days_diff": "rest_days_home - rest_days_away",
        "home_fifa_rank": "Home FIFA rank, most recent effective on/before match date",
        "away_fifa_rank": "Away FIFA rank (as-of)",
        "home_fifa_points": "Home FIFA points (as-of)",
        "away_fifa_points": "Away FIFA points (as-of)",
        "fifa_rank_diff": "away_fifa_rank - home_fifa_rank (positive => home better ranked)",
        "conf_home": "Home team confederation", "conf_away": "Away team confederation",
        "same_confederation": "Both teams in the same confederation",
        "mv_home_eur": "Home squad total market value (current Transfermarkt snapshot)",
        "mv_away_eur": "Away squad total market value (current snapshot)",
        "mv_log_ratio": "log(mv_home / mv_away)",
    }
    lines = ["# Data Dictionary — `matches_features.parquet`", "",
             f"{len(df):,} rows, {df.shape[1]} columns. "
             "All feature columns are computed strictly from data **before** each match "
             "date (no temporal leakage).", "",
             "> **Caveat — market value:** `mv_*` is a *current* Transfermarkt snapshot "
             "applied to every row, so it is anachronistic for historical matches (acts as "
             "a static team-strength prior, not a point-in-time value). It is most "
             "meaningful for recent matches and the 2026 fixtures; consider nulling it for "
             "old matches if that matters to your model.", "",
             "| Column | Dtype | Description |", "|---|---|---|"]
    for col in df.columns:
        lines.append(f"| `{col}` | {df[col].dtype} | {descriptions.get(col, '')} |")
    (PROCESSED_DIR / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
