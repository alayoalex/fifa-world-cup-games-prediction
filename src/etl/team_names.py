"""
Team-name standardization + confederation mapping.

Every source spells countries differently. We pick the **martj42 spelling as canonical**
(it is the largest, most consistent vocabulary and is the backbone match corpus) and map
every other source's aliases onto it. One canonical vocabulary makes all downstream joins
(FIFA rank, market value, confederation) trivial and correct.

Confederation is resolved data-drivenly: each team is assigned the confederation of the
continental competition it plays in most, then overridden by jfjelstul's authoritative
team->confederation table where available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etl.paths import WORLDCUP_CSV_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Alias -> canonical (martj42) name.
# Covers fifa_mens_rank.csv and jfjelstul worldcup spellings + common variants.
# ---------------------------------------------------------------------------
ALIASES: dict[str, str] = {
    # --- FIFA ranking spellings ---
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Chinese Taipei": "Taiwan",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Curacao": "Curaçao",
    "FYR Macedonia": "North Macedonia",
    "Hong Kong, China": "Hong Kong",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Brunei Darussalam": "Brunei",
    "Aotearoa New Zealand": "New Zealand",
    "Swaziland": "Eswatini",
    "The Gambia": "Gambia",
    "US Virgin Islands": "United States Virgin Islands",
    "Sao Tome e Principe": "São Tomé and Príncipe",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "St. Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    # --- jfjelstul worldcup spellings ---
    "China": "China PR",
    "West Germany": "Germany",
    "East Germany": "German DR",
    "Soviet Union": "Russia",          # martj42 merges USSR into Russia
    "Dutch East Indies": "Indonesia",
    "Zaire": "DR Congo",
    "Serbia and Montenegro": "Serbia",  # martj42 merges into Serbia
    "Republic of Ireland": "Republic of Ireland",
    # --- generic variants ---
    "Cote d'Ivoire": "Ivory Coast",
    "North Macedonia": "North Macedonia",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
}


def standardize_name(name: object) -> object:
    """Map a single team name to its canonical (martj42) spelling."""
    if pd.isna(name):
        return name
    s = str(name).strip()
    return ALIASES.get(s, s)


def standardize_series(s: pd.Series) -> pd.Series:
    """Vectorized :func:`standardize_name` over a Series."""
    return s.map(standardize_name)


# ---------------------------------------------------------------------------
# Confederation inference
# ---------------------------------------------------------------------------
# Continental competitions whose participation unambiguously implies a confederation.
# Ambiguous comps (Arab Cup, Confederations Cup, Afro-Asian Games, ...) are intentionally
# excluded so they do not mis-assign.
TOURNAMENT_CONFEDERATION: dict[str, str] = {
    # UEFA
    "UEFA Euro": "UEFA",
    "UEFA Euro qualification": "UEFA",
    "UEFA Nations League": "UEFA",
    "Nordic Championship": "UEFA",
    "Baltic Cup": "UEFA",
    "Central European International Cup": "UEFA",
    # CAF
    "African Cup of Nations": "CAF",
    "African Cup of Nations qualification": "CAF",
    "CECAFA Cup": "CAF",
    "COSAFA Cup": "CAF",
    "COSAFA Cup qualification": "CAF",
    "All-African Games": "CAF",
    "African Friendship Games": "CAF",
    "West African Cup": "CAF",
    # AFC
    "AFC Asian Cup": "AFC",
    "AFC Asian Cup qualification": "AFC",
    "AFC Challenge Cup": "AFC",
    "AFC Challenge Cup qualification": "AFC",
    "AFC Solidarity Cup": "AFC",
    "Asian Games": "AFC",
    "Southeast Asian Games": "AFC",
    "Southeast Asian Peninsular Games": "AFC",
    "South Asian Games": "AFC",
    "SAFF Cup": "AFC",
    "Gulf Cup": "AFC",
    "WAFF Championship": "AFC",
    "CAFA Nations Cup": "AFC",
    "East Asian Games": "AFC",
    # CONCACAF
    "Gold Cup": "CONCACAF",
    "Gold Cup qualification": "CONCACAF",
    "CONCACAF Championship": "CONCACAF",
    "CONCACAF Championship qualification": "CONCACAF",
    "CONCACAF Nations League": "CONCACAF",
    "CONCACAF Nations League qualification": "CONCACAF",
    "CONCACAF Series": "CONCACAF",
    "CFU Caribbean Cup": "CONCACAF",
    "CFU Caribbean Cup qualification": "CONCACAF",
    "NAFC Championship": "CONCACAF",
    "Central American and Caribbean Games": "CONCACAF",
    # CONMEBOL
    "Copa América": "CONMEBOL",
    "Copa América qualification": "CONMEBOL",
    # OFC
    "Oceania Nations Cup": "OFC",
    "Oceania Nations Cup qualification": "OFC",
}

# The 10 CONMEBOL members rarely appear under a single tidy label across history, so pin
# them explicitly (cheap and unambiguous).
CONMEBOL = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Paraguay", "Peru", "Uruguay", "Venezuela",
}


def build_confederation_map(results: pd.DataFrame) -> pd.DataFrame:
    """Build a canonical ``team -> confederation`` table.

    Strategy: count how often each (standardized) team appears in each confederation's
    continental competitions, assign the argmax, then override with jfjelstul's
    authoritative table and the explicit CONMEBOL pin.

    Returns a DataFrame with columns ``team, confederation``.
    """
    df = results.copy()
    df["home_team"] = standardize_series(df["home_team"])
    df["away_team"] = standardize_series(df["away_team"])
    df["conf"] = df["tournament"].map(TOURNAMENT_CONFEDERATION)

    long = pd.concat([
        df[["home_team", "conf"]].rename(columns={"home_team": "team"}),
        df[["away_team", "conf"]].rename(columns={"away_team": "team"}),
    ])
    long = long.dropna(subset=["conf"])
    counts = long.groupby(["team", "conf"]).size().reset_index(name="n")
    inferred = (
        counts.sort_values("n", ascending=False)
        .drop_duplicates("team")
        .set_index("team")["conf"]
        .to_dict()
    )

    # Authoritative override from jfjelstul (88 WC teams).
    teams_csv = WORLDCUP_CSV_DIR / "teams.csv"
    if teams_csv.exists():
        jf = pd.read_csv(teams_csv)
        for _, row in jf.iterrows():
            team = standardize_name(row["team_name"])
            inferred[team] = row["confederation_code"]

    # Explicit CONMEBOL pin (highest priority).
    for team in CONMEBOL:
        inferred[team] = "CONMEBOL"

    all_teams = sorted(set(standardize_series(df["home_team"])) |
                       set(standardize_series(df["away_team"])))
    out = pd.DataFrame(
        {"team": all_teams,
         "confederation": [inferred.get(t, "Unknown") for t in all_teams]}
    )
    return out


if __name__ == "__main__":
    res = pd.read_csv(
        Path(__file__).resolve().parents[2]
        / "data" / "raw" / "international-results" / "results.csv"
    )
    conf = build_confederation_map(res)
    print(conf["confederation"].value_counts(dropna=False))
    print("\nUnknown teams:", conf[conf["confederation"] == "Unknown"]["team"].tolist()[:40])
