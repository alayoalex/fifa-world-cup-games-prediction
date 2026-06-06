"""
User-defined hypothetical international matches.

Custom fixtures are stored locally in ``data/interim/custom_fixtures.parquet``,
merged into the canonical match table, and scored by the same feature pipeline as
real fixtures. Nothing is sent to external services.
"""
from __future__ import annotations

import difflib
import sys
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etl.paths import INTL_RESULTS_DIR, INTERIM_DIR  # noqa: E402
from etl.team_names import standardize_name  # noqa: E402

CUSTOM_TOURNAMENT = "Custom hypothetical"
STORE = INTERIM_DIR / "custom_fixtures.parquet"

_STORE_COLUMNS = [
    "fixture_id", "date", "home_team", "away_team",
    "neutral", "city", "country", "created_at",
]


def _empty_store() -> pd.DataFrame:
    return pd.DataFrame(columns=_STORE_COLUMNS)


def _load_store() -> pd.DataFrame:
    if not STORE.exists():
        return _empty_store()
    df = pd.read_parquet(STORE)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def _save_store(df: pd.DataFrame) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True).to_parquet(
        STORE, index=False
    )


def known_teams() -> set[str]:
    """Canonical team names seen in the martj42 backbone."""
    path = INTL_RESULTS_DIR / "results.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["home_team", "away_team"])
    home = df["home_team"].map(standardize_name)
    away = df["away_team"].map(standardize_name)
    return set(home.dropna()) | set(away.dropna())


def _validate_team(name: str, *, label: str) -> str:
    canonical = standardize_name(name)
    if not isinstance(canonical, str) or not canonical:
        raise ValueError(f"{label} team name is empty.")

    vocab = known_teams()
    if canonical in vocab:
        return canonical

    hints = difflib.get_close_matches(canonical, sorted(vocab), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(hints)}?" if hints else ""
    raise ValueError(f"Unknown {label} team '{name}' (canonical: '{canonical}').{hint}")


def default_fixture_date() -> pd.Timestamp:
    """Date for custom fixtures: tomorrow after the latest played match in the corpus."""
    path = INTL_RESULTS_DIR / "results.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        played = df.dropna(subset=["home_score", "away_score"])
        if not played.empty:
            return played["date"].max().normalize() + pd.Timedelta(days=1)
    return pd.Timestamp(date.today())


def add(
    home_team: str,
    away_team: str,
    *,
    match_date: pd.Timestamp | None = None,
    neutral: bool = True,
    city: str = "",
    country: str = "",
    force: bool = False,
    skip_if_exists: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Append a hypothetical fixture to the local store.

    Returns ``(store, created)``. When ``created`` is False the fixture was already
    present and left unchanged (``skip_if_exists``) or replaced (``force``).
    """
    home = _validate_team(home_team, label="home")
    away = _validate_team(away_team, label="away")
    if home == away:
        raise ValueError("Home and away teams must be different.")

    when = (match_date or default_fixture_date()).normalize()
    row = {
        "fixture_id": uuid.uuid4().hex[:12],
        "date": when,
        "home_team": home,
        "away_team": away,
        "neutral": neutral,
        "city": city,
        "country": country,
        "created_at": pd.Timestamp.utcnow(),
    }

    store = _load_store()
    if not store.empty:
        dup = store[
            (store["home_team"] == home)
            & (store["away_team"] == away)
            & (store["date"].dt.normalize() == when)
        ]
    else:
        dup = store
    if not dup.empty:
        fixture_id = dup.iloc[0]["fixture_id"]
        if force:
            store = store.loc[~dup.index].reset_index(drop=True)
            store = pd.concat([store, pd.DataFrame([row])], ignore_index=True)
            _save_store(store)
            print(
                f"[custom_fixtures] replaced {home} vs {away} on {when.date()} "
                f"(was {fixture_id})"
            )
            return store, True
        if skip_if_exists:
            print(
                f"[custom_fixtures] already exists: {home} vs {away} on {when.date()} "
                f"(id={fixture_id}) — skipping add"
            )
            return store, False
        raise ValueError(
            f"Fixture already exists: {home} vs {away} on {when.date()} (id={fixture_id}).\n"
            "Options:\n"
            f"  - re-score only:  add_fixture.py predict\n"
            f"  - replace it:     add ... --force\n"
            f"  - remove it:        add_fixture.py remove --id {fixture_id} --rebuild"
        )

    store = pd.concat([store, pd.DataFrame([row])], ignore_index=True)
    _save_store(store)
    print(f"[custom_fixtures] added {home} vs {away} on {when.date()} (neutral={neutral})")
    return store, True


def list_fixtures() -> pd.DataFrame:
    """Return all stored custom fixtures."""
    return _load_store()


def remove(fixture_id: str) -> pd.DataFrame:
    """Delete one custom fixture by ``fixture_id``."""
    store = _load_store()
    if store.empty:
        raise ValueError("No custom fixtures stored.")
    mask = store["fixture_id"] == fixture_id
    if not mask.any():
        raise ValueError(f"Unknown fixture_id: {fixture_id}")
    removed = store.loc[mask].iloc[0]
    store = store.loc[~mask].reset_index(drop=True)
    _save_store(store)
    print(
        f"[custom_fixtures] removed {removed['home_team']} vs {removed['away_team']} "
        f"({fixture_id})"
    )
    return store


def clear() -> None:
    """Delete every custom fixture."""
    if STORE.exists():
        STORE.unlink()
    print("[custom_fixtures] cleared")


def to_match_rows() -> pd.DataFrame:
    """Convert stored custom fixtures into canonical match-table rows."""
    store = _load_store()
    if store.empty:
        return pd.DataFrame()

    rows = []
    for fx in store.itertuples(index=False):
        rows.append({
            "date": fx.date,
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "home_score": pd.NA,
            "away_score": pd.NA,
            "result": pd.NA,
            "played": False,
            "neutral": bool(fx.neutral),
            "tournament": CUSTOM_TOURNAMENT,
            "competition_type": "friendly",
            "is_world_cup": False,
            "city": fx.city or "",
            "country": fx.country or "",
        })
    return pd.DataFrame(rows)
