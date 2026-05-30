"""
Source: martj42/international_results — the backbone match corpus.

Every international men's match from 1872 to the present (friendlies, qualifiers,
continental cups, World Cups, ...). This is what makes recent-form, head-to-head and
Elo features meaningful — the local data only covers World Cup matches.

Repo: https://github.com/martj42/international_results
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from etl.paths import INTL_RESULTS_DIR  # noqa: E402

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ("results.csv", "shootouts.csv", "goalscorers.csv")


def download(force: bool = False) -> dict[str, Path]:
    """Download the martj42 CSVs into ``data/raw/international-results/``.

    Returns a mapping of filename -> local path. Skips files already present unless
    ``force`` is set.
    """
    out: dict[str, Path] = {}
    for name in FILES:
        dest = INTL_RESULTS_DIR / name
        if dest.exists() and not force:
            print(f"[international_results] cached  {dest.relative_to(dest.parents[3])}")
        else:
            url = f"{BASE_URL}/{name}"
            print(f"[international_results] fetching {url}")
            df = pd.read_csv(url)
            df.to_csv(dest, index=False)
            print(f"[international_results] saved   {dest} ({len(df):,} rows)")
        out[name] = dest
    return out


def load_results() -> pd.DataFrame:
    """Load the downloaded results table with parsed dates."""
    df = pd.read_csv(INTL_RESULTS_DIR / "results.csv", parse_dates=["date"])
    return df


if __name__ == "__main__":
    paths = download(force="--force" in sys.argv)
    res = load_results()
    print(
        f"\nresults.csv: {len(res):,} matches, "
        f"{res['date'].min().date()} -> {res['date'].max().date()}"
    )
    print(f"tournaments: {res['tournament'].nunique()} distinct")
