"""
Source: Transfermarkt national-team squad market values (current snapshot).

Scrapes the "most valuable national teams" ranking. This is a **current** snapshot only
(Transfermarkt does not expose historical squad values cheaply), so as a feature it
populates recent matches and the 2026 fixtures; older matches get NaN. Built defensively:
any network/parse failure degrades to an empty table with a clear message rather than
breaking the pipeline.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from etl.paths import INTERIM_DIR  # noqa: E402
from etl.team_names import standardize_name  # noqa: E402

OUT = INTERIM_DIR / "market_value.parquet"
BASE = ("https://www.transfermarkt.us/vereins-statistik/"
        "wertvollstenationalmannschaften/marktwertetop")
HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120 Safari/537.36")}

# Transfermarkt spellings that differ from our canonical (martj42) names.
TM_ALIASES = {
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "Cote d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Democratic Republic of the Congo": "DR Congo",
    "Republic of the Congo": "Congo",
    "USA": "United States",
    "Czech Republic": "Czech Republic",
    "Republic of Ireland": "Republic of Ireland",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cape Verde",
    "DR Congo": "DR Congo",
}


def _money_to_eur(text: str) -> float | None:
    """'€1.40bn' / '€950.00m' / '€500Th.' -> euros as float."""
    m = re.search(r"€\s*([\d.,]+)\s*(bn|m|Th\.?)?", text, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower().rstrip(".")
    return val * {"bn": 1e9, "m": 1e6, "th": 1e3, "": 1.0}[unit]


def _parse_page(html: str) -> list[tuple[str, float]]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[tuple[str, float]] = []
    table = soup.find("table", class_="items")
    if table is None:
        return rows
    for tr in table.select("tbody > tr"):
        link = tr.find("td", class_="hauptlink")
        if not link or not link.get_text(strip=True):
            continue
        team = link.get_text(strip=True)
        # Market value is the right-aligned money cell in the row.
        mv = None
        for td in tr.find_all("td", class_="rechts"):
            mv = _money_to_eur(td.get_text(strip=True))
            if mv is not None:
                break
        if mv is not None:
            rows.append((team, mv))
    return rows


def scrape(max_pages: int = 10, pause: float = 1.0) -> pd.DataFrame:
    """Scrape national-team market values. Returns team, market_value_eur, retrieved_on."""
    records: dict[str, float] = {}
    try:
        for page in range(1, max_pages + 1):
            url = BASE if page == 1 else f"{BASE}/ajax/yw1/page/{page}"
            r = requests.get(BASE, headers=HEADERS, params={"page": page}, timeout=25)
            r.raise_for_status()
            page_rows = _parse_page(r.text)
            if not page_rows:
                break
            for team, mv in page_rows:
                records.setdefault(team, mv)
            time.sleep(pause)
    except Exception as e:  # noqa: BLE001 — defensive: never break the pipeline
        print(f"[market_value] scrape failed ({type(e).__name__}: {e}); "
              f"writing {len(records)} rows gathered so far.")

    if not records:
        print("[market_value] no data scraped — market-value feature will be empty.")
        df = pd.DataFrame(columns=["team", "market_value_eur", "retrieved_on"])
        df.to_parquet(OUT, index=False)
        return df

    df = pd.DataFrame(
        {"team_raw": list(records), "market_value_eur": list(records.values())}
    )
    df["team"] = df["team_raw"].map(lambda t: standardize_name(TM_ALIASES.get(t, t)))
    df["retrieved_on"] = pd.Timestamp.today().normalize()
    df = df[["team", "market_value_eur", "retrieved_on"]].sort_values(
        "market_value_eur", ascending=False
    ).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"[market_value] {len(df)} national teams "
          f"(top: {df.iloc[0]['team']} €{df.iloc[0]['market_value_eur']/1e9:.2f}bn) "
          f"-> {OUT.name}")
    return df


if __name__ == "__main__":
    scrape()
