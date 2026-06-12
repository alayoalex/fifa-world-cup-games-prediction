"""
End-to-end Phase 2 data pipeline.

    raw sources -> standardize -> canonical match table -> leakage-safe features

Run:
    python src/etl/make_dataset.py               # full pipeline
    python src/etl/make_dataset.py --skip-download   # reuse cached martj42 CSVs
    python src/etl/make_dataset.py --skip-scrape     # skip Transfermarkt scrape
    python src/etl/make_dataset.py --force           # re-download martj42

See docs/03-project-data-pipeline.md for the design.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `etl` / `features` importable whether run as a script or a module.
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from etl.sources import (  # noqa: E402
    fifa_ranking,
    fixtures_2026,
    international_results,
    market_value,
    worldcup_matches,
)
from etl import build_match_table  # noqa: E402
from features import build_features  # noqa: E402


def _step(n: int, total: int, title: str) -> None:
    print(f"\n=== [{n}/{total}] {title} ===")


def main(skip_download: bool = False, skip_scrape: bool = False, force: bool = False) -> None:
    total = 7

    _step(1, total, "Download raw sources")
    if skip_download:
        print("[make_dataset] --skip-download: using cached raw files where present")
    else:
        international_results.download(force=force)
        fifa_ranking.download(force=force)
        worldcup_matches.download(force=force)

    _step(2, total, "Clean FIFA ranking")
    fifa_ranking.refresh()
    fifa_ranking.clean()

    _step(3, total, "Clean World Cup matches (jfjelstul)")
    worldcup_matches.clean()

    _step(4, total, "Extract 2026 fixtures")
    fixtures_2026.clean()

    _step(5, total, "Scrape national-team market values (Transfermarkt)")
    if skip_scrape:
        print("[make_dataset] --skip-scrape: keeping existing market_value.parquet if present")
        if not (market_value.OUT.exists()):
            market_value.scrape(max_pages=0)  # writes an empty table
    else:
        market_value.scrape()

    _step(6, total, "Build canonical match table")
    build_match_table.build()

    _step(7, total, "Engineer leakage-safe features")
    build_features.build()

    print("\n=== pipeline complete ===")
    print("Outputs in data/processed/: matches_features.parquet, elo_history.parquet, "
          "data_dictionary.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 2 WC-2026 data pipeline")
    p.add_argument("--skip-download", action="store_true", help="reuse cached martj42 CSVs")
    p.add_argument("--skip-scrape", action="store_true", help="skip Transfermarkt scrape")
    p.add_argument("--force", action="store_true", help="re-download martj42 even if cached")
    args = p.parse_args()
    main(skip_download=args.skip_download, skip_scrape=args.skip_scrape, force=args.force)
