"""
Central path constants for the ETL / feature pipeline.

Keeping every directory in one place avoids scattered ``Path(__file__).parents[...]``
arithmetic and makes the pipeline easy to relocate.
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw sub-locations
INTL_RESULTS_DIR = RAW_DIR / "international-results"  # martj42 (downloaded)
WORLDCUP_CSV_DIR = RAW_DIR / "worldcup" / "data-csv"  # jfjelstul (on disk)
FIFA_RANK_CSV = RAW_DIR / "fifa_mens_rank.csv"        # on disk
FIFA_PLAYERS_CSV = RAW_DIR / "FIFA-players" / "fifa_cleaned.csv"

for _d in (INTERIM_DIR, PROCESSED_DIR, INTL_RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
