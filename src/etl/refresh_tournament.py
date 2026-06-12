"""
Refresh all local data and regenerate every prediction after new match results.

Use during the tournament: downloads the latest martj42 results, rebuilds features,
and runs the unified prediction pipeline (logistic + Poisson + ensemble).

Run:
    uv run python src/etl/refresh_tournament.py
    uv run python src/etl/refresh_tournament.py --skip-download   # offline rebuild
    uv run python src/etl/refresh_tournament.py --skip-scrape
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from etl import make_dataset  # noqa: E402
from models.predict_all import predict_everything  # noqa: E402


def main(
    skip_download: bool = False,
    skip_scrape: bool = False,
    force: bool = False,
    logreg_weight: float = 0.55,
) -> None:
    print("=== [1/2] Refresh dataset ===")
    make_dataset.main(
        skip_download=skip_download,
        skip_scrape=skip_scrape,
        force=force,
    )

    print("\n=== [2/2] Regenerate all predictions ===")
    predict_everything(logreg_weight=logreg_weight)

    print("\n=== refresh complete ===")
    print("Outputs:")
    print("  data/processed/wc2026_predictions_full.csv")
    print("  data/processed/custom_predictions_full.csv  (if custom fixtures exist)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download, rebuild, and predict (tournament refresh)")
    p.add_argument("--skip-download", action="store_true", help="reuse cached martj42 CSVs")
    p.add_argument("--skip-scrape", action="store_true", help="skip Transfermarkt scrape")
    p.add_argument("--force", action="store_true", help="re-download martj42 even if cached")
    p.add_argument("--logreg-weight", type=float, default=0.55, help="ensemble weight for logistic")
    args = p.parse_args()
    main(
        skip_download=args.skip_download,
        skip_scrape=args.skip_scrape,
        force=args.force,
        logreg_weight=args.logreg_weight,
    )
