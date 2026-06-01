"""
Temporal cross-validation folds for the WC-2026 predictor.

Expanding-windows, calendar-based walk-forward:
    for each validation year Y:
        train = all PLAYED matches with date.year <  Y
        val   = all PLAYED matches with date.year == Y

This is the ONLY validation scheme used in the project. Random K-fold is forbidden here: it 
would train on the future to predict the past (leakage), inflating metrics that then collapse during live tournament.

The splitter does exactly one thing: partition row indices by time. It does NOT impute, scale, or drop features --
that is the model's responsability.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

# Recent, data-dense years. 2020 is intentionally skipped (COVID: only ~350 matches, an unrepresentative sample).
# We can add it back if we want 
DEFAULT_VAL_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]


@dataclass
class Fold:
    name: str          # e.g. "val2022"
    val_year: int
    train_idx: pd.Index    # row labels for training
    val_idx: pd.Index      # row labels for validation

def temporal_folds(
        df: pd.DataFrame,
        val_years: list[int] = DEFAULT_VAL_YEARS,
        *,
        date_col: str = "date",
        played_col: str = "played",
):
    """
    Yield exapnding-windows walk-forward folds (oldest validation year first).
    Unplayed fixtures (the 2026 WC) are excluded from every fold.
    """
    played = df[df[played_col]]
    years = played[date_col].dt.year
    for y in val_years:
        train_mask = (years < y).to_numpy()
        val_mask = (years == y).to_numpy()
        if val_mask.sum() == 0:
            continue   # no matches that year -> skip the fold
        yield Fold(
            name=f"val{y}",
            val_year=y,
            train_idx=played.index[train_mask],
            val_idx=played.index[val_mask],
        )


if __name__ == "__main__":
    # Print the fold table against the real feature store, so we can SEE it.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # puth src/ on path
    from etl.paths import PROCESSED_DIR

    df = pd.read_parquet(PROCESSED_DIR / "matches_features.parquet")
    print(f"{'fold':>8}  {'train':>7}  {'val':>5}")
    for f in temporal_folds(df):
        print(f"{f.name:>8}  {len(f.train_idx):>7}  {len(f.val_idx):>5}")