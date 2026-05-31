# Phase 2 — Data Pipeline

> It is vital to understand that, in the ML world, the raw material is data, and the
> choices we make for data have profound downstream consequences on the performance,
> scalability, and reliability of our entire ML system.

This document records the concrete plan and decisions for **Phase 2 (data collection &
dataset construction)** of the WC-2026 match-prediction project. It builds on the
strategy in [`01-project-general.md`](./01-project-general.md).

## Goal

Produce a **leakage-safe, match-level feature store** (one row per international match,
both teams' *pre-match* state) that the Phase 4 models train on with a temporal split.

```
data/raw/        raw sources, as downloaded (read-only)
data/interim/    each source cleaned + team names standardized
data/processed/  final feature store (parquet) + data dictionary
```

## Sources

### Already on disk (`data/raw/`)

| Dataset | Content | Coverage | Role |
|---|---|---|---|
| `worldcup/` (jfjelstul) | Clean WC matches, squads, goals, standings | WC only, 1930–2022 (men + women) | WC match metadata, tagging |
| `All-the-results.../` | WC match results (Kaggle) | WC only, 1930–2014 | Backup, ignored (redundant) |
| `FIFA-World-Cup/` | Per-tournament standings & summary | WC only | Tournament context |
| `fifa_mens_rank.csv` | FIFA ranking | 1992–2024, **year+semester** granularity | FIFA-rank feature (coarse) |
| `FIFA-players/fifa_cleaned.csv` | EA FIFA videogame player ratings | ~one snapshot (~FIFA 19) | Squad-strength proxy (optional) |
| `FIFA-World-Cup-2022/` (PFF FC) | Event + tracking data | 2022 WC only | xG source — **deferred** |

### Fetched from the internet

| Source | Method | Status |
|---|---|---|
| **martj42/international_results** (`results.csv`, `shootouts.csv`, `goalscorers.csv`, ~48k matches 1872→present) | GitHub raw CSV download | **Backbone** — required for Elo/form/H2H |
| Fresh FIFA ranking (post-2024) + 2026 WC fixtures / qualified teams | scrape / public source | Included |
| Transfermarkt squad market value | BeautifulSoup scrape | Included (fragile — defer if bot-blocked) |
| Elo ratings | **computed from the match corpus** (World Football Elo formula) | No scraping of eloratings.net (more reproducible) |
| xG (broad history) | — | Skipped (paywalled; only 2022 available locally) |

## Pipeline stages

```
ingest raw → standardize team names → canonical match table → leakage-safe features → processed store
```

1. **Ingest** (`src/etl/sources/*.py`) — download missing sources into `data/raw/`.
2. **Team-name standardization** (`src/etl/team_names.py`) — the single biggest integration
   headache: every source spells countries differently ("USA" / "United States",
   "Korea Republic" / "South Korea", "IR Iran" / "Iran"). One canonical map drives every
   join. Also maps each team → confederation.
3. **Canonical match table** (`src/etl/build_match_table.py`) — one row per match across
   all of history, with standardized names, `neutral` flag, and WC/tournament tags.
4. **Feature engineering** (`src/features/build_features.py`) — for every match, compute
   each team's state **using only matches strictly before that match's date**:
   - Elo rating (home/away/diff) — computed incrementally over the corpus
   - FIFA rank (home/away/diff)
   - Recent form: last-N goals for/against, win-rate, with exponential decay
   - Head-to-head: win-rate, goal average
   - Rest days since each team's previous match
   - Confederation (home/away) and same-confederation flag
   - Squad market value (when available)
   - **Targets:** `result_3way` (W/D/L from home perspective), `home_goals`, `away_goals`
5. **Orchestration** (`src/etl/make_dataset.py`) — runs the full pipeline end to end.

## Non-negotiable guardrail: no temporal leakage

K-fold on temporal data lets the model "see the future". Every engineered feature is
computed from data **strictly prior** to the match date, so the dataset is safe for the
time-series split used in Phase 4. Features are built by iterating matches in
chronological order and snapshotting state *before* applying each result.

## Output

- `data/processed/matches_features.parquet` — the model-ready feature store.
- `data/processed/data_dictionary.md` — every column documented.
- `data/processed/elo_history.parquet` — per-team Elo time series (reusable).
