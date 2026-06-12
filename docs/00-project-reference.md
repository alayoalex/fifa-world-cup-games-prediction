# WC 2026 Match Prediction — Project Reference

Complete guide for running this repository locally. Setup uses **uv**.

**Last updated:** 2026-06-06 — personal/local mode, fully operational.

---

## What this project is

A **local machine learning system** to predict international football matches, focused on the **2026 FIFA World Cup**.

It estimates:

- **3-way outcome** — home win / draw / away win (`H` / `D` / `A`)
- **Exact scorelines** — via bivariate Poisson (`2x0`, `1x1`, …)
- **Ensemble prediction** — blend of logistic + Poisson (recommended for 3-way)

Everything runs on your machine. Predictions are written to local CSV/Parquet files. **No data is sent to external services** after the optional initial download.

```
ETL → feature store → models → CSV / Streamlit UI
```

---

## Quick start

### First time (double-click on Windows)

```
bat-sync.bat      → install dependencies
bat-setup.bat     → download data + build feature store
bat-run_ui.bat    → open the app in your browser
```

### First time (terminal)

```bash
uv sync
uv run python src/etl/make_dataset.py --skip-scrape
uv run streamlit run src/ui/app.py
```

In the UI: click **Refresh all predictions**.

### Daily use

| Goal | Action |
|------|--------|
| Open the app | `bat-run_ui.bat` or `uv run streamlit run src/ui/app.py` |
| Refresh predictions | UI → **Refresh all predictions** |
| Update after matchday | UI → **Full tournament refresh** or `bat-refresh_offline.bat` (no download) |
| Custom match (Spain vs Brazil) | UI → **Custom matches** tab, or `add_fixture.py` |

---

## Windows batch files (`bat-*.bat`)

Commands that are **not** tied to the Streamlit UI:

| File | Command |
|------|---------|
| `bat-run_ui.bat` | Start Streamlit UI |
| `bat-sync.bat` | `uv sync` — install/update dependencies |
| `bat-setup.bat` | `uv sync` + `make_dataset.py --skip-scrape` (first-time setup) |
| `bat-refresh_offline.bat` | Rebuild features + predict **without internet** |
| `bat-mlflow_ui.bat` | Open local MLflow experiment viewer |

Predictions, tournament refresh, and custom fixtures are available **inside the UI**.

---

## Streamlit UI

```bash
uv run streamlit run src/ui/app.py
```

Default URL: `http://localhost:8501`

| Tab | Features |
|-----|----------|
| **WC 2026 fixtures** | Ensemble predictions, scorelines, charts; refresh all / full tournament refresh |
| **Custom matches** | Add hypothetical fixtures, predict, remove |
| **Team explorer** | Elo, form, FIFA rank, Elo history chart |

Sidebar shows dataset status and maps UI buttons to CLI commands.

---

## Models (implemented)

| Model | Script | Role |
|-------|--------|------|
| Baselines | `src/models/baseline.py` | Elo + majority class — must be beaten |
| Logistic | `src/models/logistic.py` | 3-way classifier (log-loss ~0.85) |
| Poisson | `src/models/poisson.py` | Expected goals + scorelines (MAE ~1.05 / 0.85) |
| **Ensemble** | `src/models/ensemble.py` | 55% logistic + 45% Poisson (log-loss ~0.853) |

### Recommended prediction command

```bash
uv run python src/models/predict_all.py           # WC 2026
uv run python src/models/predict_all.py --custom  # custom fixtures
```

**Primary output:** `data/processed/wc2026_predictions_full.csv`

Key columns:

| Column | Meaning |
|--------|---------|
| `pred_home_goals`, `pred_away_goals` | Modal score as integers (Excel-safe) |
| `predicted_score` | Score as `2x0` format (avoids Excel date parsing) |
| `score_result` | `H`/`D`/`A` from the modal scoreline |
| `ensemble_pick`, `ensemble_p_H/D/A` | **Recommended 3-way prediction** |
| `logreg_pick`, `poisson_pick` | Individual model picks |
| `lambda_home`, `lambda_away` | Expected goals |
| `top_scores` | Top 3 scorelines with probabilities |

### Tournament refresh

```bash
uv run python src/etl/refresh_tournament.py --skip-scrape              # download + rebuild + predict
uv run python src/etl/refresh_tournament.py --skip-download --skip-scrape  # offline
```

### Custom hypothetical matches

```bash
uv run python src/etl/add_fixture.py add --home Spain --away Brazil --predict
uv run python src/etl/add_fixture.py list
uv run python src/etl/add_fixture.py remove --id <fixture_id> --rebuild
```

If the fixture already exists, `add ... --predict` skips the duplicate and re-predicts.

**Output:** `data/processed/custom_predictions_full.csv`

---

## Data pipeline

```bash
uv run python src/etl/make_dataset.py
```

| Flag | Effect |
|------|--------|
| `--skip-download` | Reuse cached martj42 CSVs |
| `--skip-scrape` | Skip Transfermarkt (recommended; `mv_log_ratio` will be empty) |
| `--force` | Re-download martj42 even if cached |

**Auto-downloads on first run** (no manual files needed):

- martj42 international results
- Dato-Futbol FIFA rankings → `fifa_mens_rank.csv`
- jfjelstul World Cup matches

**Pipeline outputs** (`data/processed/`):

- `matches_features.parquet` — feature store (~49k matches + 72 WC 2026 fixtures)
- `elo_history.parquet` — per-team Elo timeline
- `data_dictionary.md` — column docs

---

## Repository layout

```
bat-*.bat                 # Windows launchers (UI, setup, offline refresh)
data/
  raw/                    # Downloaded sources (gitignored)
  interim/                # Cleaned tables + custom_fixtures.parquet
  processed/              # Feature store + prediction CSVs
src/
  etl/                    # Pipeline, add_fixture, refresh_tournament
  features/               # Leakage-safe feature engineering
  models/                 # baselines, logistic, poisson, ensemble, predict_all
  ui/app.py               # Streamlit app
mlflow/                   # Local experiment tracking (optional)
docs/                     # Project documentation
```

---

## Personal / local use

### What stays on your machine

| Component | Location |
|-----------|----------|
| Feature store | `data/processed/*.parquet` |
| Predictions | `data/processed/*_predictions_full.csv` |
| MLflow (optional) | `mlflow/mlflow.db` |
| Custom fixtures | `data/interim/custom_fixtures.parquet` |

### What uses the internet

Only `make_dataset.py` / `refresh_tournament.py` (without `--skip-download`) fetch public GitHub data. All modeling and prediction runs offline afterward.

---

## Evaluation metrics (temporal CV, verified)

| Model | Log-loss | Accuracy |
|-------|----------|----------|
| Baseline Elo | 0.95 | 60.5% |
| Logistic | 0.85 | 61.1% |
| Poisson (3-way from score matrix) | 0.86 | 60.8% |
| **Ensemble** | **0.853** | **61.1%** |

Metrics use **walk-forward validation by year** — no random K-fold (prevents temporal leakage).

---

## Optional: experiment tracking

```bash
uv run python src/models/baseline.py
uv run python src/models/logistic.py
uv run python src/models/poisson.py
uv run python src/models/ensemble.py
uv run mlflow ui --backend-store-uri "sqlite:///mlflow/mlflow.db"
# or: bat-mlflow_ui.bat
```

---

## Excel tip

Do **not** rely on `2-0` score strings — Excel converts them to dates (`feb-00`).

Use instead:

- `predicted_score` → `2x0` format
- `pred_home_goals` + `pred_away_goals` → integer columns

---

## Project phases (status)

| Phase | Status |
|-------|--------|
| 1. Problem definition | ✅ Documented |
| 2. Data pipeline | ✅ Implemented + auto-download |
| 3. Experiment tracking | ✅ MLflow (local) |
| 4. Modeling | ✅ Baselines, logistic, Poisson, ensemble |
| 5. Deployment (API/web) | ⏭️ Not needed for personal use |
| UI | ✅ Streamlit |
| Windows launchers | ✅ `bat-*.bat` |

---

## Future extensions (not implemented)

- XGBoost / LightGBM + Optuna
- Dixon-Coles Poisson variant
- Transfermarkt market value (when scrape works)
- Scheduled auto-refresh (cron / Task Scheduler)
- Public API deployment

---

## Command cheat sheet

```bash
uv sync
uv run python src/etl/make_dataset.py --skip-scrape
uv run python src/models/predict_all.py
uv run python src/etl/refresh_tournament.py --skip-scrape
uv run streamlit run src/ui/app.py
```

---

## Related docs

- [`01-project-general.md`](./01-project-general.md) — original 5-phase strategy
- [`02-project-data-pipeline.md`](./02-project-data-pipeline.md) — pipeline design
- [`sports-ml-models.md`](./sports-ml-models.md) — sports ML model landscape
