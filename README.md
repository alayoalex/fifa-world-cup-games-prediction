# FIFA World Cup 2026 — Match Prediction

Local ML system to predict international football matches: 3-way outcomes, exact scorelines, and ensemble predictions for the 2026 World Cup.

**Runs entirely on your machine** — no cloud, no data sent to third parties.

## Quick start (Windows)

```
bat-sync.bat      → install dependencies
bat-setup.bat     → build dataset (first time)
bat-run_ui.bat    → open the app
```

## Quick start (terminal)

```bash
uv sync
uv run python src/etl/make_dataset.py --skip-scrape
uv run streamlit run src/ui/app.py
```

Then click **Refresh all predictions** in the UI.

## What you get

- **72 WC 2026 group-stage fixtures** with H/D/A probabilities
- **Exact scorelines** (Poisson: `2x0`, `1x1`, …)
- **Ensemble model** (logistic + Poisson blend)
- **Custom hypothetical matches** (e.g. Spain vs Brazil)
- **Team explorer** (Elo, form, FIFA rank)

Primary output: `data/processed/wc2026_predictions_full.csv`

## Documentation

Full reference (setup, models, UI, batch files, workflows):

**[docs/00-project-reference.md](docs/00-project-reference.md)**

Additional design docs in `docs/`.

## Stack

Python · uv · pandas · scikit-learn · Streamlit · MLflow (local, optional)
