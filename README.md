# Data Science Project Template

problem → approach → demo → impact

A standardized folder structure for data science projects covering the complete project lifecycle.

## Project Structure

```
├── data/
│   ├── raw/              # Original, immutable data dump
│   ├── interim/          # Intermediate data that has been transformed
│   ├── processed/        # Final, canonical data sets for modeling
│   └── external/         # Data from third party sources
│
├── notebooks/            # Jupyter notebooks for exploration and analysis
│                         # Naming convention: number-initials-description.ipynb
│                         # Example: 1.0-jqp-initial-data-exploration.ipynb
│
├── src/                 # Source code for use in this project
│   ├── data/            # Scripts to download or generate data
│   ├── features/        # Scripts for feature engineering
│   ├── models/          # Scripts to train models and make predictions
│   └── visualization/   # Scripts to create visualizations
│
├── smodels/             # Trained and serialized models
│   ├── trained/         # Saved model files (.pkl, .h5, .pt, etc.)
│   └── predictions/     # Model predictions and scores
│
├── reports/             # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/         # Generated graphics and figures for reporting
│
├── tests/               # Unit tests and integration tests
│
├── configs/             # Configuration files (hyperparameters, paths, etc.)
│
├── docs/                # Project documentation
│
├── requirements.txt     # Python package dependencies
│
└── README.md            # Project overview and instructions
```

## Data Science Lifecycle Coverage

This structure supports all major phases:

1. **Data Collection** → `data/raw/`, `data/external/`, `src/data/`
2. **Data Exploration** → `notebooks/`, `reports/figures/`
3. **Data Preprocessing** → `data/interim/`, `data/processed/`, `src/data/`
4. **Feature Engineering** → `src/features/`
5. **Model Development** → `notebooks/`, `src/models/`
6. **Model Training** → `src/models/`, `models/trained/`
7. **Model Evaluation** → `smodels/predictions/`, `reports/`
8. **Documentation** → `docs/`, `reports/`
9. **Testing** → `tests/`
10. **Deployment** → `src/models/`, `configs/`

## Getting Started

1. Install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Add your raw data to `data/raw/`

3. Start exploring in `notebooks/`

4. Use `notebooks/00-template.ipynb` as a starting point

5. Move production code to `src/`

## Best Practices

- Keep raw data immutable - never edit files in `data/raw/`
- Use notebooks for exploration, move production code to `src/`
- Version control your code, not your data
- Document your process in notebooks and `docs/`
- Use meaningful names and maintain a consistent structure

Next Steps

  1. Install dependencies: pip install -r requirements.txt
  2. Add your raw data to data/raw/
  3. Use notebooks/00-template.ipynb as a starting point
  4. Move production-ready code from notebooks to src/
