@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not on PATH.
    pause
    exit /b 1
)

if not exist "mlflow\mlflow.db" (
    echo [WARN] mlflow\mlflow.db not found. Run model training first, e.g.:
    echo   uv run python src/models/ensemble.py
    echo.
)

echo Starting MLflow UI (local experiments)...
echo Open http://127.0.0.1:5000 in your browser.
echo Press Ctrl+C to stop.
echo.

uv run mlflow ui --backend-store-uri "sqlite:///mlflow/mlflow.db"
if errorlevel 1 pause
