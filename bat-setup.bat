@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not on PATH.
    echo Install: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

echo === Step 1/2: Sync dependencies ===
uv sync
if errorlevel 1 (
    echo [ERROR] uv sync failed.
    pause
    exit /b 1
)

echo.
echo === Step 2/2: Build dataset (downloads from GitHub, skips Transfermarkt) ===
uv run python src/etl/make_dataset.py --skip-scrape
if errorlevel 1 (
    echo [ERROR] Dataset build failed.
    pause
    exit /b 1
)

echo.
echo Setup complete. Run bat-run_ui.bat to open the app.
pause
