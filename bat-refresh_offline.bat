@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not on PATH.
    pause
    exit /b 1
)

echo Offline refresh (reuse cached data, rebuild features, predict)...
uv run python src/etl/refresh_tournament.py --skip-download --skip-scrape
if errorlevel 1 (
    echo [ERROR] offline refresh failed.
    pause
    exit /b 1
)

echo.
echo Done. No internet download was performed.
pause
