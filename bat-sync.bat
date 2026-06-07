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

echo Syncing Python dependencies (uv sync)...
uv sync
if errorlevel 1 (
    echo [ERROR] uv sync failed.
    pause
    exit /b 1
)

echo.
echo Done. Dependencies are ready.
pause
