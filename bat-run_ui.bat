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

echo Starting local UI (Streamlit)...
echo Open http://localhost:8501 in your browser.
echo Press Ctrl+C to stop.
echo.

uv run streamlit run src/ui/app.py
if errorlevel 1 pause
