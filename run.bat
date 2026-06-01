@echo off
cd /d %~dp0

:: Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate
) else (
    echo Virtual environment not found. Run install.bat first!
    pause
    exit /b 1
)

:: Quick check — feedparser and yaml are the most commonly missing
python -c "import feedparser, yaml, streamlit" 2>nul
if errorlevel 1 (
    echo.
    echo Some required packages are missing. Installing now...
    pip install -r requirements.txt --quiet
    echo Done.
    echo.
)

streamlit run market_dashboard/app.py
pause
