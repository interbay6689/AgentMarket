@echo off
cd /d %~dp0
echo ============================================
echo  AgentMarket — Install / Update
echo ============================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Download from https://python.org/downloads
    pause
    exit /b 1
)

:: Create venv if it doesn't exist
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Done.
)

:: Activate venv
call venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)

:: Upgrade pip silently
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install all requirements
echo Installing packages (this may take 1-3 minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Some packages failed to install. Check the output above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Installation complete!
echo  Run the dashboard with: run.bat
echo ============================================
pause
