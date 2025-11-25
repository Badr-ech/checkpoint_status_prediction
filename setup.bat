@echo off
REM Checkpoint Status Prediction System - Setup and Run Script

echo ================================================
echo Checkpoint Status Prediction System Setup
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Installing dependencies...
pip install -r requirements.txt

echo.
echo Step 4: Setting up environment variables...
if not exist .env (
    copy .env.example .env
    echo.
    echo ================================================
    echo IMPORTANT: Please edit .env file with your API keys:
    echo   - Telegram API credentials
    echo   - Reddit API credentials
    echo   - Google Maps API key
    echo ================================================
    echo.
    pause
)

echo.
echo Step 5: Initializing database...
python -m src.database.init_db

echo.
echo Step 6: Initializing checkpoints...
python -m src.collectors.init_checkpoints --init

echo.
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo Next steps:
echo 1. Start data collection (run for 7+ days):
echo    python -m src.collectors.telegram_collector
echo    python -m src.collectors.reddit_collector
echo.
echo 2. After collecting data, train models:
echo    python -m src.models.train
echo.
echo 3. Start the API server:
echo    python -m src.api.main
echo.
echo 4. Open browser to: http://localhost:8000
echo ================================================
echo.
pause
