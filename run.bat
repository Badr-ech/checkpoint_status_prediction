@echo off
REM Quick start script for the API server

echo Starting Checkpoint Status Prediction API...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if models exist
if not exist models\checkpoint_models_latest.joblib (
    echo.
    echo WARNING: No trained models found!
    echo Please train models first:
    echo   python -m src.models.train
    echo.
    echo Starting API anyway for testing...
    timeout /t 3
)

REM Start the API server
python -m src.api.main

pause
