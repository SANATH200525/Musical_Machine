@echo off
echo =========================================
echo MoodTune Initialization and Run Script
echo =========================================

:: Check if the virtual environment exists
IF NOT EXIST venv\Scripts\activate.bat (
    echo [1/3] Virtual environment not found. Creating 'venv'...
    python -m venv venv
) ELSE (
    echo [1/3] Virtual environment found.
)

:: Activate the environment
echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install/Upgrade requirements
echo [3/3] Installing dependencies...
pip install -r requirements.txt

:: Start the FastAPI server
echo =========================================
echo Starting Server... (Will train on first boot if artifacts.pkl is missing)
echo =========================================
uvicorn main:app --reload

pause