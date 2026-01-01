@echo off
REM ============================================
REM Academic Burnout Prevention System - Startup Script (Windows)
REM ============================================

echo.
echo ========================================
echo  Academic Burnout Prevention System
echo  Starting up...
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Check if requirements are installed
echo [INFO] Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo.
echo [INFO] Starting the Burnout Advisor API Server...
echo [INFO] API will be available at: http://localhost:8000
echo [INFO] Swagger Docs: http://localhost:8000/docs
echo [INFO] Frontend: Open frontend/index.html in your browser
echo.
echo [INFO] Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the server
cd backend
python main.py

pause
