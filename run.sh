#!/bin/bash
# ============================================
# Academic Burnout Prevention System - Startup Script (Linux/Mac)
# ============================================

echo ""
echo "========================================"
echo " Academic Burnout Prevention System"
echo " Starting up..."
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed or not in PATH."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[INFO] Python version: $PYTHON_VERSION"

# Check if requirements are installed
echo "[INFO] Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "[INFO] Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies."
        exit 1
    fi
fi

echo ""
echo "[INFO] Starting the Burnout Advisor API Server..."
echo "[INFO] API will be available at: http://localhost:8000"
echo "[INFO] Swagger Docs: http://localhost:8000/docs"
echo "[INFO] Frontend: Open frontend/index.html in your browser"
echo ""
echo "[INFO] Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start the server
cd backend
python3 main.py
