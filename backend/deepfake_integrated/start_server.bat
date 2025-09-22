@echo off
echo Starting Deepfake Detection Server...
echo.

cd /d "%~dp0"
cd backend

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo.

uvicorn server:app --host 0.0.0.0 --port 8000 --reload
