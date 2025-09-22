@echo off
echo Starting Deepfake Detection Server (Simple Version)...
echo.

cd /d "%~dp0"
cd backend

echo Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

py simple_server.py
