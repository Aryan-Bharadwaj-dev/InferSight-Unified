#!/usr/bin/env pwsh

Write-Host "Starting Deepfake Detection Server..." -ForegroundColor Green
Write-Host ""

# Get the directory of this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Set-Location backend

Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host ""

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
