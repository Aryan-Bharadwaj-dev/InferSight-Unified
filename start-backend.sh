#!/bin/bash

# InferSight Unified Backend Startup Script
# This script starts the unified backend server

echo "🔍 Starting InferSight Unified Backend Server..."
echo "========================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ to continue."
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "backend/unified_server.py" ]; then
    echo "❌ Please run this script from the project root directory."
    echo "Usage: ./start-backend.sh"
    exit 1
fi

# Navigate to backend directory
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -r ../requirements.txt

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo ""
echo "🚀 Starting the unified backend server..."
echo "📊 API Documentation will be available at: http://localhost:8000/docs"
echo "🌐 API Root will be available at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 unified_server.py