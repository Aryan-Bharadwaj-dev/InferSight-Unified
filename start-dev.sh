#!/bin/bash

# InferSight Complete Development Environment Startup Script
# This script starts both backend and frontend servers concurrently

echo "🚀 Starting InferSight Development Environment"
echo "=============================================="

# Check if we're in the correct directory
if [ ! -f "start-backend.sh" ] || [ ! -f "start-frontend.sh" ]; then
    echo "❌ Please run this script from the project root directory."
    exit 1
fi

# Make scripts executable
chmod +x start-backend.sh
chmod +x start-frontend.sh

# Check if 'concurrently' is available for running both servers
if command -v concurrently &> /dev/null; then
    echo "🔄 Starting backend and frontend concurrently..."
    concurrently \
        --names "BACKEND,FRONTEND" \
        --prefix-colors "blue,green" \
        "./start-backend.sh" \
        "./start-frontend.sh"
else
    echo "⚠️  'concurrently' not found. Installing globally..."
    npm install -g concurrently
    
    if command -v concurrently &> /dev/null; then
        echo "🔄 Starting backend and frontend concurrently..."
        concurrently \
            --names "BACKEND,FRONTEND" \
            --prefix-colors "blue,green" \
            "./start-backend.sh" \
            "./start-frontend.sh"
    else
        echo "❌ Failed to install 'concurrently'. Please start servers manually:"
        echo "   Terminal 1: ./start-backend.sh"
        echo "   Terminal 2: ./start-frontend.sh"
        exit 1
    fi
fi