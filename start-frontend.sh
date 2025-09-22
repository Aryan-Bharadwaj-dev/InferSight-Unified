#!/bin/bash

# InferSight Frontend Startup Script
# This script starts the React frontend development server

echo "🎨 Starting InferSight React Frontend..."
echo "====================================="

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 14+ to continue."
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm to continue."
    exit 1
fi

# Check if we're in the correct directory
if [ ! -d "frontend" ]; then
    echo "❌ Please run this script from the project root directory."
    echo "Usage: ./start-frontend.sh"
    exit 1
fi

# Navigate to frontend directory
cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
else
    echo "📦 Dependencies already installed"
fi

echo ""
echo "🚀 Starting the React development server..."
echo "🌐 Frontend will be available at: http://localhost:3000"
echo "🔄 Hot reloading is enabled for development"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the development server
npm start