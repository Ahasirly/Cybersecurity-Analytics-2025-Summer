#!/bin/bash
# Cybersecurity Fusion System - Backend Startup Script

echo "🚀 Starting Cybersecurity Fusion System Backend..."

# Navigate to backend directory
cd backend

# Check if virtual environment exists
if [ -d "../venv" ]; then
    echo "📦 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Start Flask server
echo "🌐 Starting Flask server on http://localhost:5000"
python app.py 