#!/usr/bin/env python3
"""
Simple script to run the Steganography Detection API
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    import uvicorn
    from api import app
    
    print("🚀 Starting Steganography Detection API...")
    print("=" * 50)
    print("Server will be available at:")
    print("  📱 Frontend: http://localhost:8000/app")
    print("  📚 API Docs: http://localhost:8000/docs")
    print("  🔧 ReDoc: http://localhost:8000/redoc")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    # Start the server
    uvicorn.run(
        app,
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0
        port=8000,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n💡 Make sure all dependencies are installed:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
    
except KeyboardInterrupt:
    print("\n👋 Server stopped by user")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)
