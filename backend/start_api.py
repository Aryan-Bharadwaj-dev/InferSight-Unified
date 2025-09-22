#!/usr/bin/env python3
"""
Start script for the Steganography Detection API
"""

import uvicorn
import sys
from pathlib import Path

def main():
    """Start the FastAPI server"""
    print("Starting Steganography Detection API...")
    print("=" * 50)
    print("API will be available at:")
    print("  - Main API: http://localhost:8000")
    print("  - Interactive docs: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Import the FastAPI app
        from api import app
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        print(f"Error importing API: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)

if __name__ == "__main__":
    main()
