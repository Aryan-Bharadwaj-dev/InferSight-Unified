#!/usr/bin/env python3
"""
Test script to verify the integrated deepfake detection structure
"""
import os
from pathlib import Path

def test_structure():
    """Test that all required files are present"""
    base_dir = Path(__file__).parent
    
    # Required directories
    required_dirs = [
        "backend",
        "static", 
        "uploads"
    ]
    
    # Required backend files
    required_backend_files = [
        "backend/server.py",
        "backend/models.py", 
        "backend/detection_service.py",
        "backend/deepfake_detector.py",
        "backend/file_handler.py",
        "backend/requirements.txt",
        "backend/.env"
    ]
    
    # Required static files
    required_static_files = [
        "static/deepfake.html",
        "static/styles.css",
        "static/logo.png"
    ]
    
    print("Testing integrated deepfake detection structure...")
    print("=" * 50)
    
    # Test directories
    print("\nChecking directories:")
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - MISSING")
    
    # Test backend files
    print("\nChecking backend files:")
    for file_path in required_backend_files:
        full_path = base_dir / file_path
        if full_path.exists() and full_path.is_file():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
    
    # Test static files
    print("\nChecking static files:")
    for file_path in required_static_files:
        full_path = base_dir / file_path
        if full_path.exists() and full_path.is_file():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
    
    # Test HTML content
    print("\nChecking HTML integration:")
    deepfake_html = base_dir / "static/deepfake.html"
    if deepfake_html.exists():
        content = deepfake_html.read_text()
        if '/static/styles.css' in content:
            print("✓ CSS path updated correctly")
        else:
            print("✗ CSS path not updated")
            
        if '/static/logo.png' in content:
            print("✓ Logo path updated correctly") 
        else:
            print("✗ Logo path not updated")
            
        if 'API_BASE_URL' in content:
            print("✓ Backend API integration present")
        else:
            print("✗ Backend API integration missing")
    
    print("\n" + "=" * 50)
    print("Structure verification complete!")
    print("\nTo start the server:")
    print("1. Run: start_server.bat")
    print("2. Open browser to: http://localhost:8000")

if __name__ == "__main__":
    test_structure()
