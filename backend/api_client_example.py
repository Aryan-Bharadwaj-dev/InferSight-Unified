#!/usr/bin/env python3
"""
Example client for the Steganography Detection API
Demonstrates how to use the API endpoints
"""

import requests
import json
import time
from pathlib import Path

# API base URL
API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test API health endpoint"""
    print("Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✓ API is healthy")
            return True
        else:
            print(f"✗ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure it's running on localhost:8000")
        return False

def get_api_info():
    """Get API information"""
    print("\nGetting API information...")
    response = requests.get(f"{API_BASE_URL}/")
    if response.status_code == 200:
        info = response.json()
        print(f"API Version: {info['version']}")
        print("Supported formats:")
        for format_type, extensions in info['supported_formats'].items():
            print(f"  {format_type}: {', '.join(extensions)}")
        return True
    return False

def analyze_file_sync(file_path):
    """Synchronously analyze a file"""
    print(f"\nAnalyzing file synchronously: {file_path}")
    
    if not Path(file_path).exists():
        print(f"✗ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/octet-stream')}
            response = requests.post(f"{API_BASE_URL}/analyze", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Analysis completed for {result['file_name']}")
            print(f"  File type: {result['file_type']}")
            print(f"  File size: {result['file_size']} bytes")
            
            if result['detections']:
                print("  🚨 SUSPICIOUS ACTIVITY DETECTED:")
                for detection in result['detections']:
                    print(f"    - {detection['method']}: {detection['confidence']:.1f}% confidence")
                    print(f"      {detection['description']}")
            else:
                print("  ✓ No steganography detected")
            
            return result
        else:
            print(f"✗ Analysis failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Error analyzing file: {e}")
        return None

def analyze_file_async(file_path):
    """Asynchronously analyze a file"""
    print(f"\nAnalyzing file asynchronously: {file_path}")
    
    if not Path(file_path).exists():
        print(f"✗ File not found: {file_path}")
        return None
    
    try:
        # Upload file
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/octet-stream')}
            response = requests.post(f"{API_BASE_URL}/upload", files=files)
        
        if response.status_code != 200:
            print(f"✗ Upload failed: {response.status_code} - {response.text}")
            return None
        
        upload_result = response.json()
        file_id = upload_result['file_id']
        print(f"✓ File uploaded with ID: {file_id}")
        
        # Poll for results
        max_attempts = 30
        for attempt in range(max_attempts):
            response = requests.get(f"{API_BASE_URL}/status/{file_id}")
            if response.status_code == 200:
                status = response.json()
                print(f"  Status: {status['status']} - {status['message']}")
                
                if status['status'] == 'completed':
                    # Get results
                    response = requests.get(f"{API_BASE_URL}/results/{file_id}")
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✓ Analysis completed for {result['file_name']}")
                        
                        if result['detections']:
                            print("  🚨 SUSPICIOUS ACTIVITY DETECTED:")
                            for detection in result['detections']:
                                print(f"    - {detection['method']}: {detection['confidence']:.1f}% confidence")
                                print(f"      {detection['description']}")
                        else:
                            print("  ✓ No steganography detected")
                        
                        return result
                    break
                elif status['status'] == 'failed':
                    print("✗ Analysis failed")
                    break
                    
            time.sleep(1)  # Wait 1 second before next poll
        
        print("✗ Analysis timed out")
        return None
        
    except Exception as e:
        print(f"✗ Error analyzing file: {e}")
        return None

def download_report(file_id):
    """Download analysis report"""
    print(f"\nDownloading report for file ID: {file_id}")
    
    try:
        response = requests.get(f"{API_BASE_URL}/results/{file_id}/report")
        if response.status_code == 200:
            report_filename = f"report_{file_id}.json"
            with open(report_filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Report downloaded: {report_filename}")
            return report_filename
        else:
            print(f"✗ Failed to download report: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error downloading report: {e}")
        return None

def main():
    """Main function demonstrating API usage"""
    print("Steganography Detection API - Client Example")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        print("\nMake sure the API is running:")
        print("python api.py")
        return
    
    # Get API info
    get_api_info()
    
    # Test with sample files (if they exist)
    test_files = [
        "test_image_with_stego.png",
        "test_image_clean.png",
        "test_audio_with_stego.wav",
        "test_text_suspicious.txt"
    ]
    
    print(f"\n{'='*50}")
    print("Testing with sample files...")
    
    for file_path in test_files:
        if Path(file_path).exists():
            # Test synchronous analysis
            result = analyze_file_sync(file_path)
            
            # Test asynchronous analysis
            # result = analyze_file_async(file_path)
            
        else:
            print(f"Sample file not found: {file_path}")
    
    print(f"\n{'='*50}")
    print("API client example completed!")
    print("\nTo create test files, run:")
    print("python test_detector.py")

if __name__ == "__main__":
    main()
