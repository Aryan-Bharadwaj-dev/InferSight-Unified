#!/usr/bin/env python3
"""
Test script for the steganography detection tool
Creates sample files and tests the detection capabilities
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import wave

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from stego_detector import SteganographyDetector

def create_test_image():
    """Create a test image with high entropy LSBs (simulated steganography)"""
    # Create a random image
    width, height = 200, 200
    image_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Modify LSBs to have high entropy (simulate steganography)
    random_bits = np.random.randint(0, 2, (height, width, 3))
    image_data = (image_data & 0xFE) | random_bits
    
    img = Image.fromarray(image_data)
    test_path = Path("test_image_with_stego.png")
    img.save(test_path)
    return test_path

def create_clean_image():
    """Create a clean test image without steganography"""
    # Create a simple gradient image
    width, height = 200, 200
    image_data = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            image_data[i, j] = [i % 256, j % 256, (i + j) % 256]
    
    img = Image.fromarray(image_data)
    test_path = Path("test_image_clean.png")
    img.save(test_path)
    return test_path

def create_test_audio():
    """Create a test audio file with high entropy LSBs"""
    sample_rate = 44100
    duration = 2  # seconds
    samples = int(sample_rate * duration)
    
    # Generate audio with random LSBs
    audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)
    
    test_path = Path("test_audio_with_stego.wav")
    with wave.open(str(test_path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return test_path

def create_test_text():
    """Create a test text file with suspicious patterns"""
    suspicious_text = """This is a test text file.
The	text	has	unusual	tabs	everywhere.
                    And lots of spaces.
This might indicate steganography using whitespace encoding.
Some words repeat repeat repeat repeat repeat repeat many times.
"""
    
    test_path = Path("test_text_suspicious.txt")
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(suspicious_text)
    
    return test_path

def run_tests():
    """Run tests on created sample files"""
    print("Steganography Detection Tool - Test Script")
    print("=" * 50)
    
    detector = SteganographyDetector()
    
    # Create test files
    print("\nCreating test files...")
    
    try:
        stego_image = create_test_image()
        clean_image = create_clean_image()
        stego_audio = create_test_audio()
        suspicious_text = create_test_text()
        
        print(f"Created: {stego_image}")
        print(f"Created: {clean_image}")
        print(f"Created: {stego_audio}")
        print(f"Created: {suspicious_text}")
        
        # Test image with simulated steganography
        print(f"\n--- Testing {stego_image} ---")
        result = detector.detect_file(stego_image)
        if result and result.get('detections'):
            print(f"[ALERT] Potential steganography detected!")
            for detection in result['detections']:
                print(f"  - {detection['method']}: {detection['confidence']:.2f}% confidence")
                print(f"    {detection['description']}")
        else:
            print("[CLEAN] No steganography detected")
        
        # Test clean image
        print(f"\n--- Testing {clean_image} ---")
        result = detector.detect_file(clean_image)
        if result and result.get('detections'):
            print(f"[ALERT] Potential steganography detected!")
            for detection in result['detections']:
                print(f"  - {detection['method']}: {detection['confidence']:.2f}% confidence")
        else:
            print("[CLEAN] No steganography detected")
        
        # Test audio with simulated steganography
        print(f"\n--- Testing {stego_audio} ---")
        result = detector.detect_file(stego_audio)
        if result and result.get('detections'):
            print(f"[ALERT] Potential steganography detected!")
            for detection in result['detections']:
                print(f"  - {detection['method']}: {detection['confidence']:.2f}% confidence")
                print(f"    {detection['description']}")
        else:
            print("[CLEAN] No steganography detected")
        
        # Test suspicious text
        print(f"\n--- Testing {suspicious_text} ---")
        result = detector.detect_file(suspicious_text)
        if result and result.get('detections'):
            print(f"[ALERT] Potential steganography detected!")
            for detection in result['detections']:
                print(f"  - {detection['method']}: {detection['confidence']:.2f}% confidence")
                print(f"    {detection['description']}")
        else:
            print("[CLEAN] No steganography detected")
        
        print("\n" + "=" * 50)
        print("Test completed! Clean up test files if desired.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure all required dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    run_tests()
