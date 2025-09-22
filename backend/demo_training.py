#!/usr/bin/env python3
"""
Quick demo of InferSight's enhanced training capabilities
Generates a small dataset and trains a simple model
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from training.data_generator import SteganographyDataGenerator
    from training.train_models import ModelTrainer
    print("✅ Enhanced ML modules loaded successfully!")
    
    # Quick demo of data generation
    print("\n🎨 Generating sample training data...")
    data_gen = SteganographyDataGenerator()
    
    # Generate a few sample images
    clean_img = data_gen.generate_clean_image()
    print("✅ Generated clean image")
    
    stego_img = data_gen.apply_lsb_steganography(clean_img, "Hidden message for demo")
    print("✅ Generated steganographic image with LSB embedding")
    
    # Test different steganography methods
    noise_stego = data_gen.apply_noise_based_stego(clean_img)
    print("✅ Generated noise-based steganographic image")
    
    freq_stego = data_gen.apply_frequency_domain_stego(clean_img)
    print("✅ Generated frequency-domain steganographic image")
    
    print("\n🧠 Training capabilities ready!")
    print("To run full training:")
    print("  python training/train_models.py --epochs 10")
    
    print("\n🚀 InferSight Enhanced Version Features:")
    print("  ✅ Synthetic data generation")
    print("  ✅ Multiple steganography techniques")
    print("  ✅ CNN-based detection model") 
    print("  ✅ Comprehensive training pipeline")
    print("  ✅ Model evaluation and benchmarking")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("  source venv/bin/activate")
    print("  pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check that you're running from the correct directory")

print("\n" + "="*60)
print("InferSight - Advanced Steganography Detection Tool")
print("Now with enhanced ML training capabilities! 🔍🧠")
print("="*60)
