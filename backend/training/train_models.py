#!/usr/bin/env python3
"""
Training Script for InferSight Steganography Detection Models
This script handles training of various ML models with enhanced datasets
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import shutil
import random

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from models.image_classifier import ImageStegoClassifier
from training.data_generator import SteganographyDataGenerator

class ModelTrainer:
    """Main class for training steganography detection models"""
    
    def __init__(self, data_dir, models_dir):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize data generator
        self.data_generator = SteganographyDataGenerator()
        
    def prepare_training_data(self, num_samples=1000):
        """Prepare training dataset with clean and stego images"""
        
        print("Preparing training data...")
        
        # Create dataset directories
        train_dir = self.data_dir / 'training'
        train_dir.mkdir(exist_ok=True)
        (train_dir / 'clean').mkdir(exist_ok=True)
        (train_dir / 'stego').mkdir(exist_ok=True)
        
        # Generate clean images (simple patterns, gradients, textures)
        print("Generating clean images...")
        for i in range(num_samples // 2):
            clean_img = self.data_generator.generate_clean_image()
            clean_img.save(train_dir / 'clean' / f'clean_{i:06d}.png')
            
            if i % 100 == 0:
                print(f"Generated {i} clean images...")
        
        # Generate steganographic images
        print("Generating steganographic images...")
        for i in range(num_samples // 2):
            # Generate base image
            base_img = self.data_generator.generate_clean_image()
            
            # Apply steganography
            stego_img = self.data_generator.apply_lsb_steganography(
                base_img, 
                self.data_generator.generate_random_message()
            )
            
            stego_img.save(train_dir / 'stego' / f'stego_{i:06d}.png')
            
            if i % 100 == 0:
                print(f"Generated {i} steganographic images...")
        
        print(f"Training dataset prepared with {num_samples} samples")
        return train_dir
    
    def train_image_classifier(self, epochs=50, batch_size=32):
        """Train the CNN-based image classifier"""
        
        print("Training image steganography classifier...")
        
        # Prepare data
        train_dir = self.prepare_training_data(num_samples=2000)
        
        # Initialize classifier
        classifier = ImageStegoClassifier()
        
        # Train model
        model_path = self.models_dir / 'image_stego_detector.pth'
        classifier.train_model(
            data_dir=train_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            save_path=model_path
        )
        
        print(f"Image classifier trained and saved to {model_path}")
        return model_path
    
    def evaluate_model(self, model_path, test_data_dir):
        """Evaluate trained model performance"""
        
        print("Evaluating model performance...")
        
        classifier = ImageStegoClassifier(model_path)
        
        # Test on clean images
        clean_dir = Path(test_data_dir) / 'clean'
        if clean_dir.exists():
            clean_images = list(clean_dir.glob('*.*'))[:100]  # Test on 100 images
            clean_correct = 0
            
            for img_path in clean_images:
                result = classifier.predict(img_path)
                if result['prediction'] == 'clean':
                    clean_correct += 1
            
            clean_accuracy = (clean_correct / len(clean_images)) * 100
            print(f"Clean image accuracy: {clean_accuracy:.2f}%")
        
        # Test on stego images
        stego_dir = Path(test_data_dir) / 'stego'
        if stego_dir.exists():
            stego_images = list(stego_dir.glob('*.*'))[:100]  # Test on 100 images
            stego_correct = 0
            
            for img_path in stego_images:
                result = classifier.predict(img_path)
                if result['prediction'] == 'steganography':
                    stego_correct += 1
            
            stego_accuracy = (stego_correct / len(stego_images)) * 100
            print(f"Steganographic image accuracy: {stego_accuracy:.2f}%")
    
    def create_benchmark_dataset(self, output_dir, num_samples=500):
        """Create a benchmark dataset for testing"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        (output_dir / 'clean').mkdir(exist_ok=True)
        (output_dir / 'stego').mkdir(exist_ok=True)
        
        print("Creating benchmark dataset...")
        
        # Generate diverse clean images
        for i in range(num_samples):
            if i % 3 == 0:
                img = self.data_generator.generate_natural_image()
            elif i % 3 == 1:
                img = self.data_generator.generate_texture_image()
            else:
                img = self.data_generator.generate_clean_image()
            
            img.save(output_dir / 'clean' / f'benchmark_clean_{i:06d}.png')
        
        # Generate diverse steganographic images
        for i in range(num_samples):
            base_img = self.data_generator.generate_clean_image()
            
            # Apply different steganography techniques
            if i % 4 == 0:
                stego_img = self.data_generator.apply_lsb_steganography(base_img, "Hidden message")
            elif i % 4 == 1:
                stego_img = self.data_generator.apply_noise_based_stego(base_img)
            elif i % 4 == 2:
                stego_img = self.data_generator.apply_frequency_domain_stego(base_img)
            else:
                stego_img = self.data_generator.apply_lsb_steganography(
                    base_img, 
                    self.data_generator.generate_random_message(length=200)
                )
            
            stego_img.save(output_dir / 'stego' / f'benchmark_stego_{i:06d}.png')
        
        print(f"Benchmark dataset created at {output_dir}")
        return output_dir

def main():
    parser = argparse.ArgumentParser(description='Train InferSight Steganography Detection Models')
    parser.add_argument('--data-dir', default='./data', help='Directory for training data')
    parser.add_argument('--models-dir', default='./trained_models', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--create-benchmark', action='store_true', help='Create benchmark dataset')
    parser.add_argument('--benchmark-dir', default='./benchmark_data', help='Benchmark dataset directory')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(args.data_dir, args.models_dir)
    
    if args.create_benchmark:
        trainer.create_benchmark_dataset(args.benchmark_dir)
    
    # Train image classifier
    model_path = trainer.train_image_classifier(
        epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    if args.evaluate and Path(args.benchmark_dir).exists():
        trainer.evaluate_model(model_path, args.benchmark_dir)

if __name__ == "__main__":
    main()
