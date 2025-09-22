"""
Deep Learning Model for Image Steganography Detection
Uses CNN architecture to detect steganography in images
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from pathlib import Path
import pickle
import os

class StegoImageDataset(Dataset):
    """Dataset for steganography detection training"""
    
    def __init__(self, data_dir, transform=None, target_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Load image paths and labels
        self.samples = []
        if (self.data_dir / 'clean').exists():
            clean_images = list((self.data_dir / 'clean').glob('*.*'))
            self.samples.extend([(img, 0) for img in clean_images])
        
        if (self.data_dir / 'stego').exists():
            stego_images = list((self.data_dir / 'stego').glob('*.*'))
            self.samples.extend([(img, 1) for img in stego_images])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.target_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(image) / 255.0
            
            # Apply transforms if provided
            if self.transform:
                img_array = self.transform(img_array)
            
            # Convert to tensor
            img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)
            
            return img_tensor, torch.LongTensor([label])
            
        except Exception as e:
            # Return dummy data if image loading fails
            return torch.zeros(3, *self.target_size), torch.LongTensor([0])

class StegoDetectionCNN(nn.Module):
    """CNN model for steganography detection"""
    
    def __init__(self, num_classes=2):
        super(StegoDetectionCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size of flattened features
        # After 4 pooling operations: 224/16 = 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)

class ImageStegoClassifier:
    """Wrapper class for the CNN model with training and inference capabilities"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StegoDetectionCNN()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.model.to(self.device)
    
    def train_model(self, data_dir, epochs=20, batch_size=32, learning_rate=0.001, save_path=None):
        """Train the CNN model"""
        
        # Create dataset and dataloader
        dataset = StegoImageDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(dataloader):
                data, targets = data.to(self.device), targets.squeeze().to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
            
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100. * correct / total
            print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
            print(f'Model saved to {save_path}')
    
    def predict(self, image_path):
        """Predict if an image contains steganography"""
        self.model.eval()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            img_array = np.array(image) / 255.0
            img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = outputs.cpu().numpy()[0]
                
                return {
                    'clean_probability': float(probabilities[0]),
                    'stego_probability': float(probabilities[1]),
                    'prediction': 'steganography' if probabilities[1] > probabilities[0] else 'clean',
                    'confidence': float(max(probabilities)) * 100
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'clean_probability': 0.5,
                'stego_probability': 0.5,
                'prediction': 'unknown',
                'confidence': 0.0
            }
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': 'StegoDetectionCNN'
        }, path)
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
