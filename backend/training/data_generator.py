"""
Data Generator for Creating Steganographic Training Data
Generates synthetic images with and without steganographic content
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import string
import cv2
from scipy import ndimage

class SteganographyDataGenerator:
    """Generates training data for steganography detection"""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.width, self.height = image_size
    
    def generate_clean_image(self):
        """Generate a clean (non-steganographic) image"""
        
        # Randomly choose image type
        image_type = random.choice(['gradient', 'noise', 'geometric', 'texture'])
        
        if image_type == 'gradient':
            return self._generate_gradient_image()
        elif image_type == 'noise':
            return self._generate_noise_image()
        elif image_type == 'geometric':
            return self._generate_geometric_image()
        else:
            return self._generate_texture_image()
    
    def _generate_gradient_image(self):
        """Generate an image with color gradients"""
        img = Image.new('RGB', self.image_size)
        draw = ImageDraw.Draw(img)
        
        # Create gradient
        for y in range(self.height):
            for x in range(self.width):
                r = int((x / self.width) * 255)
                g = int((y / self.height) * 255)
                b = int(((x + y) / (self.width + self.height)) * 255)
                draw.point((x, y), (r, g, b))
        
        # Add some noise for realism
        img_array = np.array(img)
        noise = np.random.randint(-20, 20, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _generate_noise_image(self):
        """Generate an image with random noise patterns"""
        # Create different types of noise
        noise_type = random.choice(['gaussian', 'uniform', 'perlin'])
        
        if noise_type == 'gaussian':
            img_array = np.random.normal(128, 50, (*self.image_size, 3))
        elif noise_type == 'uniform':
            img_array = np.random.randint(0, 256, (*self.image_size, 3))
        else:  # perlin-like noise
            img_array = self._generate_perlin_noise()
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _generate_geometric_image(self):
        """Generate an image with geometric shapes"""
        img = Image.new('RGB', self.image_size, color=(random.randint(50, 200), 
                                                      random.randint(50, 200), 
                                                      random.randint(50, 200)))
        draw = ImageDraw.Draw(img)
        
        # Add random shapes
        for _ in range(random.randint(5, 15)):
            shape_type = random.choice(['rectangle', 'ellipse', 'line'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            if shape_type == 'rectangle':
                x1, y1 = random.randint(0, self.width//2), random.randint(0, self.height//2)
                x2, y2 = random.randint(x1, self.width), random.randint(y1, self.height)
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=color)
            
            elif shape_type == 'ellipse':
                x1, y1 = random.randint(0, self.width//2), random.randint(0, self.height//2)
                x2, y2 = random.randint(x1, self.width), random.randint(y1, self.height)
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=color)
            
            else:  # line
                x1, y1 = random.randint(0, self.width), random.randint(0, self.height)
                x2, y2 = random.randint(0, self.width), random.randint(0, self.height)
                draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))
        
        return img
    
    def generate_texture_image(self):
        """Generate an image with texture patterns"""
        # Create base texture
        img_array = np.random.randint(100, 156, (*self.image_size, 3))
        
        # Apply various filters to create texture
        for channel in range(3):
            # Apply random filter
            filter_type = random.choice(['gaussian', 'median', 'bilateral'])
            
            if filter_type == 'gaussian':
                img_array[:, :, channel] = ndimage.gaussian_filter(
                    img_array[:, :, channel], 
                    sigma=random.uniform(0.5, 2.0)
                )
            elif filter_type == 'median':
                img_array[:, :, channel] = ndimage.median_filter(
                    img_array[:, :, channel], 
                    size=random.choice([3, 5])
                )
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def generate_natural_image(self):
        """Generate a more natural-looking image"""
        # Create a base with multiple frequency components
        img_array = np.zeros((*self.image_size, 3))
        
        for freq in [1, 2, 4, 8]:
            for channel in range(3):
                # Create sinusoidal patterns
                x = np.linspace(0, freq * np.pi, self.width)
                y = np.linspace(0, freq * np.pi, self.height)
                X, Y = np.meshgrid(x, y)
                
                pattern = np.sin(X) * np.cos(Y) + np.cos(X) * np.sin(Y)
                pattern = (pattern + 1) * 127.5  # Normalize to 0-255
                
                img_array[:, :, channel] += pattern / freq
        
        # Add some random noise
        noise = np.random.normal(0, 10, img_array.shape)
        img_array = img_array + noise
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _generate_perlin_noise(self):
        """Generate Perlin-like noise pattern"""
        # Simple approximation of Perlin noise
        img_array = np.zeros((*self.image_size, 3))
        
        for octave in range(4):
            freq = 2 ** octave
            amp = 1.0 / (2 ** octave)
            
            for channel in range(3):
                for i in range(self.height):
                    for j in range(self.width):
                        x = j * freq / self.width
                        y = i * freq / self.height
                        noise_val = np.sin(x) * np.cos(y) + np.cos(x * 1.3) * np.sin(y * 0.7)
                        img_array[i, j, channel] += noise_val * amp * 128
        
        img_array += 128  # Offset to positive values
        return img_array
    
    def apply_lsb_steganography(self, image, message):
        """Apply LSB steganography to an image"""
        img_array = np.array(image)
        
        # Convert message to binary
        binary_message = ''.join(format(ord(char), '08b') for char in message)
        binary_message += '1111111111111110'  # End marker
        
        # Flatten image array for easier manipulation
        flat_img = img_array.flatten()
        
        # Embed message in LSBs
        for i, bit in enumerate(binary_message):
            if i < len(flat_img):
                flat_img[i] = (flat_img[i] & 0xFE) | int(bit)
        
        # Reshape back to image shape
        stego_array = flat_img.reshape(img_array.shape)
        return Image.fromarray(stego_array.astype(np.uint8))
    
    def apply_noise_based_stego(self, image):
        """Apply noise-based steganography"""
        img_array = np.array(image)
        
        # Add structured noise that could contain hidden data
        for channel in range(3):
            # Create pseudo-random pattern based on a seed
            np.random.seed(42)  # Fixed seed for consistent pattern
            noise = np.random.randint(-5, 6, img_array[:, :, channel].shape)
            img_array[:, :, channel] = np.clip(
                img_array[:, :, channel] + noise, 0, 255
            )
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def apply_frequency_domain_stego(self, image):
        """Apply frequency domain steganography simulation"""
        img_array = np.array(image)
        
        # Convert to grayscale for DCT
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply DCT to 8x8 blocks and modify coefficients
        h, w = gray.shape
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                
                # Modify mid-frequency coefficients slightly
                dct_block[2:6, 2:6] += np.random.normal(0, 1, (4, 4))
                
                # Inverse DCT
                idct_block = cv2.idct(dct_block)
                gray[i:i+8, j:j+8] = np.clip(idct_block, 0, 255)
        
        # Convert back to RGB
        stego_img = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return Image.fromarray(stego_img)
    
    def generate_random_message(self, length=None):
        """Generate a random message for steganography"""
        if length is None:
            length = random.randint(10, 100)
        
        # Create random message with various character types
        chars = string.ascii_letters + string.digits + ' .,!?'
        message = ''.join(random.choice(chars) for _ in range(length))
        
        return message
    
    def create_dataset_batch(self, batch_size, stego_ratio=0.5):
        """Create a batch of images for training"""
        images = []
        labels = []
        
        num_stego = int(batch_size * stego_ratio)
        num_clean = batch_size - num_stego
        
        # Generate clean images
        for _ in range(num_clean):
            clean_img = self.generate_clean_image()
            images.append(np.array(clean_img))
            labels.append(0)  # 0 for clean
        
        # Generate steganographic images
        for _ in range(num_stego):
            base_img = self.generate_clean_image()
            
            # Apply random steganography technique
            stego_method = random.choice(['lsb', 'noise', 'frequency'])
            
            if stego_method == 'lsb':
                stego_img = self.apply_lsb_steganography(base_img, self.generate_random_message())
            elif stego_method == 'noise':
                stego_img = self.apply_noise_based_stego(base_img)
            else:
                stego_img = self.apply_frequency_domain_stego(base_img)
            
            images.append(np.array(stego_img))
            labels.append(1)  # 1 for steganographic
        
        # Shuffle the batch
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images, labels = zip(*combined)
        
        return np.array(images), np.array(labels)
