"""
Image Steganography Detector
Detects various forms of steganography in image files
"""

import numpy as np
from PIL import Image, ExifTags
import cv2
import os
from pathlib import Path
import hashlib
import struct

# Try to import ML model
try:
    from models.image_classifier import ImageStegoClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class ImageStegoDetector:
    def __init__(self, model_path=None):
        self.detection_methods = [
            self._detect_lsb_steganography,
            self._detect_metadata_hiding,
            self._detect_file_appending,
            self._detect_statistical_anomalies,
            self._detect_color_channel_anomalies,
            self._detect_frequency_domain_hiding,
            self._check_suspicious_file_size,
            self._detect_palette_anomalies
        ]
        
        # Add ML detection if available
        if ML_AVAILABLE:
            self.detection_methods.append(self._ml_detection)
            try:
                self.ml_classifier = ImageStegoClassifier(model_path)
            except Exception:
                self.ml_classifier = None
        else:
            self.ml_classifier = None
    
    def detect(self, image_path):
        """Run all detection methods on an image"""
        detections = []
        
        try:
            # Load image with PIL
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    img = img.convert('RGB')
                
                # Run each detection method
                for method in self.detection_methods:
                    try:
                        result = method(img, image_path)
                        if result:
                            detections.append(result)
                    except Exception as e:
                        print(f"Error in detection method {method.__name__}: {e}")
                        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            
        return detections
    
    def _detect_lsb_steganography(self, img, image_path):
        """Detect LSB (Least Significant Bit) steganography"""
        try:
            img_array = np.array(img)
            
            # Check if image has multiple channels
            if len(img_array.shape) < 3:
                return None
                
            # Extract LSBs from each channel
            lsb_r = img_array[:, :, 0] & 1
            lsb_g = img_array[:, :, 1] & 1
            lsb_b = img_array[:, :, 2] & 1
            
            # Calculate entropy of LSB planes
            def calculate_entropy(data):
                unique, counts = np.unique(data, return_counts=True)
                probabilities = counts / counts.sum()
                entropy = -sum(probabilities * np.log2(probabilities))
                return entropy
            
            entropy_r = calculate_entropy(lsb_r)
            entropy_g = calculate_entropy(lsb_g)
            entropy_b = calculate_entropy(lsb_b)
            
            avg_entropy = (entropy_r + entropy_g + entropy_b) / 3
            
            # High entropy in LSB planes suggests steganography
            if avg_entropy > 0.9:
                confidence = min(100, (avg_entropy - 0.9) * 1000)
                return {
                    'method': 'LSB Steganography',
                    'confidence': confidence,
                    'description': f'High entropy in LSB planes: {avg_entropy:.3f}',
                    'details': {
                        'entropy_r': entropy_r,
                        'entropy_g': entropy_g,
                        'entropy_b': entropy_b
                    }
                }
        except Exception:
            pass
        return None
    
    def _detect_metadata_hiding(self, img, image_path):
        """Detect hidden data in EXIF metadata"""
        try:
            suspicious_tags = []
            
            # Check EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                exif_data = img._getexif()
                
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    
                    # Check for suspicious long strings or binary data
                    if isinstance(value, str):
                        if len(value) > 100:
                            suspicious_tags.append(f"{tag}: Long string ({len(value)} chars)")
                        elif any(ord(c) > 127 for c in value):
                            suspicious_tags.append(f"{tag}: Contains non-ASCII characters")
                    elif isinstance(value, bytes) and len(value) > 50:
                        suspicious_tags.append(f"{tag}: Large binary data ({len(value)} bytes)")
            
            # Check for suspicious comment fields
            if hasattr(img, 'info'):
                for key, value in img.info.items():
                    if key.lower() in ['comment', 'description', 'software']:
                        if isinstance(value, (str, bytes)) and len(str(value)) > 100:
                            suspicious_tags.append(f"{key}: Unusually long ({len(str(value))} chars)")
            
            if suspicious_tags:
                return {
                    'method': 'Metadata Hiding',
                    'confidence': min(85, len(suspicious_tags) * 30),
                    'description': 'Suspicious metadata found',
                    'details': {'suspicious_tags': suspicious_tags[:5]}  # Limit output
                }
        except Exception:
            pass
        return None
    
    def _detect_file_appending(self, img, image_path):
        """Detect data appended to end of image file"""
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
            
            # Look for common file signatures after the image data
            suspicious_signatures = [
                b'PK',      # ZIP file
                b'\x7fELF', # ELF executable
                b'MZ',      # Windows executable
                b'\x89PNG', # PNG file
                b'\xff\xd8\xff', # JPEG file
                b'GIF8',    # GIF file
                b'Rar!',    # RAR archive
            ]
            
            # Find the actual end of image data
            if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                # JPEG ends with FF D9
                jpeg_end = content.rfind(b'\xff\xd9')
                if jpeg_end != -1:
                    trailing_data = content[jpeg_end + 2:]
                    if len(trailing_data) > 10:
                        for sig in suspicious_signatures:
                            if sig in trailing_data:
                                return {
                                    'method': 'File Appending',
                                    'confidence': 90,
                                    'description': f'Suspicious data found after JPEG end marker ({len(trailing_data)} bytes)',
                                    'details': {'trailing_bytes': len(trailing_data)}
                                }
            
            elif image_path.suffix.lower() == '.png':
                # PNG ends with IEND chunk
                iend_pos = content.rfind(b'IEND')
                if iend_pos != -1:
                    # IEND chunk is 12 bytes total
                    trailing_data = content[iend_pos + 12:]
                    if len(trailing_data) > 10:
                        for sig in suspicious_signatures:
                            if sig in trailing_data:
                                return {
                                    'method': 'File Appending',
                                    'confidence': 90,
                                    'description': f'Suspicious data found after PNG IEND chunk ({len(trailing_data)} bytes)',
                                    'details': {'trailing_bytes': len(trailing_data)}
                                }
        except Exception:
            pass
        return None
    
    def _detect_statistical_anomalies(self, img, image_path):
        """Detect statistical anomalies that might indicate steganography"""
        try:
            img_array = np.array(img)
            
            if len(img_array.shape) < 3:
                return None
            
            # Calculate chi-square test for each color channel
            def chi_square_test(channel):
                hist, _ = np.histogram(channel, bins=256, range=(0, 256))
                expected = np.mean(hist)
                chi_sq = np.sum((hist - expected) ** 2 / expected) if expected > 0 else 0
                return chi_sq
            
            chi_sq_values = []
            for i in range(3):  # RGB channels
                chi_sq = chi_square_test(img_array[:, :, i])
                chi_sq_values.append(chi_sq)
            
            avg_chi_sq = np.mean(chi_sq_values)
            
            # Unusually high chi-square values might indicate steganography
            if avg_chi_sq > 500:
                confidence = min(75, (avg_chi_sq - 500) / 20)
                return {
                    'method': 'Statistical Anomaly',
                    'confidence': confidence,
                    'description': f'Unusual pixel distribution detected (χ² = {avg_chi_sq:.2f})',
                    'details': {'chi_square_values': chi_sq_values}
                }
        except Exception:
            pass
        return None
    
    def _detect_color_channel_anomalies(self, img, image_path):
        """Detect anomalies in color channel correlations"""
        try:
            img_array = np.array(img)
            
            if len(img_array.shape) < 3:
                return None
            
            # Calculate correlation between color channels
            r_channel = img_array[:, :, 0].flatten()
            g_channel = img_array[:, :, 1].flatten()
            b_channel = img_array[:, :, 2].flatten()
            
            corr_rg = np.corrcoef(r_channel, g_channel)[0, 1]
            corr_rb = np.corrcoef(r_channel, b_channel)[0, 1]
            corr_gb = np.corrcoef(g_channel, b_channel)[0, 1]
            
            avg_correlation = (abs(corr_rg) + abs(corr_rb) + abs(corr_gb)) / 3
            
            # Very low correlation might indicate independent data in channels
            if avg_correlation < 0.3:
                confidence = (0.3 - avg_correlation) * 200
                return {
                    'method': 'Color Channel Anomaly',
                    'confidence': min(70, confidence),
                    'description': f'Unusual color channel correlation: {avg_correlation:.3f}',
                    'details': {
                        'corr_rg': corr_rg,
                        'corr_rb': corr_rb,
                        'corr_gb': corr_gb
                    }
                }
        except Exception:
            pass
        return None
    
    def _detect_frequency_domain_hiding(self, img, image_path):
        """Detect steganography in frequency domain (DCT coefficients)"""
        try:
            # Convert to grayscale for DCT analysis
            if img.mode != 'L':
                gray_img = img.convert('L')
            else:
                gray_img = img
                
            img_array = np.array(gray_img, dtype=np.float32)
            
            # Apply DCT to 8x8 blocks
            h, w = img_array.shape
            dct_coeffs = []
            
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = img_array[i:i+8, j:j+8]
                    dct_block = cv2.dct(block)
                    # Focus on mid-frequency coefficients
                    mid_freq = dct_block[2:6, 2:6].flatten()
                    dct_coeffs.extend(mid_freq)
            
            if dct_coeffs:
                # Calculate variance in DCT coefficients
                dct_var = np.var(dct_coeffs)
                
                # Unusually high variance might indicate hidden data
                if dct_var > 100:
                    confidence = min(60, (dct_var - 100) / 10)
                    return {
                        'method': 'Frequency Domain Anomaly',
                        'confidence': confidence,
                        'description': f'High variance in DCT coefficients: {dct_var:.2f}',
                        'details': {'dct_variance': dct_var}
                    }
        except Exception:
            pass
        return None
    
    def _check_suspicious_file_size(self, img, image_path):
        """Check for suspicious file size compared to image dimensions"""
        try:
            file_size = os.path.getsize(image_path)
            width, height = img.size
            
            # Estimate expected file size based on dimensions and format
            if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                # JPEG compression typically results in 1-10 bytes per pixel
                expected_size_range = (width * height * 1, width * height * 10)
            elif image_path.suffix.lower() == '.png':
                # PNG is usually larger than JPEG but compressed
                expected_size_range = (width * height * 1, width * height * 4)
            else:
                return None
            
            # Check if file is significantly larger than expected
            if file_size > expected_size_range[1] * 2:
                size_ratio = file_size / expected_size_range[1]
                confidence = min(80, (size_ratio - 2) * 20)
                return {
                    'method': 'Suspicious File Size',
                    'confidence': confidence,
                    'description': f'File size {file_size} bytes is {size_ratio:.1f}x larger than expected',
                    'details': {
                        'actual_size': file_size,
                        'expected_max': expected_size_range[1],
                        'ratio': size_ratio
                    }
                }
        except Exception:
            pass
        return None
    
    def _detect_palette_anomalies(self, img, image_path):
        """Detect anomalies in color palette for indexed color images"""
        try:
            if img.mode == 'P':  # Palette mode
                palette = img.getpalette()
                if palette:
                    # Convert palette to RGB triplets
                    rgb_palette = [palette[i:i+3] for i in range(0, len(palette), 3)]
                    
                    # Check for unusual palette characteristics
                    unique_colors = len(set(tuple(color) for color in rgb_palette if color != [0, 0, 0]))
                    
                    # Very few colors in palette might indicate steganography
                    if unique_colors < 4 and len(rgb_palette) > 50:
                        confidence = 60
                        return {
                            'method': 'Palette Anomaly',
                            'confidence': confidence,
                            'description': f'Suspicious palette: {unique_colors} unique colors out of {len(rgb_palette)}',
                            'details': {
                                'unique_colors': unique_colors,
                                'palette_size': len(rgb_palette)
                            }
                        }
        except Exception:
            pass
        return None
    
    def _ml_detection(self, img, image_path):
        """Use machine learning model for steganography detection"""
        if not self.ml_classifier:
            return None
            
        try:
            result = self.ml_classifier.predict(image_path)
            
            if result.get('error'):
                return None
                
            # Only report if ML model is confident about steganography
            if result['prediction'] == 'steganography' and result['confidence'] > 70:
                return {
                    'method': 'Deep Learning Detection',
                    'confidence': result['confidence'],
                    'description': f'Neural network detected steganography with {result["confidence"]:.1f}% confidence',
                    'details': {
                        'stego_probability': result['stego_probability'],
                        'clean_probability': result['clean_probability']
                    }
                }
        except Exception:
            pass
        return None
