# InferSight - Advanced Steganography Detection Tool 🔍

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)

**InferSight** is an advanced steganography detection tool that combines traditional statistical analysis methods with modern deep learning techniques to detect hidden data in various file types including images, audio, video, and text files.

## Features

### Image Detection
- LSB (Least Significant Bit) steganography detection
- Metadata hiding detection (EXIF data analysis)
- File appending detection (data hidden at end of files)
- Statistical anomaly detection
- Color channel correlation analysis
- Frequency domain analysis (DCT coefficients)
- Suspicious file size analysis
- Palette anomaly detection

### Audio Detection
- LSB steganography in audio samples
- Spectral anomaly detection
- Noise pattern analysis
- Statistical analysis of audio properties

### Video Detection
- Frame anomaly detection
- Motion vector analysis
- Compression artifact analysis
- File structure analysis

### Text Detection
- Whitespace steganography detection
- Character distribution anomalies
- Word pattern analysis
- Unusual encoding detection

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### REST API (Recommended)

Start the FastAPI server:
```bash
python api.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

#### API Endpoints

- `POST /analyze` - Upload and analyze a file synchronously
- `POST /upload` - Upload a file for asynchronous analysis
- `GET /status/{file_id}` - Check analysis status
- `GET /results/{file_id}` - Get analysis results
- `GET /results/{file_id}/report` - Download JSON report
- `GET /health` - Health check
- `GET /supported-formats` - List supported file formats

#### Example API Usage (Python)

```python
import requests

# Analyze a file synchronously
with open('suspicious_image.jpg', 'rb') as f:
    files = {'file': ('suspicious_image.jpg', f, 'image/jpeg')}
    response = requests.post('http://localhost:8000/analyze', files=files)
    result = response.json()
    
print(f"Detections: {len(result['detections'])}")
for detection in result['detections']:
    print(f"- {detection['method']}: {detection['confidence']:.1f}% confidence")
```

### Command Line Interface

Analyze a single file:
```bash
python stego_detector.py path/to/file.jpg
```

Analyze a directory:
```bash
python stego_detector.py path/to/directory -r
```

Generate reports:
```bash
python stego_detector.py path/to/file.jpg -o reports/
```

### Command Line Options

- `-o, --output`: Output directory for reports
- `-r, --recursive`: Recursively scan directories
- `-v, --verbose`: Enable verbose logging

## Supported File Types

- **Images**: .jpg, .jpeg, .png, .bmp, .tiff, .gif
- **Audio**: .wav, .mp3, .flac, .ogg, .aac
- **Video**: .mp4, .avi, .mkv, .mov, .wmv
- **Text**: .txt, .md, .html, .xml

## 🧠 Machine Learning Training

InferSight includes advanced machine learning capabilities with a comprehensive training pipeline.

### Quick Training

```bash
# Train with default parameters (50 epochs, 2000 samples)
python training/train_models.py

# Custom training parameters
python training/train_models.py --epochs 100 --batch-size 64

# Create benchmark dataset for evaluation
python training/train_models.py --create-benchmark --benchmark-dir ./benchmark

# Evaluate trained model performance
python training/train_models.py --evaluate --benchmark-dir ./benchmark
```

### Training Features

- **Synthetic Data Generation**: Creates diverse training datasets with multiple steganography techniques
- **CNN Architecture**: Deep convolutional neural network for image analysis
- **Data Augmentation**: Various image transformations and steganography methods
- **Performance Evaluation**: Comprehensive testing on benchmark datasets
- **Model Export**: Saves trained models for real-time inference

### Supported Training Data Types

- **Clean Images**: Gradients, noise patterns, geometric shapes, textures
- **Steganographic Techniques**: LSB embedding, noise-based hiding, frequency domain modifications
- **Variable Payloads**: Different message lengths and content types
- **Multiple Formats**: PNG, JPEG simulation for training

## Detection Methods

### Image Steganography
1. **LSB Analysis**: Detects high entropy in least significant bits
2. **Metadata Analysis**: Checks EXIF data for suspicious content
3. **File Appending**: Looks for data appended after image end markers
4. **Statistical Tests**: Chi-square tests for pixel distribution anomalies
5. **Channel Correlation**: Analyzes correlation between color channels
6. **Frequency Domain**: DCT coefficient variance analysis
7. **File Size**: Compares actual vs expected file size
8. **Palette Analysis**: Checks color palette anomalies

### Audio Steganography
1. **LSB Analysis**: Entropy analysis of least significant bits
2. **Spectral Analysis**: FFT-based frequency domain analysis
3. **Noise Patterns**: Statistical analysis of noise characteristics
4. **Audio Properties**: Mean and standard deviation analysis

### Video Steganography
1. **Frame Analysis**: Inter-frame difference analysis
2. **Motion Vectors**: Optical flow consistency checks
3. **Compression**: DCT coefficient and quality analysis
4. **File Structure**: Size and metadata analysis

### Text Steganography
1. **Whitespace**: Unusual space, tab, and newline patterns
2. **Character Distribution**: Entropy and diversity analysis
3. **Word Patterns**: Frequency analysis of word usage
4. **Encoding**: Detection of unusual encoding markers

## Output

The tool provides:
- Confidence scores (0-100%) for each detection method
- Detailed descriptions of detected anomalies
- JSON reports (when output directory specified)
- Console output with summary information

## Example Output

```
[ALERT] Potential steganography detected in: image.jpg
  - LSB Steganography: 85.67% confidence
    High entropy in LSB planes: 0.987
  - Suspicious File Size: 72.30% confidence
    File size 2,547,821 bytes is 3.2x larger than expected
```

## Limitations

- Detection accuracy depends on the steganography method used
- Some methods may produce false positives with highly compressed or processed files
- Advanced steganography techniques may evade detection
- Audio analysis is currently limited to WAV files for full functionality

## Contributing

Feel free to contribute by:
- Adding new detection methods
- Improving existing algorithms
- Supporting additional file formats
- Enhancing the reporting system



