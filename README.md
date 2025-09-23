# InferSight - Advanced Media Analysis Platform 🔍

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18+-61dafb.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**InferSight** is a comprehensive media analysis platform that combines advanced steganography detection with deepfake detection capabilities. This merged project integrates the best features from multiple repositories to provide a unified solution for detecting hidden data and manipulated media.

## 🌟 Features

### Steganography Detection
- **LSB (Least Significant Bit)** steganography detection
- **Metadata hiding** detection (EXIF data analysis)
- **File appending** detection (data hidden at end of files)
- **Statistical anomaly** detection
- **Color channel correlation** analysis
- **Frequency domain** analysis (DCT coefficients)
- **Suspicious file size** analysis
- **Palette anomaly** detection

### Deepfake Detection
- **AI-powered deepfake** detection for videos and images
- **Advanced neural network** models for media authenticity verification
- **Real-time processing** with progress tracking
- **Multiple format support** for various media types

### User Interface
- **Modern React.js** frontend with Tailwind CSS
- **Legacy HTML** interface (available in frontend/legacy)
- **Real-time analysis** progress tracking
- **Detailed results** with confidence scores
- **File upload** with drag-and-drop support

## 🏗️ Project Structure

```
InferSight-Merged/
├── backend/                    # Combined backend services
│   ├── api.py                 # FastAPI steganography detection API
│   ├── server.py              # Flask deepfake detection server
│   ├── stego_detector.py      # Steganography detection core
│   ├── deepfake_detector.py   # Deepfake detection core
│   ├── detectors/             # Detection algorithms
│   ├── models/                # ML models and weights
│   ├── training/              # Training scripts and data
│   ├── utils/                 # Utility functions
│   └── deepfake_integrated/   # Integrated deepfake detection
├── frontend/                  # React.js frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/            # Page components
│   │   └── App.js            # Main React application
│   ├── legacy/               # Original HTML/CSS/JS frontend
│   └── package.json          # Frontend dependencies
├── tests/                    # Test files
├── docs/                     # Documentation
└── requirements.txt          # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Quick Start (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aryan-Bharadwaj-dev/InferSight-Unified.git
   cd InferSight-Unified
   ```

2. **Start the complete development environment**
   ```bash
   ./start-dev.sh
   ```
   This will start both backend and frontend servers automatically!

### Manual Setup

If you prefer to start services individually:

#### Backend Setup
```bash
# Start the unified backend server
./start-backend.sh
```

#### Frontend Setup
```bash
# Start the React frontend
./start-frontend.sh
```

### Accessing the Application

- **🎨 React Frontend**: http://localhost:3000
- **🔍 Unified API**: http://localhost:8000
- **📊 API Documentation**: http://localhost:8000/docs
- **📄 Legacy Frontend**: Open `frontend/legacy/index.html` in your browser

### Features Access

**Steganography Detection:**
- Upload images, audio, video, or text files
- Choose analysis type (Quick, Comprehensive, Deep)
- Get detailed detection results with confidence scores

**Deepfake Detection:**
- Upload images or videos
- Get authenticity analysis with confidence scores
- View detailed detection metrics

## 🎯 Usage

### Steganography Detection
1. Navigate to the steganography detection section
2. Upload an image, audio, video, or text file
3. Select the analysis type
4. Review detailed detection results

### Deepfake Detection
1. Navigate to the deepfake detection section
2. Upload an image or video file
3. Wait for AI analysis to complete
4. Review authenticity results with confidence scores

## 🔧 API Endpoints

### Steganography Detection API (FastAPI)
- `POST /api/detect` - Analyze file for steganography
- `GET /api/health` - Health check
- `GET /docs` - Interactive API documentation

### Deepfake Detection API (Flask)
- `POST /api/detect-deepfake` - Analyze media for deepfake content
- `GET /api/status` - Server status

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 🛠️ Technologies Used

### Backend
- **FastAPI** - High-performance API framework
- **Flask** - Lightweight web framework
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing
- **Pillow** - Image processing
- **librosa** - Audio analysis

### Frontend
- **React.js** - Modern UI framework
- **Tailwind CSS** - Utility-first CSS framework
- **Vanilla JS/HTML/CSS** - Legacy interface

### AI/ML
- **TensorFlow/PyTorch** - Deep learning models
- **scikit-learn** - Machine learning utilities
- **Custom neural networks** - Specialized detection models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original steganography detection implementation by R-cyber-SRR
- Frontend by harsha-reddy03
- Deepfake detection system, enhancement and compilation by aryanbharadwaj

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.
