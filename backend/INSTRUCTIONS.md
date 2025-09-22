# Steganography Detection Tool - Setup Instructions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python api.py
```

### 3. Access the Application

**Frontend (Web Interface):**
- Open your browser to: http://localhost:8000/app

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🖥️ Frontend Features

### **Drag & Drop File Upload**
- Drag and drop images, videos, or audio files directly onto the upload area
- Supported formats: JPG, PNG, GIF, MP4, AVI, WAV, MP3, TXT, etc.
- Multiple file support with real-time progress tracking

### **Copy-Paste Text Analysis**
- Switch to the "Text Analysis" tab
- Paste text directly into the textarea
- Instant analysis with keyboard shortcut (Ctrl+Enter)
- Character count display

### **Real-time Results**
- Color-coded confidence levels (High/Medium/Low risk)
- Detailed detection method descriptions
- File information and analysis timestamps
- Clean and suspicious file indicators

## 🔧 API Endpoints

- `POST /analyze` - Upload and analyze files synchronously
- `POST /analyze-text` - Analyze text content for steganography
- `POST /upload` - Upload files for background analysis
- `GET /status/{file_id}` - Check analysis status
- `GET /results/{file_id}` - Get analysis results
- `GET /health` - API health check

## 📁 Project Structure

```
Infersight/
├── api.py                 # FastAPI backend server
├── frontend/              # Web interface
│   ├── index.html        # Main HTML page
│   ├── style.css         # Responsive CSS styles
│   └── script.js         # JavaScript functionality
├── detectors/             # Detection modules
│   ├── image_detector.py # Image analysis
│   ├── audio_detector.py # Audio analysis
│   ├── video_detector.py # Video analysis
│   └── text_detector.py  # Text analysis
├── utils/                 # Utility modules
└── requirements.txt       # Python dependencies
```

## 🎯 Usage Examples

### File Upload via Web Interface
1. Open http://localhost:8000/app
2. Drag files onto the upload area or click to browse
3. Watch real-time analysis progress
4. View detailed results with confidence scores

### Text Analysis via Web Interface
1. Click the "Text Analysis" tab
2. Paste or type text into the textarea
3. Click "Analyze Text" or press Ctrl+Enter
4. View instant results

### API Usage (Python)
```python
import requests

# Analyze a file
with open('suspicious_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/analyze', files=files)
    result = response.json()

# Analyze text
data = {'text_content': 'Your text here'}
response = requests.post('http://localhost:8000/analyze-text', data=data)
result = response.json()
```

## 🔍 Detection Methods

### Images
- LSB (Least Significant Bit) analysis
- Metadata hiding detection
- File appending detection
- Statistical anomaly detection
- Color channel correlation analysis
- Frequency domain analysis

### Audio
- LSB steganography detection
- Spectral anomaly detection
- Noise pattern analysis
- Statistical analysis

### Video
- Frame-by-frame analysis
- Motion vector anomalies
- Compression artifact detection
- File structure analysis

### Text
- Whitespace steganography
- Character distribution analysis
- Word pattern analysis
- Encoding anomaly detection

## 🛠️ Development

To run in development mode:
```bash
python start_api.py
```

To test the CLI version:
```bash
python test_detector.py
```

## 📋 Notes

- The frontend automatically checks API health on startup
- Files are temporarily stored during analysis and cleaned up afterward
- Results are stored in memory (use a database for production)
- CORS is enabled for development (configure for production)

## 🎨 Frontend Features

- **Responsive Design**: Works on desktop and mobile
- **Modern UI**: Clean, professional interface with animations
- **Real-time Feedback**: Progress bars, status indicators, notifications
- **Keyboard Shortcuts**: Ctrl+Enter to analyze text, Escape to clear results
- **File Validation**: Client-side format checking
- **Error Handling**: User-friendly error messages and warnings
