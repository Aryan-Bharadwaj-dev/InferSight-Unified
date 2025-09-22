# Deepfake Detection - Integrated Application

This is an integrated deepfake detection application that combines the InferSight HTML frontend with a powerful FastAPI backend.

## Project Structure

```
deepfake_integrated/
├── backend/                    # FastAPI backend
│   ├── server.py              # Main FastAPI application
│   ├── models.py              # Pydantic models
│   ├── detection_service.py   # Detection orchestration
│   ├── deepfake_detector.py   # Core detection logic
│   ├── file_handler.py        # File management
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
├── static/                    # Frontend HTML files
│   ├── deepfake.html         # Main deepfake detection page
│   ├── styles.css            # Styling
│   ├── logo.png              # InferSight logo
│   └── other HTML pages...
├── uploads/                   # Temporary upload directory
├── start_server.bat          # Windows startup script
└── README.md                 # This file
```

## Features

- **HTML Frontend**: Clean, responsive UI with InferSight branding
- **FastAPI Backend**: Robust API with real-time progress tracking
- **File Upload**: Support for images and videos
- **Real-time Progress**: Live updates during analysis
- **MongoDB Integration**: Results storage and history
- **Static File Serving**: Integrated frontend and backend

## Quick Start

1. **Start the Server**:
   ```bash
   # On Windows
   start_server.bat
   
   # Or manually
   cd backend
   pip install -r requirements.txt
   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the Application**:
   Open your browser and go to: http://localhost:8000

## API Endpoints

- `GET /` - Serves the main deepfake detection page
- `GET /static/*` - Serves static files (CSS, images, etc.)
- `POST /api/upload` - Upload file for analysis
- `GET /api/analysis/{upload_id}` - Get analysis progress
- `GET /api/result/{upload_id}` - Get final detection result
- `GET /api/history` - Get recent detection results
- `GET /api/health` - Health check endpoint

## Environment Setup

Make sure your `.env` file in the `backend/` directory contains:
```
MONGO_URL=your_mongodb_connection_string
DB_NAME=your_database_name
```

## Dependencies

See `backend/requirements.txt` for the complete list of Python dependencies, including:
- FastAPI for the web framework
- PyTorch/TensorFlow for deep learning models
- OpenCV for image/video processing
- MongoDB Motor for database integration
- And many more...

## Usage

1. Choose between Image or Video detection
2. Upload your file
3. Watch the real-time progress indicator
4. View detailed detection results with confidence scores

The application supports both image and video analysis with multiple AI models for comprehensive deepfake detection.
