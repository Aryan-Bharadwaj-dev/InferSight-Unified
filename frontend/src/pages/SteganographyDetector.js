import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import AnalysisProgress from '../components/AnalysisProgress';
import DetectionResult from '../components/DetectionResult';

const API_BASE_URL = 'http://localhost:8000';

const SteganographyDetector = () => {
  const [file, setFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [analysisType, setAnalysisType] = useState('comprehensive');

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
  };

  const analyzeFile = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setProgress(0);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await fetch(`${API_BASE_URL}/api/stego/analyze?analysis_type=${analysisType}`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const analysisResult = await response.json();
      setResult(analysisResult);
    } catch (err) {
      setError(err.message);
      console.error('Error analyzing file:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
    setIsAnalyzing(false);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Steganography Detection
        </h2>
        <p className="text-gray-600">
          Detect hidden data in images, audio, video files, and documents using advanced analysis techniques.
        </p>
      </div>

      {!file && !result && (
        <div className="space-y-6">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-blue-800 mb-2">Analysis Types</h3>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  name="analysisType"
                  value="comprehensive"
                  checked={analysisType === 'comprehensive'}
                  onChange={(e) => setAnalysisType(e.target.value)}
                  className="text-blue-600"
                />
                <span className="text-sm">Comprehensive - Full analysis (LSB, metadata, statistical)</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  name="analysisType"
                  value="quick"
                  checked={analysisType === 'quick'}
                  onChange={(e) => setAnalysisType(e.target.value)}
                  className="text-blue-600"
                />
                <span className="text-sm">Quick - Fast analysis (basic techniques)</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  name="analysisType"
                  value="deep"
                  checked={analysisType === 'deep'}
                  onChange={(e) => setAnalysisType(e.target.value)}
                  className="text-blue-600"
                />
                <span className="text-sm">Deep - Advanced analysis (all methods + AI)</span>
              </label>
            </div>
          </div>
          <FileUpload
            onFileSelect={handleFileSelect}
            acceptedTypes=".jpg,.jpeg,.png,.bmp,.tiff,.gif,.wav,.mp3,.flac,.ogg,.mp4,.avi,.mov,.mkv,.txt,.docx,.pdf"
            maxSize={50 * 1024 * 1024} // 50MB
            title="Upload File for Steganography Analysis"
            description="Supported: Images, Audio, Video, and Text files"
          />
        </div>
      )}

      {file && !result && !isAnalyzing && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-800">
                Ready to Analyze
              </h3>
              <p className="text-gray-600">
                File: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
              </p>
              <p className="text-sm text-gray-500">
                Analysis Type: {analysisType}
              </p>
            </div>
            <div className="space-x-3">
              <button
                onClick={resetAnalysis}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={analyzeFile}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Start Analysis
              </button>
            </div>
          </div>
        </div>
      )}

      {isAnalyzing && (
        <AnalysisProgress
          progress={progress}
          currentStep={
            progress < 30 ? 'Uploading file...' :
            progress < 60 ? 'Performing steganographic analysis...' :
            progress < 90 ? 'Analyzing results...' :
            'Finalizing...'
          }
        />
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <div className="flex">
            <div className="text-red-600">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-red-800 font-medium">Analysis Failed</h3>
              <p className="text-red-700 mt-1">{error}</p>
            </div>
          </div>
          <button
            onClick={resetAnalysis}
            className="mt-3 px-4 py-2 bg-red-100 text-red-800 rounded hover:bg-red-200"
          >
            Try Again
          </button>
        </div>
      )}

      {result && (
        <DetectionResult
          result={{
            ...result,
            type: 'steganography'
          }}
          onReset={resetAnalysis}
        />
      )}

      <div className="mt-8 bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">
          Supported Analysis Methods
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Image Analysis</h4>
            <ul className="text-gray-600 space-y-1">
              <li>• LSB (Least Significant Bit) detection</li>
              <li>• EXIF metadata analysis</li>
              <li>• Color channel correlation</li>
              <li>• Statistical anomaly detection</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Audio/Video Analysis</h4>
            <ul className="text-gray-600 space-y-1">
              <li>• Spectral analysis</li>
              <li>• Phase coding detection</li>
              <li>• Echo hiding detection</li>
              <li>• Temporal analysis</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SteganographyDetector;