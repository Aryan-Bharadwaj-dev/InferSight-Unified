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

      let analysisResult;
      
      try {
        // Try to connect to API first
        const response = await fetch(`${API_BASE_URL}/api/stego/analyze?analysis_type=${analysisType}`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('API not available');
        }

        analysisResult = await response.json();
      } catch (apiError) {
        // If API fails, use mock data for demo purposes
        console.log('API not available, using mock data for demonstration');
        
        // Simulate realistic analysis time
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Generate mock analysis result based on file type
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const isImageFile = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff'].includes(fileExtension);
        const isAudioFile = ['wav', 'mp3', 'flac', 'ogg'].includes(fileExtension);
        const isVideoFile = ['mp4', 'avi', 'mov', 'mkv'].includes(fileExtension);
        
        const mockResults = {
          id: `demo-${Date.now()}`,
          filename: file.name,
          file_type: isImageFile ? 'image' : isAudioFile ? 'audio' : isVideoFile ? 'video' : 'document',
          analysis_type: analysisType,
          has_hidden_data: Math.random() > 0.7, // 30% chance of hidden data
          confidence_score: Math.random() * 0.4 + 0.6, // 60-100%
          detection_methods: [
            ...(isImageFile ? ['LSB Analysis', 'EXIF Metadata Check', 'Color Channel Analysis'] : []),
            ...(isAudioFile ? ['Spectral Analysis', 'Phase Coding Detection'] : []),
            ...(isVideoFile ? ['Frame Analysis', 'Temporal Anomaly Detection'] : []),
            'Statistical Analysis',
            'File Size Anomaly Check'
          ],
          detailed_results: {
            lsb_analysis: isImageFile ? {
              suspicious_pixels: Math.floor(Math.random() * 1000),
              pattern_detected: Math.random() > 0.8
            } : null,
            metadata_analysis: {
              suspicious_fields: Math.floor(Math.random() * 5),
              hidden_text_found: Math.random() > 0.9
            },
            statistical_analysis: {
              entropy_score: Math.random() * 2 + 6,
              anomaly_detected: Math.random() > 0.8
            }
          },
          processing_time: Math.random() * 3 + 1,
          timestamp: new Date().toISOString(),
          demo_mode: true
        };
        
        analysisResult = mockResults;
      }

      clearInterval(progressInterval);
      setProgress(100);
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
        <div className="space-y-6">
          {/* Demo Mode Banner */}
          {result.demo_mode && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-yellow-800">
                    Demo Mode Active
                  </h3>
                  <div className="mt-2 text-sm text-yellow-700">
                    <p>The backend API is not currently running, so this is simulated analysis data for demonstration purposes. Results are randomly generated and not based on actual file analysis.</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Main Result */}
          <div className={`border rounded-lg p-6 ${
            result.has_hidden_data 
              ? 'bg-red-50 border-red-200' 
              : 'bg-green-50 border-green-200'
          }`}>
            <div className="text-center space-y-4">
              <div className="flex items-center justify-center">
                <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                  result.has_hidden_data ? 'bg-red-100' : 'bg-green-100'
                }`}>
                  {result.has_hidden_data ? (
                    <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                  ) : (
                    <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  )}
                </div>
              </div>
              
              <div>
                <h2 className={`text-2xl font-bold ${
                  result.has_hidden_data ? 'text-red-600' : 'text-green-600'
                }`}>
                  {result.has_hidden_data ? 'Hidden Data Detected' : 'No Hidden Data Found'}
                </h2>
                <p className="text-lg text-gray-600 mt-2">
                  {(result.confidence_score * 100).toFixed(1)}% confidence
                </p>
              </div>
            </div>
          </div>

          {/* Analysis Details */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Analysis Details</h3>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">File Type:</span>
                  <span className="ml-2 font-medium capitalize">{result.file_type}</span>
                </div>
                <div>
                  <span className="text-gray-500">Processing Time:</span>
                  <span className="ml-2 font-medium">{result.processing_time.toFixed(2)}s</span>
                </div>
                <div>
                  <span className="text-gray-500">Analysis Type:</span>
                  <span className="ml-2 font-medium capitalize">{result.analysis_type}</span>
                </div>
              </div>
              
              <div>
                <span className="text-gray-500 text-sm">Detection Methods Used:</span>
                <div className="mt-1 flex flex-wrap gap-2">
                  {result.detection_methods.map((method, index) => (
                    <span key={index} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {method}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Detailed Analysis Results */}
          {result.detailed_results && (
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Detailed Analysis</h3>
              <div className="space-y-3">
                {result.detailed_results.lsb_analysis && (
                  <div className="text-sm">
                    <h4 className="font-medium text-gray-700">LSB Analysis</h4>
                    <p className="text-gray-600">Suspicious pixels found: {result.detailed_results.lsb_analysis.suspicious_pixels}</p>
                    <p className="text-gray-600">Pattern detected: {result.detailed_results.lsb_analysis.pattern_detected ? 'Yes' : 'No'}</p>
                  </div>
                )}
                
                {result.detailed_results.metadata_analysis && (
                  <div className="text-sm">
                    <h4 className="font-medium text-gray-700">Metadata Analysis</h4>
                    <p className="text-gray-600">Suspicious fields: {result.detailed_results.metadata_analysis.suspicious_fields}</p>
                    <p className="text-gray-600">Hidden text found: {result.detailed_results.metadata_analysis.hidden_text_found ? 'Yes' : 'No'}</p>
                  </div>
                )}
                
                {result.detailed_results.statistical_analysis && (
                  <div className="text-sm">
                    <h4 className="font-medium text-gray-700">Statistical Analysis</h4>
                    <p className="text-gray-600">Entropy score: {result.detailed_results.statistical_analysis.entropy_score.toFixed(2)}</p>
                    <p className="text-gray-600">Anomaly detected: {result.detailed_results.statistical_analysis.anomaly_detected ? 'Yes' : 'No'}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Reset Button */}
          <div className="flex justify-center">
            <button
              onClick={resetAnalysis}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 font-medium"
            >
              Analyze Another File
            </button>
          </div>
        </div>
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