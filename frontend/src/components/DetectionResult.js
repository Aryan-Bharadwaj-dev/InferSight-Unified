import React from 'react';

const DetectionResult = ({ result }) => {
  if (!result) return null;

  const getResultColor = (isDeepfake, confidence) => {
    if (isDeepfake) {
      return confidence > 0.8 ? 'text-red-600' : 'text-orange-600';
    } else {
      return confidence > 0.8 ? 'text-green-600' : 'text-yellow-600';
    }
  };

  const getResultBg = (isDeepfake, confidence) => {
    if (isDeepfake) {
      return confidence > 0.8 ? 'bg-red-50 border-red-200' : 'bg-orange-50 border-orange-200';
    } else {
      return confidence > 0.8 ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200';
    }
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence > 0.9) return 'Very High';
    if (confidence > 0.8) return 'High';
    if (confidence > 0.6) return 'Medium';
    return 'Low';
  };

  const formatAnalysisDetails = (analysis) => {
    if (!analysis) return null;
    
    const details = [];
    
    if (analysis.faces_detected !== undefined) {
      details.push(`Faces detected: ${analysis.faces_detected}`);
    }
    if (analysis.sharpness !== undefined) {
      details.push(`Image sharpness: ${analysis.sharpness.toFixed(2)}`);
    }
    if (analysis.frames_analyzed !== undefined) {
      details.push(`Frames analyzed: ${analysis.frames_analyzed}`);
    }
    if (analysis.spectral_centroid_mean !== undefined) {
      details.push(`Audio spectral analysis completed`);
    }
    
    return details;
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      {/* Main Result */}
      <div className={`border rounded-lg p-6 ${getResultBg(result.is_deepfake, result.confidence_score)}`}>
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center">
            <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
              result.is_deepfake ? 'bg-red-100' : 'bg-green-100'
            }`}>
              {result.is_deepfake ? (
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
            <h2 className={`text-2xl font-bold ${getResultColor(result.is_deepfake, result.confidence_score)}`}>
              {result.is_deepfake ? 'Deepfake Detected' : 'Authentic Content'}
            </h2>
            <p className="text-lg text-gray-600 mt-2">
              {result.deepfake_probability.toFixed(1)}% confidence
            </p>
            <p className="text-sm text-gray-500">
              Confidence Level: {getConfidenceLabel(result.confidence_score)}
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
              <span className="ml-2 font-medium">{result.file_type}</span>
            </div>
            <div>
              <span className="text-gray-500">Processing Time:</span>
              <span className="ml-2 font-medium">{result.processing_time.toFixed(2)}s</span>
            </div>
          </div>
          
          <div>
            <span className="text-gray-500 text-sm">Models Used:</span>
            <div className="mt-1 flex flex-wrap gap-2">
              {result.models_used.map((model, index) => (
                <span key={index} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  {model}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Visual Analysis */}
      {result.visual_analysis && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Visual Analysis</h3>
          <div className="space-y-2 text-sm">
            {formatAnalysisDetails(result.visual_analysis)?.map((detail, index) => (
              <p key={index} className="text-gray-600">• {detail}</p>
            ))}
          </div>
        </div>
      )}

      {/* Audio Analysis */}
      {result.audio_analysis && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Audio Analysis</h3>
          <div className="space-y-2 text-sm">
            {formatAnalysisDetails(result.audio_analysis)?.map((detail, index) => (
              <p key={index} className="text-gray-600">• {detail}</p>
            ))}
          </div>
        </div>
      )}

      {/* Temporal Analysis */}
      {result.temporal_analysis && Object.keys(result.temporal_analysis).length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Temporal Analysis</h3>
          <div className="space-y-2 text-sm">
            {formatAnalysisDetails(result.temporal_analysis)?.map((detail, index) => (
              <p key={index} className="text-gray-600">• {detail}</p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DetectionResult;