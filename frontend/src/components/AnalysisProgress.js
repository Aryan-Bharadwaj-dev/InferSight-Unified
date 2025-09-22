import React from 'react';

const AnalysisProgress = ({ progress, isComplete, error }) => {
  if (error) {
    return (
      <div className="w-full max-w-2xl mx-auto bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Analysis Failed</h3>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!progress) return null;

  const progressPercentage = Math.round(progress.progress * 100);

  return (
    <div className="w-full max-w-2xl mx-auto bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900">
          {isComplete ? 'Analysis Complete' : 'Analyzing...'}
        </h3>
        <span className="text-sm text-gray-500">{progressPercentage}%</span>
      </div>
      
      <div className="mb-4">
        <div className="bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-300 ${
              isComplete ? 'bg-green-500' : 'bg-blue-500'
            }`}
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
      </div>
      
      <div className="flex items-center">
        {!isComplete && (
          <div className="flex-shrink-0 mr-3">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
          </div>
        )}
        <p className="text-sm text-gray-600">
          {progress.current_step || 'Processing...'}
        </p>
      </div>
      
      {progress.estimated_time_remaining && (
        <p className="text-xs text-gray-500 mt-2">
          Estimated time remaining: {Math.round(progress.estimated_time_remaining)}s
        </p>
      )}
    </div>
  );
};

export default AnalysisProgress;