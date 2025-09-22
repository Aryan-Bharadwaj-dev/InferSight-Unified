import React, { useState } from "react";
import "./App.css";
import DeepfakeDetector from "./pages/DeepfakeDetector";
import SteganographyDetector from "./pages/SteganographyDetector";

function App() {
  const [activeTab, setActiveTab] = useState('steganography');

  return (
    <div className="App">
      <div className="bg-gray-100 min-h-screen">
        <div className="container mx-auto px-4 py-8">
          <header className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">
              🔍 InferSight Unified
            </h1>
            <p className="text-gray-600 text-lg">
              Advanced Media Analysis Platform - Steganography & Deepfake Detection
            </p>
          </header>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex border-b mb-6">
              <button
                className={`px-6 py-3 font-medium transition-colors ${
                  activeTab === 'steganography'
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => setActiveTab('steganography')}
              >
                📄 Steganography Detection
              </button>
              <button
                className={`px-6 py-3 font-medium transition-colors ml-4 ${
                  activeTab === 'deepfake'
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => setActiveTab('deepfake')}
              >
                🎭 Deepfake Detection
              </button>
            </div>

            {activeTab === 'steganography' && <SteganographyDetector />}
            {activeTab === 'deepfake' && <DeepfakeDetector />}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
