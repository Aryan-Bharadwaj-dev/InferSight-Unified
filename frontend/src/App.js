import React from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import DeepfakeDetector from "./pages/DeepfakeDetector";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<DeepfakeDetector />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;