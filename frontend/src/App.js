import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [detectedStars, setDetectedStars] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cameraScalingFactor, setCameraScalingFactor] = useState(18.18);
  const fileInputRef = useRef(null);

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setProcessedImage(null);
      setDetectedStars([]);
      setError(null);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setProcessedImage(null);
      setDetectedStars([]);
      setError(null);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const uploadImage = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);
    formData.append('camera_scaling_factor', cameraScalingFactor);
    formData.append('al_parameter', 1);

    try {
      const response = await fetch('http://localhost:5001/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setProcessedImage(data.processed_image);
        setDetectedStars(data.detected_stars);
      } else {
        setError(data.error || 'Failed to process image');
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const resetApp = () => {
    setSelectedImage(null);
    setProcessedImage(null);
    setDetectedStars([]);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="App">
      <div className="stars-background"></div>
      
      <header className="App-header">
        <h1 className="title">
          ⭐ Star Identifier ⭐
        </h1>
        <p className="subtitle">
          Upload an image of the night sky and identify the stars by name!
        </p>
      </header>

      <main className="main-content">
        {!selectedImage && !processedImage && (
          <div 
            className="upload-zone"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-content">
              <div className="upload-icon">🌌</div>
              <h3>Drop your star image here</h3>
              <p>or click to browse</p>
              <div className="supported-formats">
                Supports: JPG, PNG, GIF, BMP, TIFF
              </div>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              style={{ display: 'none' }}
            />
          </div>
        )}

        {selectedImage && !processedImage && (
          <div className="image-preview-section">
            <div className="image-container">
              <img
                src={URL.createObjectURL(selectedImage)}
                alt="Selected"
                className="preview-image"
              />
            </div>
            
            <div className="parameters-section">
              <h4>Detection Parameters</h4>
              <div className="parameter-input">
                <label htmlFor="cameraScaling">Camera Scaling Factor:</label>
                <input
                  id="cameraScaling"
                  type="number"
                  step="0.01"
                  value={cameraScalingFactor}
                  onChange={(e) => setCameraScalingFactor(parseFloat(e.target.value))}
                  className="scaling-input"
                />
                <small>Adjust based on your camera/telescope setup (default: 18.18)</small>
              </div>
            </div>
            
            <div className="action-buttons">
              <button 
                className="detect-button"
                onClick={uploadImage}
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="spinner"></span>
                    Identifying Stars...
                  </>
                ) : (
                  <>
                    🔍 Identify Stars
                  </>
                )}
              </button>
              <button className="reset-button" onClick={resetApp}>
                🔄 Choose Another Image
              </button>
            </div>
          </div>
        )}

        {processedImage && (
          <div className="results-section">
            <h2 className="results-title">✨ Stars Identified! ✨</h2>
            
            <div className="image-container">
              <img
                src={processedImage}
                alt="Processed with identified stars"
                className="processed-image"
              />
            </div>

            <div className="stars-list">
              <h3>Identified Stars:</h3>
              <div className="stars-grid">
                {detectedStars.map((star, index) => (
                  <div key={index} className="star-card">
                    <div className="star-icon">⭐</div>
                    <div className="star-name">{star.name}</div>
                    <div className="star-coords">
                      ({star.x}, {star.y})
                    </div>
                    <div className="star-confidence">
                      Confidence: {star.confidence}
                    </div>
                    <div className="star-hr">
                      HR: {star.hr}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="action-buttons">
              <button className="reset-button" onClick={resetApp}>
                🔄 Identify Another Image
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            <div className="error-icon">⚠️</div>
            <div className="error-text">{error}</div>
            <button className="retry-button" onClick={resetApp}>
              Try Again
            </button>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Made with ❤️ for astronomy enthusiasts</p>
      </footer>
    </div>
  );
}

export default App; 