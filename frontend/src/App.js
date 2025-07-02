import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// Backend Configuration - Change these URLs to point to your backend
const BACKEND_CONFIG = 'http://localhost:5001';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [detectedStars, setDetectedStars] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cameraScalingFactor, setCameraScalingFactor] = useState(18.18);
  const [availableSphts, setAvailableSphts] = useState([]);
  const [selectedSpht, setSelectedSpht] = useState('default');
  const [showSphtManager, setShowSphtManager] = useState(false);
  const [newSphtAl, setNewSphtAl] = useState(2.0);
  const [newSphtName, setNewSphtName] = useState('');
  const [isGeneratingSpht, setIsGeneratingSpht] = useState(false);
  const [sphtGenerationProgress, setSphtGenerationProgress] = useState('');
  const [catalogSize, setCatalogSize] = useState(200);
  const [maxCatalogSize, setMaxCatalogSize] = useState(1000);
  const fileInputRef = useRef(null);

  // Load available SPHTs on component mount
  useEffect(() => {
    loadAvailableSphts();
  }, []);

  const loadAvailableSphts = async () => {
    try {
      const response = await fetch(`${BACKEND_CONFIG}/list-sphts`);
      const data = await response.json();
      
      if (data.success) {
        setAvailableSphts(data.sphts);
        if (data.max_catalog_size) {
          setMaxCatalogSize(data.max_catalog_size);
        }
        
        // Handle selectedSpht state
        if (data.sphts.length === 0) {
          setSelectedSpht(''); // No SPHTs available
        } else {
          // Check if currently selected SPHT still exists
          const sphtNames = data.sphts.map(spht => spht.name);
          if (!sphtNames.includes(selectedSpht)) {
            // Current selection no longer exists, select the first available
            setSelectedSpht(sphtNames[0]);
          }
        }
      } else {
        console.error('Failed to load SPHTs:', data.error);
      }
    } catch (err) {
      console.error('Failed to connect to server for SPHT list');
    }
  };

  const generateSpht = async () => {
    if (newSphtAl <= 0) {
      setError('AL parameter must be positive');
      return;
    }

    if (catalogSize <= 0 || catalogSize > maxCatalogSize) {
      setError(`Catalog size must be between 1 and ${maxCatalogSize}`);
      return;
    }

    setIsGeneratingSpht(true);
    setSphtGenerationProgress(`Generating SPHT using ${catalogSize} brightest stars... This may take a few minutes.`);
    setError(null);

    try {
      const requestData = {
        al_parameter: newSphtAl,
        catalog_size: catalogSize
      };
      
      if (newSphtName.trim()) {
        requestData.name = newSphtName.trim();
      }

      const response = await fetch(`${BACKEND_CONFIG}/generate-spht`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      const data = await response.json();

      if (data.success) {
        setSphtGenerationProgress(`✅ SPHT "${data.spht_name}" generated successfully! Used ${data.catalog_size} stars, created ${data.entries} entries`);
        setNewSphtAl(2.0);
        setNewSphtName('');
        setCatalogSize(200);
        await loadAvailableSphts(); // Refresh the list
        
        // Auto-hide the success message after 5 seconds
        setTimeout(() => {
          setSphtGenerationProgress('');
        }, 5000);
      } else {
        setError(data.error || 'Failed to generate SPHT');
        setSphtGenerationProgress('');
      }
    } catch (err) {
      setError('Failed to connect to server. Make sure the backend is running.');
      setSphtGenerationProgress('');
    } finally {
      setIsGeneratingSpht(false);
    }
  };

  const deleteSpht = async (sphtName) => {
    if (sphtName === 'default') {
      setError('Cannot delete default SPHT');
      return;
    }

    if (!window.confirm(`Are you sure you want to delete SPHT "${sphtName}"?`)) {
      return;
    }

    try {
      const response = await fetch(`${BACKEND_CONFIG}/delete-spht`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: sphtName }),
      });

      const data = await response.json();

      if (data.success) {
        await loadAvailableSphts(); // Refresh the list
        if (selectedSpht === sphtName) {
          setSelectedSpht('default'); // Reset to default if deleted SPHT was selected
        }
      } else {
        setError(data.error || 'Failed to delete SPHT');
      }
    } catch (err) {
      setError('Failed to connect to server');
    }
  };

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
    formData.append('spht_name', selectedSpht);

    // Include AL parameter if using a custom SPHT
    const selectedSphtInfo = availableSphts.find(spht => spht.name === selectedSpht);
    if (selectedSphtInfo) {
      formData.append('al_parameter', selectedSphtInfo.al_parameter);
    }

    try {
      const response = await fetch(`${BACKEND_CONFIG}/upload`, {
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
        
        <div className="header-actions">
          <button 
            className="spht-manager-toggle"
            onClick={() => setShowSphtManager(!showSphtManager)}
          >
            {showSphtManager ? '🔧 Hide SPHT Manager' : '🔧 Manage SPHTs'}
          </button>
        </div>
      </header>

      {/* SPHT Manager Panel */}
      {showSphtManager && (
        <div className="spht-manager">
          <h2>🔬 SPHT Manager</h2>
          <p className="spht-explanation">
            SPHT is the algorithm's database for star pattern matching. 
            The AL (Accuracy Level) parameter controls precision: higher values = more precise matching, lower values = more general matching.
          </p>
          
          {/* Generate New SPHT */}
          <div className="spht-generator">
            <h3>Generate New SPHT</h3>
            <div className="spht-form">
              <div className="form-row">
                <label>
                  AL Parameter:
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="10"
                    value={newSphtAl}
                    onChange={(e) => setNewSphtAl(parseFloat(e.target.value))}
                    disabled={isGeneratingSpht}
                  />
                </label>
                <label>
                  Custom Name (optional):
                  <input
                    type="text"
                    value={newSphtName}
                    onChange={(e) => setNewSphtName(e.target.value)}
                    placeholder={`AL_${newSphtAl}`}
                    disabled={isGeneratingSpht}
                  />
                </label>
              </div>
              <div className="form-row">
                <label>
                  Catalog Size:
                  <div className="slider-container">
                    <input
                      type="range"
                      min="50"
                      max={maxCatalogSize}
                      step="10"
                      value={catalogSize}
                      onChange={(e) => setCatalogSize(parseInt(e.target.value))}
                      disabled={isGeneratingSpht}
                      className="catalog-slider"
                    />
                    <div className="slider-labels">
                      <span>50 (fastest)</span>
                      <span className="current-value">{catalogSize} stars</span>
                      <span>{maxCatalogSize} (most accurate)</span>
                    </div>
                  </div>
                </label>
              </div>
              <div className="catalog-explanation">
                <p>💡 Using the <strong>{catalogSize}</strong> brightest stars from the catalog. More stars = better accuracy but slower generation.</p>
              </div>
              <button 
                className="generate-spht-button"
                onClick={generateSpht}
                disabled={isGeneratingSpht}
              >
                {isGeneratingSpht ? '⏳ Generating...' : '🚀 Generate SPHT'}
              </button>
            </div>
            
            {sphtGenerationProgress && (
              <div className="spht-progress">
                {sphtGenerationProgress}
              </div>
            )}
          </div>

          {/* Available SPHTs List */}
          <div className="spht-list">
            <h3>Available SPHTs ({availableSphts.length})</h3>
            {availableSphts.length === 0 ? (
              <div className="no-sphts-message">
                <p>🔍 No SPHT algorithms found.</p>
                <p>Generate your first SPHT above to start identifying stars!</p>
              </div>
            ) : (
              <div className="sphts-grid">
                {availableSphts.map((spht) => (
                  <div key={spht.name} className={`spht-card ${spht.name === selectedSpht ? 'selected' : ''}`}>
                    <div className="spht-header">
                      <h4>{spht.name}</h4>
                      <span className={`spht-type ${spht.created}`}>{spht.created}</span>
                    </div>
                    <div className="spht-details">
                      <p>🎯 AL Parameter: {spht.al_parameter}</p>
                      <p>⭐ Catalog Size: {spht.catalog_size} stars</p>
                      <p>📊 Entries: {spht.entries.toLocaleString()}</p>
                    </div>
                    <div className="spht-actions">
                      <button
                        className={`select-spht ${spht.name === selectedSpht ? 'selected' : ''}`}
                        onClick={() => setSelectedSpht(spht.name)}
                      >
                        {spht.name === selectedSpht ? '✅ Selected' : '📌 Select'}
                      </button>
                      {spht.created === 'custom' && (
                        <button
                          className="delete-spht"
                          onClick={() => deleteSpht(spht.name)}
                        >
                          🗑️ Delete
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

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
            
            {availableSphts.length === 0 ? (
              <div className="no-spht-warning">
                <h4>⚠️ No SPHT Algorithms Available</h4>
                <p>You need to generate at least one SPHT algorithm before you can identify stars.</p>
                <button 
                  className="open-spht-manager"
                  onClick={() => setShowSphtManager(true)}
                >
                  🔧 Open SPHT Manager
                </button>
              </div>
            ) : (
              <>
                <div className="parameters-section">
                  <h4>Detection Parameters</h4>
                  
                  <div className="parameter-input">
                    <label htmlFor="sphtSelect">SPHT Algorithm:</label>
                    <select
                      id="sphtSelect"
                      value={selectedSpht}
                      onChange={(e) => setSelectedSpht(e.target.value)}
                      className="spht-select"
                    >
                      {availableSphts.map((spht) => (
                        <option key={spht.name} value={spht.name}>
                          {spht.name} (AL: {spht.al_parameter}, Stars: {spht.catalog_size}, Entries: {spht.entries.toLocaleString()})
                        </option>
                      ))}
                    </select>
                    <small>Choose which SPHT algorithm to use for star identification</small>
                  </div>
                  
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
              </>
            )}
          </div>
        )}

        {processedImage && (
          <div className="results-section">
            <h2 className="results-title">✨ Stars Identified! ✨</h2>
            
            <div className="results-info">
              <p>Using SPHT: <strong>{selectedSpht}</strong></p>
              {availableSphts.find(s => s.name === selectedSpht) && (
                <p>AL Parameter: <strong>{availableSphts.find(s => s.name === selectedSpht).al_parameter}</strong></p>
              )}
            </div>
            
            <div className="image-container">
              <img
                src={processedImage}
                alt="Processed with identified stars"
                className="processed-image"
              />
            </div>

            <div className="stars-list">
              <h3>Identified Stars ({detectedStars.length}):</h3>
              
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
                🔄 Analyze Another Image
              </button>
            </div>
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