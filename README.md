# ⭐ Star Detector ⭐

A fun and interactive web application that detects and identifies stars in astronomical images. Built with React frontend and Python Flask backend.

## 🌟 Features

- **Beautiful Space-Themed UI**: Modern, animated interface with starry background
- **Drag & Drop Upload**: Easy image upload with drag-and-drop functionality
- **Image Processing**: Upload images and get them processed with star detection
- **Star Identification**: Mock star detection with named celestial objects
- **Responsive Design**: Works great on desktop and mobile devices
- **Real-time Feedback**: Loading states and error handling

## 🚀 Project Structure

```
star-ditector/
├── frontend/          # React application
│   ├── src/
│   │   ├── App.js     # Main React component
│   │   ├── App.css    # Space-themed styling
│   │   └── ...
│   └── package.json
├── backend/           # Python Flask API
│   ├── app.py         # Main Flask application
│   ├── requirements.txt
│   └── README.md
└── README.md
```

## 🛠️ Setup Instructions

### Backend (Python Flask)

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Flask server:
```bash
python app.py
```

The backend will start on `http://localhost:5000`

### Frontend (React)

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will start on `http://localhost:3000`

## 🎯 How to Use

1. **Start both servers** (backend on port 5000, frontend on port 3000)
2. **Open your browser** and go to `http://localhost:3000`
3. **Upload an image** by either:
   - Dragging and dropping an image file onto the upload zone
   - Clicking the upload zone to browse and select a file
4. **Click "Detect Stars"** to process the image
5. **View the results** with detected stars marked and identified

## 🔧 Technical Details

### Frontend Technologies
- React 19.1.0
- Modern CSS with animations and gradients
- Fetch API for backend communication

### Backend Technologies
- Flask 2.3.3
- OpenCV for image processing
- Pillow (PIL) for image manipulation
- Flask-CORS for cross-origin requests
- Base64 encoding for image transfer
