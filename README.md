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

## 🎨 Design Features

- **Animated starry background** with moving stars
- **Gradient text effects** with color-shifting animations
- **Glassmorphism design** with backdrop blur effects
- **Smooth hover animations** and transitions
- **Space-themed color palette** (blues, teals, purples)
- **Custom fonts**: Orbitron for headings, Space Mono for body text

## 🔧 Technical Details

### Frontend Technologies
- React 19.1.0
- Modern CSS with animations and gradients
- Fetch API for backend communication
- Responsive design with CSS Grid and Flexbox

### Backend Technologies
- Flask 2.3.3
- OpenCV for image processing
- Pillow (PIL) for image manipulation
- Flask-CORS for cross-origin requests
- Base64 encoding for image transfer

## 🌌 Future Enhancements

- [ ] Implement real star detection using computer vision algorithms
- [ ] Integrate with astronomical databases for accurate star identification
- [ ] Add constellation detection and mapping
- [ ] Implement image preprocessing (noise reduction, contrast enhancement)
- [ ] Add support for different image formats and sizes
- [ ] Create user accounts and save detection history
- [ ] Add educational content about detected stars
- [ ] Implement real-time star tracking
- [ ] Add augmented reality features

## 🤝 Contributing

This project is in early development. Future contributions will focus on:
- Implementing actual star detection algorithms
- Improving the UI/UX
- Adding more astronomical features
- Performance optimizations

## 📝 License

This project is open source and available under the MIT License.

## 🌟 Acknowledgments

- Inspired by the beauty of astronomy and stargazing
- Built with modern web technologies for an engaging user experience
- Designed to make astronomy accessible and fun for everyone 