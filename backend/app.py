from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
import random
from helper_functions import detect_stars, get_star_catalog, load_spht_from_json, build_spht_offline, save_spht_to_json
from algorithms import stars_identification

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '../uploads'
PROCESSED_FOLDER = '../processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load star catalog and build SPHT exactly like the research project
print("Loading star catalog and building SPHT like research project...")
star_catalog = get_star_catalog()


# Try to load existing research SPHT, otherwise build it
# try:
#     spht = load_spht_from_json("spht.json")
#     print(f"Loaded SPHT with {len(spht)} entries")
# except:
print("Building reduced SPHT...")
reduced_star_catalog = random.sample(star_catalog, 200)
spht = build_spht_offline(reduced_star_catalog, 1)  # al_parameter = 1
save_spht_to_json(spht, "spht.json")
print(f"Built and saved SPHT with {len(spht)} entries")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_star_name_by_hr(hr_number):
    """Get star name from HR number using the star catalog"""
    for star in star_catalog:
        if star.get('HR') == hr_number:
            return star.get('N', f"HR {hr_number}")
    return f"HR {hr_number}"

def detect_and_identify_stars(image_path, camera_scaling_factor=18.18, al_parameter=1):
    """
    Complete star identification pipeline using exact research setup
    """
    try:
        # Step 1: Detect stars using computer vision
        print("Detecting stars in image...")
        detected_stars = detect_stars(image_path)
        
        if len(detected_stars) < 3:
            return None, [], "Need at least 3 stars for identification"
        
        print(f"Detected {len(detected_stars)} stars")
        
        # Step 2: Identify stars using SPHT and algorithms (exact research setup)
        print("Identifying stars using research SPHT...")
        identified_stars = stars_identification(
            detected_stars, 
            spht, 
            al_parameter, 
            camera_scaling_factor
        )
        
        print(f"Identified {len(identified_stars)} stars")
        
        # Step 3: Create visualization
        img = cv2.imread(image_path)
        if img is None:
            return None, [], "Could not read image"
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw circles and labels for identified stars
        for star_data in identified_stars:
            x, y = star_data['coords']
            hr_number = star_data['spht_value']
            confidence = star_data['confidence']
            
            # Get star name
            star_name = get_star_name_by_hr(hr_number)
            
            # Draw red circle around identified star
            radius = 15
            cv2.circle(img_rgb, (x, y), radius, (255, 0, 0), 2)
            
            # Add star name and confidence
            label = f"{star_name} ({confidence})"
            cv2.putText(img_rgb, label, (x-30, y-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return img_rgb, identified_stars, None
        
    except Exception as e:
        print(f"Error in star identification: {e}")
        return None, [], str(e)

@app.route('/')
def home():
    return jsonify({
        "message": "Star Detector API is running!",
        "endpoints": {
            "/upload": "POST - Upload an image for star identification",
            "/health": "GET - Health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "service": "star-detector-api",
        "stars_loaded": len(star_catalog),
        "subset_stars": len(subset_bsc),
        "spht_entries": len(spht)
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Use exact research parameters
        camera_scaling_factor = float(request.form.get('camera_scaling_factor', 18.18))
        al_parameter = float(request.form.get('al_parameter', 1))
        
        if file:
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Save the uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Process the image with research-exact star identification
            processed_image, identified_stars, error = detect_and_identify_stars(
                upload_path, camera_scaling_factor, al_parameter
            )
            
            if processed_image is None:
                return jsonify({'error': error or 'No stars identified in image'}), 500
            
            # Convert processed image to base64 for sending back to frontend
            pil_image = Image.fromarray(processed_image)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Format identified stars for frontend
            stars_list = []
            for star_data in identified_stars:
                x, y = star_data['coords']
                hr_number = star_data['spht_value']
                confidence = star_data['confidence']
                star_name = get_star_name_by_hr(hr_number)
                
                stars_list.append({
                    "name": star_name,
                    "x": x,
                    "y": y,
                    "hr": hr_number,
                    "confidence": confidence
                })
            
            return jsonify({
                'success': True,
                'message': f'Identified {len(identified_stars)} stars in image',
                'processed_image': f'data:image/png;base64,{img_base64}',
                'detected_stars': stars_list,
                'parameters': {
                    'camera_scaling_factor': camera_scaling_factor,
                    'al_parameter': al_parameter
                }
            })
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 