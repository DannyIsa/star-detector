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

# Exact same setup as research project
ursa_major_stars = [
    { "B": "μ", "N": "Tania Australis", "C": "UMa", "Dec": "+41° 29′ 58″", "F": "34", "HR": "4069", "K": "3500", "RA": "10h 22m 19.7s", "V": "3.05" },
    { "B": "λ", "N": "Tania Borealis", "C": "UMa", "Dec": "+42° 54′ 52″", "F": "33", "HR": "4033", "K": "9500", "RA": "10h 17m 05.8s", "V": "3.45" },
    { "B": "θ", "N": "Sarir", "C": "UMa", "Dec": "+51° 40′ 38″", "F": "25", "HR": "3775", "K": "6600", "RA": "09h 32m 51.4s", "V": "3.17" },
    { "B": "β", "N": "Merak", "C": "UMa", "Dec": "+56° 22′ 57″", "F": "48", "HR": "4295", "K": "9750", "RA": "11h 01m 50.5s", "V": "2.37" },
    { "B": "ψ", "C": "UMa", "Dec": "+44° 29′ 55″", "F": "52", "HR": "4335", "K": "4850", "RA": "11h 09m 39.8s", "V": "3.01" },
    { "B": "χ", "N": "Al Kaphrah", "C": "UMa", "Dec": "+47° 46′ 46″", "F": "63", "HR": "4518", "K": "5000", "RA": "11h 46m 03.0s", "V": "3.71" },
    { "B": "γ", "N": "Phecda", "C": "UMa", "Dec": "+53° 41′ 41″", "F": "64", "HR": "4554", "K": "10000", "RA": "11h 53m 49.8s", "V": "2.44" },
    { "B": "δ", "N": "Megrez", "C": "UMa", "Dec": "+57° 01′ 57″", "F": "69", "HR": "4660", "K": "9250", "RA": "12h 15m 25.6s", "V": "3.31" },
    { "B": "ε", "N": "Alioth", "C": "UMa", "Dec": "+55° 57′ 35″", "F": "77", "HR": "4905", "K": "10000", "RA": "12h 54m 01.7s", "V": "1.77" },
    { "B": "α", "N": "Dubhe", "C": "UMa", "Dec": "+61° 45′ 03″", "F": "50", "HR": "4301", "K": "5000", "RA": "11h 03m 43.7s", "V": "1.79" },
    { "B": "υ", "C": "UMa", "Dec": "+59° 02′ 19″", "F": "29", "HR": "3888", "K": "7200", "RA": "09h 50m 59.4s", "V": "3.80" },
    { "C": "UMa", "Dec": "+63° 03′ 43″", "F": "23", "HR": "3757", "K": "7500", "RA": "09h 31m 31.7s", "V": "3.67" },
    { "B": "ο", "N": "Muscida", "C": "UMa", "Dec": "+60° 43′ 05″", "F": "1", "HR": "3323", "K": "5500", "RA": "08h 30m 15.9s", "V": "3.36" },
    { "B": "η", "N": "Alkaid", "C": "UMa", "Dec": "+49° 18′ 48″", "F": "85", "HR": "5191", "K": "24000", "RA": "13h 47m 32.4s", "V": "1.86" },
]

# Build subset exactly like research project
random.seed(42)
subset_bsc = random.sample(star_catalog, 5)  # Take exactly 5 random stars

# Add Ursa Major stars from the original BSC by HR value
hr_values = set(str(star["HR"]) for star in ursa_major_stars)
for star in star_catalog:
    if str(star.get("HR")) in hr_values and star not in subset_bsc:
        subset_bsc.append(star)

print(f"Using subset of {len(subset_bsc)} stars for SPHT")

# Try to load existing research SPHT, otherwise build it
try:
    spht = load_spht_from_json("spht_research.json")
    print(f"Loaded research SPHT with {len(spht)} entries")
except:
    print("Building SPHT from subset...")
    spht = build_spht_offline(subset_bsc, 1)  # al_parameter = 1
    save_spht_to_json(spht, "spht_research.json")
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