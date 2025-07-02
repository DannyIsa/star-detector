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

# Constants
UPLOAD_FOLDER = '../uploads'
PROCESSED_FOLDER = '../processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
DEFAULT_CAMERA_SCALING_FACTOR = 18.18
DEFAULT_AL_PARAMETER = 1
REDUCED_CATALOG_SIZE = 200
MIN_STARS_FOR_IDENTIFICATION = 3
STAR_CIRCLE_RADIUS = 15
SPHT_FILENAME = "spht.json"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Global variables for star data
star_catalog = None
subset_bsc = None
spht = None

def create_reduced_catalog(main_catalog, subset_size=REDUCED_CATALOG_SIZE):
    print(f"Creating reduced catalog with {subset_size} stars...")
    return random.sample(main_catalog, subset_size)

def spht_file_exists():
    """Check if SPHT file already exists"""
    return os.path.exists(SPHT_FILENAME)

def initialize_star_data():
    """Initialize star catalog and build SPHT for star identification"""
    global star_catalog, subset_bsc, spht
    
    print("Initializing Star Detector API...")
    print("Loading main star catalog...")
    star_catalog = get_star_catalog()
    print(f"Loaded {len(star_catalog)} stars from main catalog")
    # star_catalog = create_reduced_catalog(star_catalog) # uncomment this to use a reduced catalog
    
    # Check if SPHT file already exists
    if spht_file_exists():
        print(f"SPHT file '{SPHT_FILENAME}' already exists. Loading from file...")
        spht = load_spht_from_json(SPHT_FILENAME)
        print(f"Loaded existing SPHT with {len(spht)} entries")
    else:
        print(f"SPHT file '{SPHT_FILENAME}' not found. Building new SPHT...")
        # Build SPHT (Spherical Polar Hash Table) with 45-degree filtering
        print("Building SPHT (Spherical Polar Hash Table)")
        spht = build_spht_offline(star_catalog, DEFAULT_AL_PARAMETER, max_angular_distance=45.0)
        save_spht_to_json(spht, SPHT_FILENAME)
        print(f"Built and saved SPHT with {len(spht)} entries")
    
    print("Star data initialization complete!")

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_star_name_by_hr(hr_number):
    """
    Get star name from HR number using the star catalog.
    Returns proper name if available, otherwise HR number.
    """
    for star in star_catalog:
        if star.get('HR') == hr_number:
            return star.get('N', f"HR {hr_number}")
    return f"HR {hr_number}"

def detect_stars_in_image(image_path):
    """Detect stars in the uploaded image using computer vision"""
    try:
        print("Detecting stars in image...")
        detected_stars = detect_stars(image_path)
        
        if len(detected_stars) < MIN_STARS_FOR_IDENTIFICATION:
            raise ValueError(f"Need at least {MIN_STARS_FOR_IDENTIFICATION} stars for identification, found {len(detected_stars)}")
        
        print(f"Detected {len(detected_stars)} stars")
        return detected_stars
    
    except Exception as e:
        raise Exception(f"Star detection failed: {str(e)}")

def identify_stars_using_spht(detected_stars, camera_scaling_factor, al_parameter):
    try:
        print("Identifying stars using SPHT algorithm with angular distance filtering...")
        identified_stars = stars_identification(
            detected_stars, 
            spht, 
            al_parameter, 
            camera_scaling_factor,
            max_angular_distance=45.0
        )
        
        print(f"Identified {len(identified_stars)} stars")
        return identified_stars
    
    except Exception as e:
        raise Exception(f"Star identification failed: {str(e)}")

def create_annotated_image(image_path, identified_stars):
    """Create annotated image with identified stars marked and labeled"""
    try:
        # Load and convert image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw annotations for each identified star
        for star_data in identified_stars:
            x, y = star_data['coords']
            hr_number = star_data['spht_value']
            confidence = star_data['confidence']
            
            # Get star name
            star_name = get_star_name_by_hr(hr_number)
            
            # Draw red circle around identified star
            cv2.circle(img_rgb, (x, y), STAR_CIRCLE_RADIUS, (255, 0, 0), 2)
            
            # Add star name and confidence label
            label = f"{star_name} ({confidence})"
            cv2.putText(img_rgb, label, (x-30, y-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return img_rgb
    
    except Exception as e:
        raise Exception(f"Image annotation failed: {str(e)}")

def process_star_identification(image_path, camera_scaling_factor=DEFAULT_CAMERA_SCALING_FACTOR, 
                              al_parameter=DEFAULT_AL_PARAMETER):
    """
    Complete star identification pipeline
    
    Args:
        image_path: Path to the uploaded image
        camera_scaling_factor: Camera calibration parameter
        al_parameter: Algorithm parameter
    
    Returns:
        tuple: (annotated_image, identified_stars, error_message)
    """
    try:
        # Step 1: Detect stars using computer vision
        detected_stars = detect_stars_in_image(image_path)
        
        # Step 2: Identify stars using SPHT algorithm
        identified_stars = identify_stars_using_spht(detected_stars, camera_scaling_factor, al_parameter)
        
        # Step 3: Create annotated visualization
        annotated_image = create_annotated_image(image_path, identified_stars)
        
        return annotated_image, identified_stars, None
        
    except Exception as e:
        print(f"Error in star identification pipeline: {e}")
        return None, [], str(e)

def image_to_base64(image_array):
    """Convert numpy image array to base64 string for API response"""
    try:
        pil_image = Image.fromarray(image_array)
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise Exception(f"Image conversion failed: {str(e)}")

def format_stars_for_response(identified_stars):
    """Format identified stars data for JSON API response, sorted by confidence (DESC)"""
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
    
    # Sort by confidence in descending order (highest confidence first)
    stars_list.sort(key=lambda star: star['confidence'], reverse=True)
    
    return stars_list

# API Routes
@app.route('/')
def home():
    """API home endpoint with service information"""
    return jsonify({
        "message": "Star Detector API is running!",
        "version": "1.0",
        "endpoints": {
            "/upload": "POST - Upload an image for star identification",
            "/health": "GET - Health check and service status"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint with system status"""
    return jsonify({
        "status": "healthy", 
        "service": "star-detector-api",
        "stars_loaded": len(star_catalog) if star_catalog else 0,
        "subset_stars": len(subset_bsc) if subset_bsc else 0,
        "spht_entries": len(spht) if spht else 0
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle image upload and star identification
    
    Expected form data:
        - file: Image file
        - camera_scaling_factor (optional): Camera calibration parameter  
        - al_parameter (optional): Algorithm parameter
    
    Returns:
        JSON response with identified stars and annotated image
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Get processing parameters
        camera_scaling_factor = float(request.form.get('camera_scaling_factor', DEFAULT_CAMERA_SCALING_FACTOR))
        al_parameter = float(request.form.get('al_parameter', DEFAULT_AL_PARAMETER))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Process the image through star identification pipeline
        processed_image, identified_stars, error = process_star_identification(
            upload_path, camera_scaling_factor, al_parameter
        )
        
        if processed_image is None:
            return jsonify({'error': error or 'No stars identified in image'}), 500
        
        # Convert processed image to base64
        img_base64 = image_to_base64(processed_image)
        
        # Format stars data for response
        stars_list = format_stars_for_response(identified_stars)
        
        return jsonify({
            'success': True,
            'message': f'Successfully identified {len(identified_stars)} stars in image',
            'processed_image': f'data:image/png;base64,{img_base64}',
            'detected_stars': stars_list,
            'parameters': {
                'camera_scaling_factor': camera_scaling_factor,
                'al_parameter': al_parameter
            },
            'stats': {
                'total_identified': len(identified_stars),
                'filename': filename
            }
        })
            
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Application startup
if __name__ == '__main__':
    # Only initialize on the main process, not on reloader restart
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        print("Flask reloader detected - skipping initialization in main process")
    else:
        initialize_star_data()
    
    print("Starting Flask server on http://0.0.0.0:5001")
    app.run(debug=True, host='0.0.0.0', port=5001) 