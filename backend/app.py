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

def get_ursa_major_stars():
    """
    Get the key Ursa Major constellation stars with numeric coordinates.
    These are the major stars including the Big Dipper.
    Returns stars in the same format as the main catalog for direct use.
    """
    return [
        {"B": "μ", "N": "Tania Australis", "C": "UMa", "HR": 4069, "RA": 155.58208, "Dec": 41.49944, "V": 3.05, "K": 3500, "F": "34"},
        {"B": "λ", "N": "Tania Borealis", "C": "UMa", "HR": 4033, "RA": 154.27417, "Dec": 42.91444, "V": 3.45, "K": 9500, "F": "33"},
        {"B": "θ", "N": "Sarir", "C": "UMa", "HR": 3775, "RA": 143.21417, "Dec": 51.67722, "V": 3.17, "K": 6600, "F": "25"},
        {"B": "β", "N": "Merak", "C": "UMa", "HR": 4295, "RA": 165.46042, "Dec": 56.38250, "V": 2.37, "K": 9750, "F": "48"},
        {"B": "ψ", "C": "UMa", "HR": 4335, "RA": 167.41583, "Dec": 44.49861, "V": 3.01, "K": 4850, "F": "52"},
        {"B": "χ", "N": "Al Kaphrah", "C": "UMa", "HR": 4518, "RA": 176.51250, "Dec": 47.77944, "V": 3.71, "K": 5000, "F": "63"},
        {"B": "γ", "N": "Phecda", "C": "UMa", "HR": 4554, "RA": 178.45750, "Dec": 53.69472, "V": 2.44, "K": 10000, "F": "64"},
        {"B": "δ", "N": "Megrez", "C": "UMa", "HR": 4660, "RA": 183.85667, "Dec": 57.03250, "V": 3.31, "K": 9250, "F": "69"},
        {"B": "ε", "N": "Alioth", "C": "UMa", "HR": 4905, "RA": 193.50708, "Dec": 55.95972, "V": 1.77, "K": 10000, "F": "77"},
        {"B": "α", "N": "Dubhe", "C": "UMa", "HR": 4301, "RA": 165.93208, "Dec": 61.75083, "V": 1.79, "K": 5000, "F": "50"},
        {"B": "υ", "C": "UMa", "HR": 3888, "RA": 147.74750, "Dec": 59.03861, "V": 3.80, "K": 7200, "F": "29"},
        {"C": "UMa", "HR": 3757, "RA": 142.88208, "Dec": 63.06194, "V": 3.67, "K": 7500, "F": "23"},
        {"B": "ο", "N": "Muscida", "C": "UMa", "HR": 3323, "RA": 127.56625, "Dec": 60.71806, "V": 3.36, "K": 5500, "F": "1"},
        {"B": "η", "N": "Alkaid", "C": "UMa", "HR": 5191, "RA": 206.88500, "Dec": 49.31333, "V": 1.86, "K": 24000, "F": "85"},
    ]

def create_enhanced_subset_optimized(main_catalog, ursa_major_stars, subset_size=REDUCED_CATALOG_SIZE):
    """
    Create an enhanced subset by directly adding Ursa Major stars with numeric coordinates.
    Much faster than the previous approach since we don't need to loop through main catalog.
    
    Args:
        main_catalog: Full star catalog
        ursa_major_stars: List of Ursa Major star data (now with numeric coordinates)
        subset_size: Target size for the subset
    
    Returns:
        Enhanced subset with Ursa Major stars guaranteed to be included
    """
    print(f"Creating optimized enhanced subset with {subset_size} stars...")
    
    # Start with random sample
    subset = random.sample(main_catalog, subset_size)
    
    # Create a set of existing HR numbers for quick lookup
    existing_hr_numbers = {star.get('HR') for star in subset}
    
    # Directly add missing Ursa Major stars (no need to search main catalog!)
    added_count = 0
    for ursa_star in ursa_major_stars:
        hr_number = ursa_star.get('HR')
        if hr_number not in existing_hr_numbers:
            subset.append(ursa_star)
            existing_hr_numbers.add(hr_number)
            added_count += 1
    
    print(f"Directly added {added_count} Ursa Major stars to subset")
    print(f"Final subset size: {len(subset)} stars")
    
    return subset

def initialize_star_data():
    """Initialize star catalog and build SPHT for star identification"""
    global star_catalog, subset_bsc, spht
    
    print("Initializing Star Detector API...")
    print("Loading main star catalog...")
    star_catalog = get_star_catalog()
    print(f"Loaded {len(star_catalog)} stars from main catalog")
    
    print("Getting Ursa Major constellation data...")
    ursa_major_stars = get_ursa_major_stars()
    print(f"Prepared {len(ursa_major_stars)} key Ursa Major stars")
    
    # Create enhanced subset with Ursa Major stars
    subset_bsc = create_enhanced_subset_optimized(star_catalog, ursa_major_stars)
    
    # Build SPHT (Spherical Polar Hash Table)
    print("Building SPHT (Spherical Polar Hash Table)...")
    spht = build_spht_offline(subset_bsc, DEFAULT_AL_PARAMETER)
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
    """Identify detected stars using SPHT algorithm"""
    try:
        print("Identifying stars using SPHT algorithm...")
        identified_stars = stars_identification(
            detected_stars, 
            spht, 
            al_parameter, 
            camera_scaling_factor
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