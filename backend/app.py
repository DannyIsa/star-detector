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
available_sphts = {}  # Dictionary to store multiple SPHTs {name: spht_data}

def create_reduced_catalog(main_catalog, subset_size=REDUCED_CATALOG_SIZE):
    print(f"Creating reduced catalog with {subset_size} brightest stars...")
    
    # Sort stars by visual magnitude (V parameter) - lower values = brighter stars
    # Filter out stars without V parameter first
    stars_with_magnitude = [star for star in main_catalog if 'V' in star and star['V'] is not None]
    
    if not stars_with_magnitude:
        print("Warning: No stars with magnitude data found, using first stars in catalog")
        return main_catalog[:subset_size]
    
    # Sort by V magnitude (ascending - brightest first)
    try:
        sorted_stars = sorted(stars_with_magnitude, key=lambda star: float(star['V']))
        selected_stars = sorted_stars[:subset_size]
        
        print(f"Selected stars with magnitudes from {selected_stars[0]['V']} to {selected_stars[-1]['V']}")
        return selected_stars
        
    except (ValueError, TypeError) as e:
        print(f"Error sorting by magnitude: {e}, falling back to first {subset_size} stars")
        return main_catalog[:subset_size]

def spht_file_exists():
    """Check if SPHT file already exists"""
    return os.path.exists(SPHT_FILENAME)

def get_spht_filename(al_parameter):
    """Generate SPHT filename based on AL parameter"""
    return f"spht_al_{al_parameter}.json"

def load_available_sphts():
    """Load all available SPHT files and populate available_sphts dictionary"""
    global available_sphts
    available_sphts = {}
    
    # Load default SPHT only if it exists
    if spht_file_exists():
        try:
            default_spht = load_spht_from_json(SPHT_FILENAME)
            available_sphts["default"] = {
                "spht": default_spht,
                "al_parameter": DEFAULT_AL_PARAMETER,
                "filename": SPHT_FILENAME,
                "created": "default",
                "catalog_size": len(star_catalog)  # Default uses full catalog
            }
            print(f"Loaded default SPHT with {len(default_spht)} entries")
        except Exception as e:
            print(f"Error loading default SPHT: {e}")
    
    # Look for custom SPHT files
    import glob
    spht_files = glob.glob("spht_al_*.json")
    for filename in spht_files:
        try:
            # Extract AL parameter from filename
            al_param = float(filename.replace("spht_al_", "").replace(".json", ""))
            custom_spht = load_spht_from_json(filename)
            spht_name = f"AL_{al_param}"
            available_sphts[spht_name] = {
                "spht": custom_spht,
                "al_parameter": al_param,
                "filename": filename,
                "created": "custom",
                "catalog_size": REDUCED_CATALOG_SIZE  # Default for legacy files
            }
            print(f"Loaded custom SPHT '{spht_name}' with {len(custom_spht)} entries")
        except Exception as e:
            print(f"Error loading custom SPHT {filename}: {e}")
    
    if len(available_sphts) == 0:
        print("No SPHT files found. Generate SPHTs using the frontend interface.")
    else:
        print(f"Loaded {len(available_sphts)} SPHT configurations")

def initialize_star_data():
    """Initialize star catalog and load existing SPHTs (without auto-generating)"""
    global star_catalog, subset_bsc, spht
    
    print("Initializing Star Detector API...")
    print("Loading main star catalog...")
    star_catalog = get_star_catalog()
    print(f"Loaded {len(star_catalog)} stars from main catalog")
    # star_catalog = create_reduced_catalog(star_catalog) # uncomment this to use a reduced catalog
    
    # Load default SPHT if it exists, but don't create it automatically
    if spht_file_exists():
        print(f"Loading existing default SPHT from '{SPHT_FILENAME}'...")
        spht = load_spht_from_json(SPHT_FILENAME)
        print(f"Loaded default SPHT with {len(spht)} entries")
    else:
        print(f"No default SPHT found. Use the frontend to generate SPHTs as needed.")
        spht = None
    
    # Load all available SPHTs
    load_available_sphts()
    
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

def identify_stars_using_spht(detected_stars, camera_scaling_factor, al_parameter, selected_spht=None):
    try:
        print("Identifying stars using SPHT algorithm with angular distance filtering...")
        
        # Use selected SPHT or default
        spht_to_use = selected_spht if selected_spht is not None else spht
        
        identified_stars = stars_identification(
            detected_stars, 
            spht_to_use, 
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
                              al_parameter=DEFAULT_AL_PARAMETER, spht_name="default"):
    """
    Complete star identification pipeline
    
    Args:
        image_path: Path to the uploaded image
        camera_scaling_factor: Camera calibration parameter
        al_parameter: Algorithm parameter
        spht_name: Name of the SPHT to use
    
    Returns:
        tuple: (annotated_image, identified_stars, error_message)
    """
    try:
        # Step 1: Detect stars using computer vision
        detected_stars = detect_stars_in_image(image_path)
        
        # Step 2: Get the appropriate SPHT
        selected_spht = None
        if spht_name != "default" and spht_name in available_sphts:
            selected_spht = available_sphts[spht_name]["spht"]
            # Use the AL parameter from the SPHT if not explicitly provided
            if al_parameter == DEFAULT_AL_PARAMETER:
                al_parameter = available_sphts[spht_name]["al_parameter"]
        
        # Step 3: Identify stars using SPHT algorithm
        identified_stars = identify_stars_using_spht(detected_stars, camera_scaling_factor, al_parameter, selected_spht)
        
        # Step 4: Create annotated visualization
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
    
    # Remove duplicates - keep only the highest confidence for each star
    stars_list = remove_duplicate_stars(stars_list)
    
    # Sort by confidence in descending order (highest confidence first)
    stars_list.sort(key=lambda star: star['confidence'], reverse=True)
    
    return stars_list

def remove_duplicate_stars(stars_list):
    """
    Remove duplicate star classifications, keeping only the one with highest confidence.
    Handles cases where the same star appears 2, 3, or more times.
    
    Args:
        stars_list: List of star dictionaries with 'hr', 'confidence', etc.
    
    Returns:
        List with duplicates removed, keeping highest confidence entries
    """
    if not stars_list:
        return stars_list
    
    # Group stars by HR number (catalog ID)
    star_groups = {}
    
    for star in stars_list:
        hr_number = star['hr']
        
        if hr_number not in star_groups:
            star_groups[hr_number] = []
        
        star_groups[hr_number].append(star)
    
    # Process each group - keep only the highest confidence
    unique_stars = []
    for hr_number, star_group in star_groups.items():
        if len(star_group) > 1:
            # Keep the one with highest confidence
            best_star = max(star_group, key=lambda s: s['confidence'])
            unique_stars.append(best_star)
        else:
            # No duplicates for this star
            unique_stars.append(star_group[0])
    
    return unique_stars

# API Routes
@app.route('/')
def home():
    """API home endpoint with service information"""
    return jsonify({
        "message": "Star Detector API is running!",
        "version": "2.0",
        "endpoints": {
            "/upload": "POST - Upload an image for star identification",
            "/generate-spht": "POST - Generate a new SPHT with custom AL parameter",
            "/list-sphts": "GET - List all available SPHTs",
            "/delete-spht": "POST - Delete a custom SPHT",
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
        "spht_entries": len(spht) if spht else 0,
        "available_sphts": len(available_sphts)
    })

@app.route('/generate-spht', methods=['POST'])
def generate_spht():
    """
    Generate a new SPHT with custom AL parameter
    
    Expected JSON data:
        - al_parameter: float - The AL parameter for SPHT generation
        - name: string (optional) - Custom name for the SPHT
        - catalog_size: int (optional) - Number of stars to use from catalog (default: 200)
    
    Returns:
        JSON response with generation status
    """
    try:
        data = request.get_json()
        
        if not data or 'al_parameter' not in data:
            return jsonify({'error': 'al_parameter is required'}), 400
        
        al_parameter = float(data['al_parameter'])
        custom_name = data.get('name', f"AL_{al_parameter}")
        catalog_size = int(data.get('catalog_size', REDUCED_CATALOG_SIZE))
        
        if al_parameter <= 0:
            return jsonify({'error': 'al_parameter must be positive'}), 400
        
        if catalog_size <= 0 or catalog_size > len(star_catalog):
            return jsonify({'error': f'catalog_size must be between 1 and {len(star_catalog)}'}), 400
        
        # Generate filename
        filename = get_spht_filename(al_parameter)
        
        # Check if SPHT already exists
        if custom_name in available_sphts:
            return jsonify({'error': f'SPHT with name "{custom_name}" already exists'}), 400
        
        print(f"Building new SPHT with AL parameter: {al_parameter}, using {catalog_size} stars")
        
        # Create reduced catalog if needed
        working_catalog = star_catalog
        if catalog_size < len(star_catalog):
            working_catalog = create_reduced_catalog(star_catalog, catalog_size)
            print(f"Using reduced catalog with {len(working_catalog)} stars")
        
        # Build the SPHT
        new_spht = build_spht_offline(working_catalog, al_parameter, max_angular_distance=45.0)
        
        # Save to file
        save_spht_to_json(new_spht, filename)
        
        # Add to available SPHTs
        available_sphts[custom_name] = {
            "spht": new_spht,
            "al_parameter": al_parameter,
            "filename": filename,
            "created": "custom",
            "catalog_size": catalog_size
        }
        
        return jsonify({
            'success': True,
            'message': f'SPHT "{custom_name}" generated successfully',
            'spht_name': custom_name,
            'al_parameter': al_parameter,
            'catalog_size': catalog_size,
            'filename': filename,
            'entries': len(new_spht)
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/list-sphts', methods=['GET'])
def list_sphts():
    """
    List all available SPHTs
    
    Returns:
        JSON response with list of available SPHTs
    """
    try:
        sphts_info = []
        for name, info in available_sphts.items():
            sphts_info.append({
                'name': name,
                'al_parameter': info['al_parameter'],
                'catalog_size': info.get('catalog_size', 'Unknown'),
                'entries': len(info['spht']),
                'created': info['created'],
                'filename': info['filename']
            })
        
        return jsonify({
            'success': True,
            'sphts': sphts_info,
            'total': len(sphts_info),
            'max_catalog_size': len(star_catalog)
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/delete-spht', methods=['POST'])
def delete_spht():
    """
    Delete a custom SPHT
    
    Expected JSON data:
        - name: string - Name of the SPHT to delete
    
    Returns:
        JSON response with deletion status
    """
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({'error': 'SPHT name is required'}), 400
        
        spht_name = data['name']
        
        if spht_name == "default":
            return jsonify({'error': 'Cannot delete default SPHT'}), 400
        
        if spht_name not in available_sphts:
            return jsonify({'error': f'SPHT "{spht_name}" not found'}), 404
        
        # Get filename and delete file
        filename = available_sphts[spht_name]['filename']
        if os.path.exists(filename):
            os.remove(filename)
        
        # Remove from available SPHTs
        del available_sphts[spht_name]
        
        return jsonify({
            'success': True,
            'message': f'SPHT "{spht_name}" deleted successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle image upload and star identification
    
    Expected form data:
        - file: Image file
        - camera_scaling_factor (optional): Camera calibration parameter  
        - al_parameter (optional): Algorithm parameter
        - spht_name (optional): Name of the SPHT to use
    
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
        
        # Check if any SPHTs are available
        if len(available_sphts) == 0:
            return jsonify({'error': 'No SPHT algorithms available. Please generate an SPHT first using the SPHT Manager.'}), 400
        
        # Get processing parameters
        camera_scaling_factor = float(request.form.get('camera_scaling_factor', DEFAULT_CAMERA_SCALING_FACTOR))
        al_parameter = float(request.form.get('al_parameter', DEFAULT_AL_PARAMETER))
        spht_name = request.form.get('spht_name', 'default')
        
        # If default is requested but doesn't exist, use the first available SPHT
        if spht_name == 'default' and 'default' not in available_sphts:
            if len(available_sphts) > 0:
                spht_name = list(available_sphts.keys())[0]
                print(f"Default SPHT not found, using '{spht_name}' instead")
            else:
                return jsonify({'error': 'No SPHT algorithms available. Please generate an SPHT first.'}), 400
        
        # Validate SPHT selection
        if spht_name not in available_sphts:
            return jsonify({'error': f'SPHT "{spht_name}" not found. Available SPHTs: {list(available_sphts.keys())}'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Process the image through star identification pipeline
        processed_image, identified_stars, error = process_star_identification(
            upload_path, camera_scaling_factor, al_parameter, spht_name
        )
        
        if processed_image is None:
            return jsonify({'error': error or 'No stars identified in image'}), 500
        
        # Convert processed image to base64
        img_base64 = image_to_base64(processed_image)
        
        # Format stars data for response
        stars_list = format_stars_for_response(identified_stars)
        
        return jsonify({
            'success': True,
            'message': f'Successfully identified {len(stars_list)} stars in image',
            'processed_image': f'data:image/png;base64,{img_base64}',
            'detected_stars': stars_list,
            'parameters': {
                'camera_scaling_factor': camera_scaling_factor,
                'al_parameter': al_parameter,
                'spht_name': spht_name
            },
            'stats': {
                'total_identified': len(stars_list),
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