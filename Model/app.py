# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
import sys
import os
import traceback # For detailed error logging

# --- Dynamically add hga.py's directory to sys.path ---
# Option 1: If hga.py is in a known fixed location outside the project
hga_file_path_abs = r'd:\Users\BONJOUR\Downloads\Model\hga.py'
hga_dir_abs = os.path.dirname(hga_file_path_abs)
if hga_dir_abs not in sys.path:
    sys.path.insert(0, hga_dir_abs)

# Option 2: If hga.py is copied into the same directory as app.py (recommended for simplicity)
# In this case, direct import 'import hga' should work without sys.path modification.
# If you choose this, you can comment out the sys.path modification above.

try:
    import hga # This will now import your hga.py
except ImportError as e:
    print(f"CRITICAL: Error importing hga.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Attempted to load from directory: {hga_dir_abs if 'hga_dir_abs' in locals() else 'current directory'}")
    print("Ensure hga.py is accessible and all its dependencies (numpy, matplotlib) are installed.")
    raise # Re-raise the exception to stop the app if hga.py can't be loaded.

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, helpful for development

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        container_data = data.get('container')
        boxes_data = data.get('boxes')
        ga_settings = data.get('ga_settings', {})

        if not container_data or not boxes_data:
            return jsonify({"error": "Missing container or boxes data"}), 400

        # Validate data types (basic example)
        try:
            container_data = {k: float(v) for k, v in container_data.items()}
            valid_boxes = []
            for i, b in enumerate(boxes_data):
                if not all(k in b for k in ['id', 'db', 'wb', 'hb']):
                     return jsonify({"error": f"Box at index {i} is missing required fields (id, db, wb, hb)"}), 400
                
                # Get allowed_orientations, default to all if not present or invalid
                allowed_orientations_input = b.get('allowed_orientations')
                if isinstance(allowed_orientations_input, list) and \
                   all(isinstance(x, int) and 1 <= x <= 6 for x in allowed_orientations_input) and \
                   allowed_orientations_input: # Ensure it's not an empty list
                    # Use set to remove duplicates, then convert back to list
                    processed_orientations = sorted(list(set(allowed_orientations_input))) 
                else:
                    # If not provided, or malformed, or empty, default to all 6 orientations
                    processed_orientations = list(range(1, 7))
                valid_boxes.append({
                    'id': int(b['id']),
                    'db': float(b['db']),
                    'wb': float(b['wb']),
                    'hb': float(b['hb']),
                    'allowed_orientations': processed_orientations # Add to the dict passed to hga
                })
            boxes_data = valid_boxes
        except ValueError:
            return jsonify({"error": "Invalid data type for container or box dimensions. Must be numbers."}), 400
        
        except TypeError: # Catch if 'allowed_orientations' is not iterable, etc.
            return jsonify({"error": "Invalid format for allowed_orientations. Must be a list of integers (1-6)."}), 400


        default_ga_params = {
            'npop': int(ga_settings.get('npop', 50)),
            'pcross': float(ga_settings.get('pcross', 0.67)),
            'pmut': float(ga_settings.get('pmut', 0.33)),
            'nrep': int(ga_settings.get('nrep', 10)),
            'nmerge': int(ga_settings.get('nmerge', 10)),
            'max_generations': int(ga_settings.get('max_generations', 50)),
            'max_time_limit': int(ga_settings.get('max_time_limit', 30))
        }
        
        app.logger.info(f"Received solve request. Container: {container_data}, Boxes: {len(boxes_data)}, GA Params: {default_ga_params}")

        solution = hga.solve_loading_problem(container_data, boxes_data, default_ga_params)
        
        app.logger.info("GA solution processed.")
        return jsonify(solution)

    except Exception as e:
        app.logger.error(f"Error during GA execution: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    # Ensure hga.py can be imported. If you didn't copy hga.py,
    # the sys.path modification at the top of the file handles it.
    app.run(debug=True, host='0.0.0.0') # debug=True is for development