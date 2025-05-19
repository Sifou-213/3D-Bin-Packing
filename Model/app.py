from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os
import traceback

# --- IMPORTANT: Use relative import for hga.py ---
# Copy your hga.py file INTO the same folder as app.py for simplicity
# Then you don't need the sys.path hacks below; comment them out.

# sys.path hack (optional, only if hga.py is outside current folder)
# hga_file_path_abs = r'd:\Users\BONJOUR\Downloads\Model\hga.py'
# hga_dir_abs = os.path.dirname(hga_file_path_abs)
# if hga_dir_abs not in sys.path:
#     sys.path.insert(0, hga_dir_abs)

try:
    import hga
except ImportError as e:
    print(f"CRITICAL: Error importing hga.py: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

app = Flask(__name__)
CORS(app)

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

        try:
            container_data = {k: float(v) for k, v in container_data.items()}
            valid_boxes = []
            for i, b in enumerate(boxes_data):
                if not all(k in b for k in ['id', 'db', 'wb', 'hb']):
                    return jsonify({"error": f"Box at index {i} missing fields"}), 400

                allowed_orientations_input = b.get('allowed_orientations')
                if isinstance(allowed_orientations_input, list) and \
                   all(isinstance(x, int) and 1 <= x <= 6 for x in allowed_orientations_input) and \
                   allowed_orientations_input:
                    processed_orientations = sorted(list(set(allowed_orientations_input)))
                else:
                    processed_orientations = list(range(1, 7))

                valid_boxes.append({
                    'id': int(b['id']),
                    'db': float(b['db']),
                    'wb': float(b['wb']),
                    'hb': float(b['hb']),
                    'allowed_orientations': processed_orientations
                })
            boxes_data = valid_boxes
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid data types or formats"}), 400

        default_ga_params = {
            'npop': int(ga_settings.get('npop', 50)),
            'pcross': float(ga_settings.get('pcross', 0.67)),
            'pmut': float(ga_settings.get('pmut', 0.33)),
            'nrep': int(ga_settings.get('nrep', 10)),
            'nmerge': int(ga_settings.get('nmerge', 10)),
            'max_generations': int(ga_settings.get('max_generations', 50)),
            'max_time_limit': int(ga_settings.get('max_time_limit', 30))
        }

        app.logger.info(f"Received solve request: container={container_data}, boxes={len(boxes_data)}")

        solution = hga.solve_loading_problem(container_data, boxes_data, default_ga_params)
        return jsonify(solution)

    except Exception as e:
        app.logger.error(f"Error during GA execution: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
