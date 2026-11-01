from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, abort
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
import sys
import os
import io
import base64
import traceback # For detailed error logging
import json # For loading local JSON files
import logging # For more explicit logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # hga imports pyplot; never open GUI windows from the web server
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment
from openpyxl.drawing.image import Image # For when image data is available

# --- Solver modules live next to this file ---
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

try:
    import hga 
    import pallet_loading 
    import multi_container_solver
    import compaction
    import report
except ImportError as e:
    module_name = "a required module"
    if "hga" in str(e).lower(): module_name = "hga.py"
    elif "pallet_loading" in str(e).lower(): module_name = "pallet_loading.py"
    elif "multi_container_solver" in str(e).lower(): module_name = "multi_container_solver.py"
    
    print(f"CRITICAL: Error importing {module_name}: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Attempted to load from directory: {project_dir}")
    print(f"Ensure {module_name} is accessible and all its dependencies (e.g., numpy) are installed.")
    raise 

app = Flask(__name__)
CORS(app) 

# Define paths to JSON data files
ITEMS_DATA_PATH = os.path.join(app.static_folder, 'Data', 'Items data.json')
TRUCKS_DATA_PATH = os.path.join(app.static_folder, 'Data', 'camions.json')

# --- Define base directory for saving results ---
RESULTS_BASE_DIR = os.path.join(app.root_path, 'shipment_results_archive')
if not os.path.exists(RESULTS_BASE_DIR):
    os.makedirs(RESULTS_BASE_DIR)

# Configure logging
logging.basicConfig(level=logging.INFO)
werkzeug_logger = logging.getLogger('werkzeug') # Get Werkzeug logger
werkzeug_logger.setLevel(logging.INFO) # Set its level if you want to see request logs

def _count_entries(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return len(json.load(f))
    except (OSError, ValueError):
        return 0

@app.route('/')
def landing():
    return render_template('landing.html',
                           product_count=_count_entries(ITEMS_DATA_PATH),
                           truck_count=_count_entries(TRUCKS_DATA_PATH))

@app.route('/app')
def workspace():
    mode = request.args.get('mode', 'truck')
    if mode not in ('truck', 'fleet', 'pallet'):
        mode = 'truck'
    return render_template('workspace.html', mode=mode)

# Old URLs from the previous UI keep working and land in the new workspace.
@app.route('/shipment-setup')
@app.route('/solver-selection')
@app.route('/single-container-solver')
def legacy_truck():
    return redirect(url_for('workspace', mode='truck'))

@app.route('/multi-container-solver-page')
def legacy_fleet():
    return redirect(url_for('workspace', mode='fleet'))

@app.route('/pallet-solver')
def legacy_pallet():
    return redirect(url_for('workspace', mode='pallet'))

def _validate_and_prepare_box_data(boxes_data_from_frontend):
    if not boxes_data_from_frontend:
        return [], {} 

    valid_boxes_for_backend = []
    input_box_details_map = {} 

    for i, b_frontend in enumerate(boxes_data_from_frontend):
        required_fields = ['id', 'db', 'wb', 'hb', 'weight', 'lbs', 'name', 'original_type_id']
        missing_fields = [field for field in required_fields if field not in b_frontend or b_frontend[field] is None] # Check for None too
        if missing_fields:
            raise ValueError(f"Box at index {i} (Name: {b_frontend.get('name', 'N/A')}, PageID: {b_frontend.get('original_type_id', 'N/A')}) is missing or has null for required fields: {missing_fields}. All fields must have valid values.")

        try:
            box_id_instance = int(b_frontend['id']) 
            
            input_box_details_map[box_id_instance] = {
                'name': b_frontend.get('name', f"Box_{box_id_instance}"), 
                'original_type_id': b_frontend.get('original_type_id', str(box_id_instance))
            }

            allowed_orientations_input = b_frontend.get('allowed_orientations')
            if isinstance(allowed_orientations_input, list) and \
               all(isinstance(x, int) and 1 <= x <= 6 for x in allowed_orientations_input) and \
               allowed_orientations_input:
                processed_orientations = sorted(list(set(allowed_orientations_input)))
            else:
                processed_orientations = list(range(1, 7))
            
            db_val = float(b_frontend['db'])
            wb_val = float(b_frontend['wb'])
            hb_val = float(b_frontend['hb'])
            weight_val = float(b_frontend['weight'])
            lbs_val = float(b_frontend['lbs'])

            if not (db_val > 0 and wb_val > 0 and hb_val > 0 and weight_val >= 0 and lbs_val >=0):
                 raise ValueError(f"Box {b_frontend.get('name', 'N/A')} dimensions must be positive, weight/LBS non-negative.")


            valid_boxes_for_backend.append({
                'id': box_id_instance, 
                'db': db_val,
                'wb': wb_val,
                'hb': hb_val,
                'weight': weight_val,
                'lbs': lbs_val,
                'allowed_orientations': processed_orientations,
                'name': b_frontend['name'],
                'original_type_id': b_frontend['original_type_id'] 
            })
        except ValueError as ve:
            raise ValueError(f"Invalid data type for box at index {i} (Name: {b_frontend.get('name', 'N/A')}, PageID: {b_frontend.get('original_type_id', 'N/A')}): {ve}")
        except TypeError as te: # Catches float(None) for example
            raise TypeError(f"Invalid data format (likely null value for a number) for box at index {i} (Name: {b_frontend.get('name', 'N/A')}, PageID: {b_frontend.get('original_type_id', 'N/A')}): {te}")
            
    return valid_boxes_for_backend, input_box_details_map


@app.route('/solve_shipment_single', methods=['POST'])
def solve_shipment_single():
    app.logger.info(f"Received request for /solve_shipment_single")
    try:
        data = request.get_json()
        if not data:
            app.logger.error("Invalid JSON payload received for /solve_shipment_single")
            return jsonify({"error": "Invalid JSON payload"}), 400

        container_data_frontend = data.get('container')
        boxes_data_frontend = data.get('boxes') 
        ga_settings_frontend = data.get('ga_settings', {})

        if not container_data_frontend:
            app.logger.error("Missing container data for /solve_shipment_single")
            return jsonify({"error": "Missing container data"}), 400
        
        try:
            container_data_validated = {k: float(v) for k, v in container_data_frontend.items()}
            if not (container_data_validated['dc'] > 0 and container_data_validated['wc'] > 0 and container_data_validated['hc'] > 0):
                raise ValueError("Container dimensions must be positive.")
            boxes_for_hga, input_box_details_map = _validate_and_prepare_box_data(boxes_data_frontend)
            if not boxes_for_hga and boxes_data_frontend: # Validation failed but didn't raise for some reason, or no valid boxes
                 app.logger.warning("No valid boxes to process after validation for /solve_shipment_single, but frontend sent boxes.")
                 # _validate_and_prepare_box_data should raise if issues, this is a fallback
                 return jsonify({"error": "No valid boxes to process after validation."}), 400
            elif not boxes_for_hga and not boxes_data_frontend:
                 app.logger.warning("No boxes provided in the request for /solve_shipment_single.")
                 return jsonify({"error": "No boxes provided to pack."}), 400


        except (ValueError, TypeError) as e:
            app.logger.error(f"Data validation error for /solve_shipment_single: {str(e)}")
            return jsonify({"error": str(e)}), 400

        default_ga_params = {
            'npop': int(ga_settings_frontend.get('npop', 50)),
            'pcross': float(ga_settings_frontend.get('pcross', 0.67)), 
            'pmut': float(ga_settings_frontend.get('pmut', 0.33)),   
            'nrep': int(ga_settings_frontend.get('nrep', 10)),
            'nmerge': int(ga_settings_frontend.get('nmerge', 10)),
            'max_generations': int(ga_settings_frontend.get('max_generations', 50)),
            'max_time_limit': int(ga_settings_frontend.get('max_time_limit', 30))
        }
        
        app.logger.info(f"Processing SINGLE shipment. Container: {container_data_validated}, Boxes: {len(boxes_for_hga)}, GA: {default_ga_params}")
        solution = hga.solve_loading_problem(container_data_validated, boxes_for_hga, default_ga_params)
        compaction.compact_solution(solution)

        if 'metrics' in solution and 'unstowed_box_ids' in solution['metrics']:
            augmented_unstowed = []
            for box_instance_id in solution['metrics']['unstowed_box_ids']: 
                details = input_box_details_map.get(box_instance_id, 
                                                    {'name': f'Unknown_{box_instance_id}', 
                                                     'original_type_id': f'UnknownType_{box_instance_id}'})
                augmented_unstowed.append({'id': box_instance_id, 'name': details['name'], 'original_type_id': details['original_type_id']})
            solution['metrics']['unstowed_boxes_detailed'] = augmented_unstowed

        if 'visualization_data' in solution and 'layers' in solution['visualization_data']:
            for layer in solution['visualization_data']['layers']:
                for box_viz_info in layer.get('boxes', []): 
                    details = input_box_details_map.get(box_viz_info['id'], 
                                                        {'name': f'Unknown_{box_viz_info["id"]}', 
                                                         'original_type_id': f'UnknownType_{box_viz_info["id"]}'})
                    box_viz_info['name'] = details['name']
                    box_viz_info['original_type_id'] = details['original_type_id']
                    box_viz_info['rv_id'] = box_viz_info.get('rv_id', 'N/A')
            # time_taken_seconds is already part of solution['metrics'] from hga.py

        app.logger.info(f"Successfully processed /solve_shipment_single")
        return jsonify(solution)

    except Exception as e:
        app.logger.error(f"Unhandled error in /solve_shipment_single: {e}\n{traceback.format_exc()}")
        app.logger.info(f"Caught exception in /solve_shipment_single, preparing JSON error response.") # ADDED LOG
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/solve_shipment_multi', methods=['POST'])
def solve_shipment_multi():
    app.logger.info(f"Received request for /solve_shipment_multi")
    try:
        data = request.get_json()
        if not data:
            app.logger.error("Invalid JSON payload received for /solve_shipment_multi")
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Determine container data source: single definition or list of types
        defined_container_types_frontend = data.get('defined_container_types')
        single_container_data_frontend = data.get('container')
        boxes_data_frontend = data.get('boxes') 
        ga_settings_frontend = data.get('ga_settings', {})

        container_to_use_for_solver = None

        if defined_container_types_frontend and isinstance(defined_container_types_frontend, list) and len(defined_container_types_frontend) > 0:
            # For now, if multiple types are defined, use the FIRST one for the current multi_container_solver.
            # Future: multi_container_solver.py could be enhanced to use the whole list.
            app.logger.info(f"Multiple container types defined, using the first one: {defined_container_types_frontend[0]}")
            container_to_use_for_solver = defined_container_types_frontend[0]
        elif single_container_data_frontend:
            container_to_use_for_solver = single_container_data_frontend
        else:
            app.logger.error("Missing container data (neither 'container' nor 'defined_container_types' provided) for /solve_shipment_multi")
            return jsonify({"error": "Missing container data for multi-container solver"}), 400

        try:
            dc_val = float(container_to_use_for_solver['dc'])
            wc_val = float(container_to_use_for_solver['wc'])
            hc_val = float(container_to_use_for_solver['hc'])
            if not (dc_val > 0 and wc_val > 0 and hc_val > 0):
                raise ValueError("Container dimensions must be positive.")
            container_dims_tuple = (dc_val, wc_val, hc_val)
            app.logger.info(f"Multi-container solver will use container dimensions: {container_dims_tuple}")
            
            raw_boxes_for_multi, _ = _validate_and_prepare_box_data(boxes_data_frontend) # input_box_details_map not strictly needed here as multi_solver handles its own augmentation
            if not raw_boxes_for_multi and boxes_data_frontend:
                 app.logger.warning("No valid boxes to process after validation for /solve_shipment_multi, but frontend sent boxes.")
                 return jsonify({"error": "No valid boxes to process after validation."}), 400
            elif not raw_boxes_for_multi and not boxes_data_frontend:
                 app.logger.warning("No boxes provided in the request for /solve_shipment_multi.")
                 return jsonify({"error": "No boxes provided to pack."}), 400
            
            hga_box_objects_for_multi = []
            for b_dict in raw_boxes_for_multi:
                vb = b_dict['db'] * b_dict['wb'] * b_dict['hb']
                hga_box_objects_for_multi.append(hga.hybrid_GA.box(
                    id=b_dict['id'], 
                    db=b_dict['db'], wb=b_dict['wb'], hb=b_dict['hb'], vb=vb,
                    allowed_orientations=b_dict['allowed_orientations'],
                    name=b_dict['name'],
                    original_type_id=b_dict['original_type_id'], 
                    weight=b_dict['weight'],
                    load_bearing_strength=b_dict['lbs']
                ))

        except (ValueError, TypeError) as e:
            app.logger.error(f"Data validation error for /solve_shipment_multi: {str(e)}")
            return jsonify({"error": str(e)}), 400

        max_containers_to_use = int(ga_settings_frontend.get('max_containers', 5))
        generations_per_container = int(ga_settings_frontend.get('max_generations', 30)) 
        npop_per_container = int(ga_settings_frontend.get('npop', 30))
        
        app.logger.info(f"Processing MULTI shipment. Container Dims: {container_dims_tuple}, Boxes: {len(hga_box_objects_for_multi)}, Max Containers: {max_containers_to_use}")

        multi_results = multi_container_solver.solve_multi_container(
            boxes=hga_box_objects_for_multi,
            container_dims=container_dims_tuple,
            max_containers=max_containers_to_use,
            generations=generations_per_container,
            npop=npop_per_container,
            nrep=min(int(ga_settings_frontend.get('nrep', 10)), npop_per_container),
            time_limit_per_container=int(ga_settings_frontend.get('max_time_limit', 30))
        )

        for container_result in multi_results:
            compaction.compact_solution(container_result)
        # Each item in multi_results should already contain its own 'time_taken_seconds' in its 'metrics'.
        app.logger.info(f"Multi-container solver returned {len(multi_results)} container plans for /solve_shipment_multi.")
        return jsonify(multi_results) 

    except Exception as e:
        app.logger.error(f"Unhandled error in /solve_shipment_multi: {e}\n{traceback.format_exc()}")
        app.logger.info(f"Caught exception in /solve_shipment_multi, preparing JSON error response.") # ADDED LOG
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/solve', methods=['POST'])
def solve():
    app.logger.info(f"Received request for /solve (original single container page)")
    try:
        data = request.get_json()
        if not data: 
            app.logger.error("Invalid JSON payload for /solve")
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        container_data_frontend = data.get('container')
        boxes_data_frontend = data.get('boxes')
        ga_settings_frontend = data.get('ga_settings', {})

        if not container_data_frontend: 
            app.logger.error("Missing container data for /solve")
            return jsonify({"error": "Missing container data"}), 400
        
        # Use the robust validation for this route too
        try:
            container_data_validated = {k: float(v) for k, v in container_data_frontend.items()}
            if not (container_data_validated['dc'] > 0 and container_data_validated['wc'] > 0 and container_data_validated['hc'] > 0):
                raise ValueError("Container dimensions must be positive.")
            
            # The /solve route might get slightly different box structures if not loaded from shipment.
            # We adapt _validate_and_prepare_box_data or use a similar inline validation.
            # For consistency, let's try to use it, assuming frontend sends compatible 'original_type_id' or 'name'.
            # If 'original_type_id' is missing, _validate_and_prepare_box_data will default it.
            boxes_for_hga, input_box_details_map = _validate_and_prepare_box_data(boxes_data_frontend if boxes_data_frontend else [])
            
            if not boxes_for_hga and boxes_data_frontend:
                 app.logger.warning("No valid boxes to process after validation for /solve, but frontend sent boxes.")
                 return jsonify({"error": "No valid boxes to process after validation."}), 400
            elif not boxes_for_hga and not boxes_data_frontend:
                 app.logger.warning("No boxes provided in the request for /solve.")
                 return jsonify({"error": "No boxes provided to pack."}), 400

        except (ValueError, TypeError) as e:
            app.logger.error(f"Data validation error for /solve: {str(e)}")
            return jsonify({"error": str(e)}), 400

        default_ga_params = {
            'npop': int(ga_settings_frontend.get('npop', 50)),
            'pcross': float(ga_settings_frontend.get('pcross', 0.67)), 
            'pmut': float(ga_settings_frontend.get('pmut', 0.33)),
            'nrep': int(ga_settings_frontend.get('nrep', 10)), 
            'nmerge': int(ga_settings_frontend.get('nmerge', 10)),
            'max_generations': int(ga_settings_frontend.get('max_generations', 50)),
            'max_time_limit': int(ga_settings_frontend.get('max_time_limit', 30))
        }
        
        app.logger.info(f"Processing /solve request. Container: {container_data_validated}, Boxes: {len(boxes_for_hga)}, GA: {default_ga_params}")
        solution = hga.solve_loading_problem(container_data_validated, boxes_for_hga, default_ga_params)
        compaction.compact_solution(solution)
        
        if 'metrics' in solution and 'unstowed_box_ids' in solution['metrics']:
            augmented_unstowed_boxes = []
            for box_id_instance in solution['metrics']['unstowed_box_ids']:
                details = input_box_details_map.get(box_id_instance, {'name': f'Unknown_{box_id_instance}', 'original_type_id': f'Unknown_{box_id_instance}'})
                augmented_unstowed_boxes.append({'id': box_id_instance, 'name': details['name'], 'original_type_id': details['original_type_id']})
            solution['metrics']['unstowed_boxes_detailed'] = augmented_unstowed_boxes

        if 'visualization_data' in solution and 'layers' in solution['visualization_data']:
            for layer_detail in solution['visualization_data']['layers']:
                if 'boxes' in layer_detail: 
                    for packed_box_info in layer_detail['boxes']:
                        instance_id = packed_box_info['id'] 
                        details = input_box_details_map.get(instance_id, {'name': f'Unknown_{instance_id}', 'original_type_id': f'Unknown_{instance_id}'})
                        packed_box_info['name'] = details['name']
                        packed_box_info['original_type_id'] = details['original_type_id']
                        # time_taken_seconds is already part of solution['metrics'] from hga.py
                        packed_box_info['rv_id'] = packed_box_info.get('rv_id', 'N/A') 

        app.logger.info(f"Successfully processed /solve")
        return jsonify(solution)

    except Exception as e:
        app.logger.error(f"Unhandled error in /solve: {e}\n{traceback.format_exc()}")
        app.logger.info(f"Caught exception in /solve, preparing JSON error response.") # ADDED LOG
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/solve_pallet_loading', methods=['POST'])
def solve_pallet_loading_route():
    app.logger.info(f"Received request for /solve_pallet_loading")
    try:
        data = request.get_json()
        if not data:
            app.logger.error("No data provided for /solve_pallet_loading")
            return jsonify({"error": "No data provided"}), 400

        pallet_data_from_frontend = data.get('pallet_data')
        boxes_data_from_frontend = data.get('boxes_to_pack')

        if not pallet_data_from_frontend or not boxes_data_from_frontend:
            app.logger.error("Missing pallet_data or boxes_to_pack for /solve_pallet_loading")
            return jsonify({"error": "Missing pallet_data or boxes_to_pack"}), 400

        pallet_props_for_solver = {
            "length": float(pallet_data_from_frontend.get("length")),
            "width": float(pallet_data_from_frontend.get("width")),
            "empty_height": float(pallet_data_from_frontend.get("empty_height", 0)),
            "max_load_height": float(pallet_data_from_frontend.get("max_load_height")),
            "max_weight": float(pallet_data_from_frontend.get("max_weight"))
        }

        boxes_props_for_solver = []
        for box_fe in boxes_data_from_frontend:
            base_dims_input = box_fe.get("base_dimensions")
            if not isinstance(base_dims_input, list) or len(base_dims_input) != 3:
                try:
                    base_dims_input = [
                        float(box_fe.get("base_length")),
                        float(box_fe.get("base_width")),
                        float(box_fe.get("base_height"))
                    ]
                    if not all(isinstance(d, (int,float)) and d > 0 for d in base_dims_input): 
                        raise ValueError("Fallback to l,w,h failed or produced non-positive/non-numeric values")
                except (ValueError, TypeError, KeyError): # Catch KeyError if base_length etc. are missing
                    app.logger.error(f"Invalid or missing base_dimensions/l,w,h for box: {box_fe.get('name')}. Skipping.")
                    return jsonify({"error": f"Invalid base_dimensions or l,w,h for box {box_fe.get('name')}. Expected 3 positive numbers."}), 400

            boxes_props_for_solver.append({ 
                "id": box_fe.get("pageId") or box_fe.get("name", f"BoxType_{len(boxes_props_for_solver)}"),
                "name": box_fe.get("name"),
                "quantity": int(box_fe.get("quantity")),
                "base_dimensions": base_dims_input, 
                "allowed_orientations": box_fe.get("allowed_orientations_list", [1]), # Pallet solver might use different orientation system
                "weight": float(box_fe.get("weightKg", 0)),
                "lbs": float(box_fe.get("lbs", 100000)),
                "stackable": box_fe.get("is_stackable", True),
                "can_support_stacking": box_fe.get("can_support_stacking", True)
            })
        
        app.logger.info(f"Processing pallet solve. Pallet: {pallet_props_for_solver}, Boxes: {len(boxes_props_for_solver)}")
        placements, metrics, pallet_details_used, unpacked_boxes = pallet_loading.solve_pallet_loading_problem(pallet_props_for_solver, boxes_props_for_solver)
        
        app.logger.info(f"Successfully processed /solve_pallet_loading")
        return jsonify({
            "placements": placements, 
            "metrics": metrics,
            "pallet_dimensions_used": pallet_details_used,
            "unpacked_boxes": unpacked_boxes
            })
    except Exception as e:
        app.logger.error(f"Unhandled error in /solve_pallet_loading: {e}\n{traceback.format_exc()}")
        app.logger.info(f"Caught exception in /solve_pallet_loading, preparing JSON error response.") # ADDED LOG
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/get_product_types', methods=['GET'])
def get_product_types_route():
    try:
        with open(ITEMS_DATA_PATH, 'r', encoding='utf-8') as f:
            items = json.load(f)
        
        product_types = sorted(list(set(item.get("Type", "Unknown") for item in items)))
        return jsonify(product_types)
    except FileNotFoundError:
        app.logger.error(f"Items data file not found at {ITEMS_DATA_PATH}")
        return jsonify({"error": "Items data file not found"}), 404
    except Exception as e:
        app.logger.error(f"Error processing product types: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_items_data', methods=['GET'])
def get_items_data_route():
    try:
        with open(ITEMS_DATA_PATH, 'r', encoding='utf-8') as f:
            items = json.load(f)
        return jsonify(items)
    except FileNotFoundError:
        app.logger.error(f"Items data file not found at {ITEMS_DATA_PATH}")
        return jsonify({"error": "Items data file not found"}), 404
    except Exception as e:
        app.logger.error(f"Error loading items data: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_truck_data', methods=['GET'])
def get_truck_data_route():
    try:
        with open(TRUCKS_DATA_PATH, 'r', encoding='utf-8') as f:
            trucks = json.load(f)
        return jsonify(trucks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_shipment_results', methods=['POST'])
def save_shipment_results_route():
    try:
        app.logger.info("Received request for /save_shipment_results")
        payload = request.get_json()
        client_name = payload.get('client_name', 'UnknownClient')
        setup_data = payload.get('setup_data', {})
        solution_data = payload.get('solution_data', {})

        if not client_name or not setup_data or not solution_data:
            return jsonify({"error": "Missing data for saving results."}), 400

        safe_client_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in client_name)
        if not safe_client_name: safe_client_name = "DefaultClient"
        app.logger.info(f"Safe Client Name: {safe_client_name}")

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        app.logger.info(f"Date: {date_str}, Time: {time_str}")

        client_day_path = os.path.join(RESULTS_BASE_DIR, date_str, safe_client_name)
        app.logger.info(f"Attempting to create directory: {client_day_path}")
        os.makedirs(client_day_path, exist_ok=True)
        app.logger.info(f"Directory ensured: {client_day_path}")

        filename = f"results_{time_str}.xlsx"
        filepath = os.path.join(client_day_path, filename)

        app.logger.info(f"Target Excel filepath: {filepath}")

        app.logger.info("Creating Excel workbook...")

        wb = Workbook()

        # Define a bold font style
        bold_font = Font(bold=True)

        ws_setup = wb.active
        ws_setup.title = "Paramètres"
        
        row_idx = 1
        ws_setup[f'A{row_idx}'] = "Chargement :"
        ws_setup[f'B{row_idx}'] = client_name
        row_idx += 2
        ws_setup[f'A{row_idx-2}'].font = bold_font # Bold "Chargement :"

        app.logger.info("Writing container setup to Excel...")

        ws_setup[f'A{row_idx}'] = "Camion :"
        ws_setup[f'A{row_idx}'].font = bold_font
        row_idx += 1
        if setup_data.get('container'): 
            cont = setup_data['container']
            current_row = row_idx + 1 # For direct cell access
            headers = ["Type", "Longueur (m)", "Largeur (m)", "Hauteur (m)"]
            for col_num, header_text in enumerate(headers, 1):
                cell = ws_setup.cell(row=current_row, column=col_num, value=header_text)
                cell.font = bold_font
            current_row += 1
            ws_setup.cell(row=current_row, column=1, value="Camion utilisé")
            for col_idx, val in enumerate([cont.get('dc'), cont.get('wc'), cont.get('hc')], start=2):
                cell = ws_setup.cell(row=current_row, column=col_idx, value=float(val) if val is not None else None)
                cell.number_format = '0.00' # Apply number format
            row_idx += 3 # Increased spacing
        elif setup_data.get('defined_container_types'): 
            ws_setup.append(["Réf.", "Nom", "Longueur (m)", "Largeur (m)", "Hauteur (m)"])
            for cont_type in setup_data['defined_container_types']:
                ws_setup.append([
                    cont_type.get('id'), cont_type.get('name', 'N/A'), 
                    cont_type.get('dc'), cont_type.get('wc'), cont_type.get('hc')
                ]) # Assuming these are already floats or will be handled by Excel
                # If these also need specific formatting, apply similarly to above:
                # cell = ws_setup.cell(row=ws_setup.max_row, column=3, value=float(cont_type.get('dc')))
                # cell.number_format = '0.00' # etc. for wc, hc

            # Bold the headers for defined_container_types
            for col_num in range(1, ws_setup.max_column + 1):
                ws_setup.cell(row=row_idx, column=col_num).font = bold_font
            row_idx += len(setup_data['defined_container_types']) + 2 # Increased spacing
        else:
            row_idx +=1 # Minimal increment if no container data


        app.logger.info("Writing box setup to Excel...")
        ws_setup[f'A{row_idx}'] = "Colis à ranger :"
        ws_setup[f'A{row_idx}'].font = bold_font
        row_idx += 1
        box_headers = ["Réf.", "Nom", "Quantité", "Long. (m)", "Larg. (m)", "Haut. (m)", "Poids (kg)", "Charge max dessus (kg)", "Empilable", "Peut être posé dessus", "Orientations"]
        for col_num, header_text in enumerate(box_headers, 1):
            cell = ws_setup.cell(row=row_idx, column=col_num, value=header_text)
            cell.font = bold_font
        
        aggregated_boxes = {}
        for box_inst in setup_data.get('boxes', []):
            orig_id = box_inst.get('original_type_id')
            if orig_id not in aggregated_boxes:
                aggregated_boxes[orig_id] = {
                    "name": box_inst.get('name'),
                    "db": box_inst.get('db'), "wb": box_inst.get('wb'), "hb": box_inst.get('hb'),
                    "weight": box_inst.get('weight'), "lbs": box_inst.get('lbs'),
                    "is_stackable": box_inst.get('is_stackable', True), # Default if not present
                    "can_support_stacking": box_inst.get('can_support_stacking', True), # Default
                    "allowed_orientations": ", ".join(map(str, box_inst.get('allowed_orientations', []))),
                    "quantity": 0
                }
            aggregated_boxes[orig_id]["quantity"] += 1
        
        for orig_id, box_details in aggregated_boxes.items():
            current_row = ws_setup.max_row + 1
            ws_setup.cell(row=current_row, column=1, value=orig_id)
            ws_setup.cell(row=current_row, column=2, value=box_details['name'])
            ws_setup.cell(row=current_row, column=3, value=box_details['quantity']) # Integer
            for col_idx_offset, key in enumerate(['db', 'wb', 'hb', 'weight', 'lbs']):
                cell = ws_setup.cell(row=current_row, column=4 + col_idx_offset, value=float(box_details[key]) if box_details[key] is not None else None)
                cell.number_format = '0.00' # Or '0.0' or 'General' as appropriate
            ws_setup.cell(row=current_row, column=9, value="Oui" if box_details['is_stackable'] else "Non")
            ws_setup.cell(row=current_row, column=10, value="Oui" if box_details['can_support_stacking'] else "Non")
            ws_setup.cell(row=current_row, column=11, value=box_details['allowed_orientations'])
            # row_idx is managed by ws_setup.max_row below
        # Adjust row_idx based on actual rows added if not using ws_setup.max_row
        row_idx = ws_setup.max_row + 2 # Add extra spacing after the table
                

        app.logger.info("Writing GA settings to Excel...")
        ws_setup[f'A{row_idx}'] = "Réglages de la recherche :"
        ws_setup[f'A{row_idx}'].font = bold_font
        row_idx += 1
        ga_s = setup_data.get('ga_settings', {})
        ws_setup.append(["Taille de la population", ga_s.get('npop')])
        ws_setup.append(["Meilleurs plans gardés", ga_s.get('nrep')])
        ws_setup.append(["Fusions par génération", ga_s.get('nmerge')])
        ws_setup.append(["Générations maximum", ga_s.get('max_generations')])
        ws_setup.append(["Temps maximum (secondes)", ga_s.get('max_time_limit')])
        if 'max_containers' in ga_s:
             ws_setup.append(["Nombre de camions maximum", ga_s.get('max_containers')])

        app.logger.info("Creating Packing Results sheet...")
        ws_results = wb.create_sheet("Résultats")
        res_row_idx = 1
        ws_results.cell(row=res_row_idx, column=1, value="Résumé du chargement").font = bold_font
        res_row_idx += 2 # Space after title

        # Headers for the individual container metric tables (will be repeated)
        metric_labels = ["Metric", "Value"]
        solution_type = solution_data.get('type')
        actual_solution_data = solution_data.get('data')

        app.logger.info(f"Solution type: {solution_type}")
        if solution_type == 'single' and actual_solution_data:
            metrics = actual_solution_data.get('metrics', {})
            stowed_summary_str = ", ".join([f"{count} × {name}" for name, count in _summarize_boxes_from_viz(actual_solution_data.get('visualization_data')).items()])
            unstowed_summary_str = _summarize_unstowed_boxes(metrics.get('unstowed_boxes_detailed', [])) or "Aucun"

            container_name_cell = ws_results.cell(row=res_row_idx, column=1, value="Camion 1")
            container_name_cell.font = bold_font
            res_row_idx +=1

            data_to_write = [
                ("Volume chargé (m³)", f"{metrics.get('total_packed_volume', 0):.2f} / {metrics.get('container_volume', 0):.2f}"),
                ("Taux de remplissage", metrics.get('overall_volume_utilization', 'N/A')),
                ("Colis rangés", stowed_summary_str),
                ("Colis non rangés", unstowed_summary_str),
                ("Temps de calcul (s)", f"{metrics.get('time_taken_seconds', 0):.3f}")
            ]

            for label, value in data_to_write:
                label_cell = ws_results.cell(row=res_row_idx, column=1, value=label)
                label_cell.font = bold_font
                value_cell = ws_results.cell(row=res_row_idx, column=2, value=value)
                # Apply number format if the value is purely numeric and needs it
                if label == "Volume chargé (m³)": # This is a string now, formatting is tricky
                    pass # Keep as string
                elif label == "Temps de calcul (s)":
                    try: value_cell.value = float(value); value_cell.number_format = '0.000'
                    except ValueError: pass # Keep as string if not float
                res_row_idx += 1
            
            res_row_idx += 1 # Extra space before image placeholder
            # Placeholder for image:
            image_anchor_cell = f'A{res_row_idx}'
            image_data_base64 = actual_solution_data.get('visualization_image_base64')

            if image_data_base64:
                try:
                    img_str = image_data_base64.split(',')[1]
                    img_bytes = base64.b64decode(img_str)
                    img = Image(io.BytesIO(img_bytes))
                    # You might want to set img.width and img.height here if you want to control
                    # the displayed size in Excel, though Excel will also scale.
                    # e.g., img.width = 480; img.height = 360 (pixels)
                    ws_results.add_image(img, image_anchor_cell)
                    # Adjust row heights for the image area (approximate)
                    # An image of 600px height might need ~450 points. 450/15 rows = 30 points/row.
                    for r_idx in range(res_row_idx, res_row_idx + 15): # For the 15 allocated rows
                        ws_results.row_dimensions[r_idx].height = 30 
                    app.logger.info(f"Image embedded for single container at {image_anchor_cell}")
                except Exception as img_e:
                    app.logger.error(f"Error embedding image for single container: {img_e}")
                    ws_results.merge_cells(start_row=res_row_idx, start_column=1, end_row=res_row_idx + 1, end_column=6)
                    merged_cell_error = ws_results.cell(row=res_row_idx, column=1, value="[Erreur lors de l'affichage de l'image]")
                    merged_cell_error.alignment = Alignment(horizontal='center', vertical='center')
                    merged_cell_error.font = Font(italic=True, color="FF0000") # Red
            else:
                ws_results.merge_cells(start_row=res_row_idx, start_column=1, end_row=res_row_idx + 1, end_column=6)
                merged_cell_image_placeholder = ws_results.cell(row=res_row_idx, column=1, value="[Image du chargement indisponible]")
                merged_cell_image_placeholder.alignment = Alignment(horizontal='center', vertical='center')
                merged_cell_image_placeholder.font = Font(italic=True, color="808080")
            res_row_idx += 15 # Allocate ~15 rows for the image placeholder area
        elif solution_type == 'multi' and actual_solution_data:
            # res_row_idx is already after the "Résumé du chargement" title
            for i, container_res in enumerate(actual_solution_data):
                metrics = container_res.get('metrics', {})
                stowed_summary_str = ", ".join([f"{count} × {name}" for name, count in _summarize_boxes_from_viz(container_res.get('visualization_data')).items()])
                unstowed_summary_str = _summarize_unstowed_boxes(metrics.get('unstowed_boxes_detailed', [])) or "Aucun"

                container_name_cell = ws_results.cell(row=res_row_idx, column=1, value=f"Camion {container_res.get('container_id', i+1)}")
                container_name_cell.font = bold_font
                res_row_idx +=1

                data_to_write_multi = [
                    ("Volume chargé (m³)", f"{metrics.get('total_packed_volume', 0):.2f} / {metrics.get('container_volume', 0):.2f}"),
                    ("Taux de remplissage", metrics.get('overall_volume_utilization', 'N/A')),
                    ("Colis rangés", stowed_summary_str),
                    ("Colis non rangés", unstowed_summary_str),
                    ("Temps de calcul (s)", f"{metrics.get('time_taken_seconds', 0):.3f}")
                ]

                for label, value in data_to_write_multi:
                    label_cell = ws_results.cell(row=res_row_idx, column=1, value=label)
                    label_cell.font = bold_font
                    value_cell = ws_results.cell(row=res_row_idx, column=2, value=value)
                    if label == "Temps de calcul (s)":
                        try: value_cell.value = float(value); value_cell.number_format = '0.000'
                        except ValueError: pass
                    res_row_idx += 1
                
                res_row_idx += 1 # Extra space before image placeholder
                # Placeholder for image:
                image_anchor_cell_multi = f'A{res_row_idx}'
                image_data_base64_multi = container_res.get('visualization_image_base64')

                if image_data_base64_multi:
                    try:
                        img_str_multi = image_data_base64_multi.split(',')[1]
                        img_bytes_multi = base64.b64decode(img_str_multi)
                        img_multi = Image(io.BytesIO(img_bytes_multi))
                        # img_multi.width = 480; img_multi.height = 360;
                        ws_results.add_image(img_multi, image_anchor_cell_multi)
                        for r_idx in range(res_row_idx, res_row_idx + 15):
                             ws_results.row_dimensions[r_idx].height = 30
                        app.logger.info(f"Image embedded for multi-container {i+1} at {image_anchor_cell_multi}")
                    except Exception as img_e_multi:
                        app.logger.error(f"Error embedding image for multi-container {i+1}: {img_e_multi}")
                        ws_results.merge_cells(start_row=res_row_idx, start_column=1, end_row=res_row_idx + 1, end_column=6)
                        merged_cell_img_err = ws_results.cell(row=res_row_idx, column=1, value=f"[Erreur d'image pour le camion {container_res.get('container_id', i+1)}]")
                        merged_cell_img_err.alignment = Alignment(horizontal='center', vertical='center')
                        merged_cell_img_err.font = Font(italic=True, color="FF0000")
                else:
                    ws_results.merge_cells(start_row=res_row_idx, start_column=1, end_row=res_row_idx + 1, end_column=6)
                    merged_cell_image_placeholder_multi = ws_results.cell(row=res_row_idx, column=1, value=f"[Image du camion {container_res.get('container_id', i+1)} indisponible]")
                    merged_cell_image_placeholder_multi.alignment = Alignment(horizontal='center', vertical='center')
                    merged_cell_image_placeholder_multi.font = Font(italic=True, color="808080")
                res_row_idx += 15 # Allocate ~15 rows for image placeholder
                res_row_idx += 3 # Add more extra space before next container's data or "Generated At"
        
        ws_results.cell(row=res_row_idx, column=1, value="Fichier créé le :").font = bold_font
        ws_results.cell(row=res_row_idx, column=2, value=now.strftime("%Y-%m-%d %H:%M:%S")) # This line is correct
        for sheet_name in wb.sheetnames:
            app.logger.info(f"Auto-sizing columns for sheet: {sheet_name}")
            ws = wb[sheet_name]
            for col in ws.columns:
                max_length = 0; column_letter = get_column_letter(col[0].column)
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length: max_length = len(str(cell.value))
                    except: pass
                ws.column_dimensions[column_letter].width = (max_length + 2)

        app.logger.info(f"Attempting to save Excel file to: {filepath}")
        wb.save(filepath)
        app.logger.info(f"Results saved to: {filepath}")
        download_url = url_for('download_result', date_str=date_str, client=safe_client_name, filename=filename)
        return jsonify({"message": f"Results saved successfully to server path: {filepath}", "filename": filename, "download_url": download_url})
    except Exception as e:
        app.logger.error(f"Error in /save_shipment_results: {e}\n{traceback.format_exc()}")
        # Ensure a JSON response even if traceback is complex
        return jsonify({"error": "Failed to save results: " + str(e), "trace": "See server logs for full traceback."}), 500

@app.route('/export_report', methods=['POST'])
def export_report_route():
    """Builds the themed Excel report (report.py), archives it and returns its download URL."""
    try:
        payload = request.get_json()
        if not payload or not payload.get('views'):
            return jsonify({"error": "Aucun plan de chargement à exporter."}), 400
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        folder_name = "Rapports"
        folder = os.path.join(RESULTS_BASE_DIR, date_str, folder_name)
        os.makedirs(folder, exist_ok=True)
        filename = f"plan_de_chargement_{now.strftime('%H%M%S')}.xlsx"
        report.build_report(payload, now).save(os.path.join(folder, filename))
        return jsonify({"filename": filename,
                        "download_url": url_for('download_result', date_str=date_str, client=folder_name, filename=filename)})
    except Exception as e:
        app.logger.error(f"Error in /export_report: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Impossible de créer le rapport : {e}"}), 500

@app.route('/results/<date_str>/<client>/<filename>')
def download_result(date_str, client, filename):
    folder = os.path.join(RESULTS_BASE_DIR, date_str, client)
    if not os.path.abspath(folder).startswith(os.path.abspath(RESULTS_BASE_DIR)):
        abort(404)
    return send_from_directory(folder, filename, as_attachment=True)

def _summarize_boxes_from_viz(visualization_data):
    summary = {}
    if visualization_data and visualization_data.get('layers'):
        for layer in visualization_data['layers']:
            for packed_box in layer.get('boxes', []):
                key = packed_box.get('name') or packed_box.get('original_type_id') or f"Instance ID {packed_box.get('id')}"
                summary[key] = summary.get(key, 0) + 1
    return summary

def _summarize_unstowed_boxes(unstowed_boxes_detailed):
    summary = {}
    if unstowed_boxes_detailed:
        for box in unstowed_boxes_detailed:
            key = box.get('name') or box.get('original_type_id') or f"Instance ID {box.get('id')}"
            summary[key] = summary.get(key, 0) + 1
    return ", ".join([f"{count} × {name}" for name, count in summary.items()])


if __name__ == '__main__':
    # Local development only. Online, gunicorn serves the app (see render.yaml) and this block does not run.
    # The debugger stays on this computer (127.0.0.1) because it can run code.
    app.run(debug=True, host='127.0.0.1', port=int(os.environ.get('PORT', 5000)))
