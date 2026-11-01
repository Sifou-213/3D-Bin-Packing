import numpy as np

# Default properties (can be overridden by inputs from the web app)
DEFAULT_PALLET_PROPS = {
    "length": 1.2,
    "width": 1.0,
    "empty_height": 0.15,
    "max_load_height": 1.65,
    "max_weight": 1000
}

BOX_COLORS = ["#FF6347", "#4682B4", "#32CD32", "#FFD700", "#6A5ACD", "#FF69B4", "#00CED1", "#ADFF2F", "#FFA07A", "#20B2AA"]

def _get_box_color(index):
    return BOX_COLORS[index % len(BOX_COLORS)]

def _get_oriented_dimensions(base_dims, orientation_code):
    L, W, H = base_dims
    if orientation_code == 1: return (L, W, H)
    if orientation_code == 2: return (W, L, H)
    if orientation_code == 3: return (L, H, W)
    if orientation_code == 4: return (H, L, W)
    if orientation_code == 5: return (W, H, L)
    if orientation_code == 6: return (H, W, L)
    return (L, W, H)

def _can_place_box_simplified(box_to_place_details, pos, current_total_weight_on_pallet, existing_placements, pallet_details):
    """
    Simplified check based on the provided example script.
    box_to_place_details includes: "dims", "weight", "lbs", "stackable", "can_support_stacking"
    """
    x, y, z = pos
    l, w, h = box_to_place_details["dims"]
    tol = 1e-5

    # 1. Check pallet boundaries
    # Max height for goods is measured from pallet surface (empty_height)
    # So, top of the box (z + h) must be <= pallet_details["empty_height"] + pallet_details["max_load_height"]
    effective_max_pallet_height = pallet_details["empty_height"] + pallet_details["max_load_height"]
    
    if x < -tol or (x + l) > pallet_details["length"] + tol or \
       y < -tol or (y + w) > pallet_details["width"] + tol or \
       z < pallet_details["empty_height"] - tol or (z + h) > effective_max_pallet_height + tol:
        return False

    # 2. Check total weight limit for the pallet
    if current_total_weight_on_pallet + box_to_place_details["weight"] > pallet_details["max_weight"] + tol:
        return False

    # 3. Check for overlaps with already placed boxes (RE-INTRODUCED FOR CORRECTNESS)
    for p_idx, p_existing in enumerate(existing_placements):
        px, py, pz_existing = p_existing["pos"]
        pl_existing, pw_existing, ph_existing = p_existing["dims"]

        # AABB collision detection
        # Check if the new box (x,y,z,l,w,h) overlaps with p_existing
        overlap_x = (x < px + pl_existing - tol) and (x + l > px - tol)
        overlap_y = (y < py + pw_existing - tol) and (y + w > py - tol)
        overlap_z = (z < pz_existing + ph_existing - tol) and (z + h > pz_existing - tol)

        if overlap_x and overlap_y and overlap_z:
            # print(f"Overlap detected: New box {box_to_place_details['dims']} at {(x,y,z)} overlaps with existing {p_existing['dims']} at {p_existing['pos']}")
            return False # Collision detected

    # 4. Check for support from below (if not on pallet floor)
    if z > pallet_details["empty_height"] + tol:
        supported = False
        for p_idx, p_box in enumerate(existing_placements):
            px, py, pz = p_box["pos"]
            pl, pw, ph = p_box["dims"]

            # Check if the candidate box is directly on top of p_box and perfectly aligned (as per example)
            if np.isclose(px, x, atol=tol) and \
               np.isclose(py, y, atol=tol) and \
               np.isclose(pz + ph, z, atol=tol) and \
               np.isclose(pl, l, atol=tol) and \
               np.isclose(pw, w, atol=tol) : # Added dimension match for perfect alignment support

                if not p_box.get("stackable", False): # Box below must allow items on top
                    return False
                if box_to_place_details["weight"] > p_box.get("lbs", 0) + tol: # Check LBS of box below
                    return False
                if not p_box.get("can_support_stacking", False): # Box below must be marked as "can support stacking"
                     return False
                supported = True
                break 
        if not supported:
            return False # Not on pallet floor and no valid support found

    return True


def solve_pallet_loading_problem(pallet_props_input, boxes_props_input):
    placements = []
    current_total_weight_on_pallet = 0
    total_packed_volume = 0
    
    pallet_details = DEFAULT_PALLET_PROPS.copy()
    pallet_details.update(pallet_props_input)

    unpacked_boxes_map = {} 
    
    # Prepare box type data with all properties needed
    processed_box_types = []
    for type_idx, box_type_input in enumerate(boxes_props_input):
        original_type_id = box_type_input.get('id', f'boxType_{type_idx}')
        base_dims = box_type_input.get("base_dimensions")
        if not base_dims or len(base_dims) != 3:
            print(f"Warning: Skipping box type {original_type_id} due to invalid base_dimensions.")
            continue

        processed_box_types.append({
            "original_type_id": original_type_id,
            "name": box_type_input.get("name", f"Type_{original_type_id}"),
            "quantity_to_pack": box_type_input.get("quantity", 0),
            "quantity_packed_count": 0, # Counter for this type
            "base_dims": tuple(base_dims),
            "allowed_orientations_codes": box_type_input.get("allowed_orientations", [1]),
            "weight": box_type_input.get("weight", 0),
            "lbs": box_type_input.get("lbs", float('inf')),
            "stackable": box_type_input.get("stackable", True), # Can items be placed ON this box
            "can_support_stacking": box_type_input.get("can_support_stacking", True), # Is this box suitable to be part of a stack (e.g. support others / be supported)
            "color": _get_box_color(type_idx)
        })
        unpacked_boxes_map[original_type_id] = {
            "name": box_type_input.get("name", f"Type_{original_type_id}"),
            "quantity_unpacked": box_type_input.get("quantity", 0),
            "original_type_id": original_type_id
        }

    # Simplified greedy packing based on the example script
    effective_max_pallet_load_height = pallet_details["empty_height"] + pallet_details["max_load_height"]

    for box_type in processed_box_types:
        if box_type["quantity_to_pack"] == 0:
            continue

        for orientation_code in box_type["allowed_orientations_codes"]:
            if box_type["quantity_packed_count"] >= box_type["quantity_to_pack"]:
                break # All instances of this type are packed

            oriented_dims = _get_oriented_dimensions(box_type["base_dims"], orientation_code)
            l, w, h = oriented_dims
            if h < 1e-5 : continue # Skip zero height orientations

            current_z = pallet_details["empty_height"]
            while current_z + h <= effective_max_pallet_load_height + 1e-5 :
                if box_type["quantity_packed_count"] >= box_type["quantity_to_pack"]: break

                current_y = 0
                while current_y + w <= pallet_details["width"] + 1e-5 :
                    if box_type["quantity_packed_count"] >= box_type["quantity_to_pack"]: break
                    
                    current_x = 0
                    while current_x + l <= pallet_details["length"] + 1e-5 :
                        if box_type["quantity_packed_count"] >= box_type["quantity_to_pack"]:
                            break
                        
                        pos_to_try = (current_x, current_y, current_z)
                        
                        candidate_box_details = {
                            "dims": oriented_dims,
                            "weight": box_type["weight"],
                            "lbs": box_type["lbs"], # LBS of this box (if something is stacked on it)
                            "stackable": box_type["stackable"], # If this box can have items on it
                            "can_support_stacking": box_type["can_support_stacking"] # If this box is suitable for stacking
                        }

                        if _can_place_box_simplified(candidate_box_details, pos_to_try, current_total_weight_on_pallet, placements, pallet_details):
                            instance_id = f"{box_type['original_type_id']}_inst{box_type['quantity_packed_count'] + 1}"
                            placement_data = {
                                "id": instance_id,
                                "original_type_id": box_type["original_type_id"],
                                "name": box_type["name"],
                                "pos": pos_to_try,
                                "dims": oriented_dims,
                                "color": box_type["color"],
                                "weight": box_type["weight"],
                                "lbs": box_type["lbs"],
                                "stackable": box_type["stackable"],
                                "can_support_stacking": box_type["can_support_stacking"],
                                "orientation_used": orientation_code
                            }
                            placements.append(placement_data)
                            current_total_weight_on_pallet += box_type["weight"]
                            total_packed_volume += l * w * h
                            box_type["quantity_packed_count"] += 1
                            unpacked_boxes_map[box_type["original_type_id"]]["quantity_unpacked"] -=1
                        
                        current_x += l # Critical: advance by full length of current box orientation
                    current_y += w # Critical: advance by full width
                current_z += h # Critical: advance by full height
    
    # Metrics Calculation
    num_boxes_input_total = sum(bt.get("quantity",0) for bt in boxes_props_input)
    num_boxes_packed = len(placements)
    num_boxes_unpacked = num_boxes_input_total - num_boxes_packed
    
    pallet_load_volume = pallet_details["length"] * pallet_details["width"] * pallet_details["max_load_height"]
    volume_utilization = (total_packed_volume / pallet_load_volume) * 100 if pallet_load_volume > 1e-5 else 0

    final_unpacked_list = [val for val in unpacked_boxes_map.values() if val["quantity_unpacked"] > 0]

    metrics = {
        "num_boxes_input": num_boxes_input_total,
        "num_boxes_packed": num_boxes_packed,
        "num_boxes_unpacked": num_boxes_unpacked,
        "total_packed_weight": round(current_total_weight_on_pallet, 2),
        "total_packed_volume": round(total_packed_volume, 3),
        "pallet_load_volume": round(pallet_load_volume, 3),
        "volume_utilization_percent": round(volume_utilization, 2),
        "max_pallet_weight": pallet_details["max_weight"],
        "pallet_length": pallet_details["length"],
        "pallet_width": pallet_details["width"],
        "pallet_max_load_height": pallet_details["max_load_height"], # Max height of goods
        "pallet_empty_height": pallet_details["empty_height"] # Pallet's own height
    }

    return placements, metrics, pallet_details, final_unpacked_list
