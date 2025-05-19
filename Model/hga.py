import copy
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import copy
import time 


class hybrid_GA:

    # Initializing the parameters that will be used in the GA:
    npop = 50 # Population size
    nrep = 10 # Number of solutions to be reproduced per generation
    pcross = 0.67
    pmut = 0.33
    nmerge = 10 # Number of solutions to be generated through merger mutation per generation

    # Box data:
    class box:
        def __init__(self, id, db, wb, hb, vb, allowed_orientations=None): # Added allowed_orientations
            self.id = id
            self.db = db
            self.wb = wb
            self.hb = hb
            self.vb = vb
            # If allowed_orientations is None or empty, default to all 6
            self.allowed_orientations = allowed_orientations if allowed_orientations and isinstance(allowed_orientations, list) and all(isinstance(x, int) for x in allowed_orientations) else list(range(1, 7))

    class rv:
        def __init__(self, rotation_id):
            self.rotation_id = rotation_id  # 1 to 6, as defined in the passage

    class box_placing:
        def __init__(self, id, ox, oy, oz, rotation_variant): # rotation_variant is an rv object
             self.id = id
             self.ox = ox
             self.oy = oy
             self.oz = oz
             self.rv = rotation_variant


    # Container data:
    class container:
        def __init__(self, dc, wc, hc, vc):
            self.dc = dc
            self.wc = wc
            self.hc = hc
            self.vc = vc

    class layer:
        def __init__(self, d, vutil, Pl):
            """
            Represents a single filled layer in the container.
            Args:
                d: Layer depth (x-dimension determined by the layer-defining box).
                vutil: Layer utilization (volume of boxes / layer volume).
                Pl: List of box_placing objects (placings in this layer).
            """
            self.d = d                  # Layer depth (xc)
            self.vutil = vutil          # Volume utilization (0.0 to 1.0)
            self.Pl = Pl                # List of box_placing instances
            self.npl = len(Pl)          # Number of boxes in layer

        def B(self):
            """Returns the set of box IDs in this layer."""
            return {pl.id for pl in self.Pl}


    class stowage_plan:
        def __init__(self, L, Bfree, xcfree, container):
            """
            Represents a complete or partial packing solution.
            Args:
                L: List of layer objects (filled layers).
                Bfree: Set of remaining unstowed box IDs.
                xcfree: Remaining container depth (x-dimension).
            """
            self.L = L                  # List of layers
            self.nl = len(L)            # Number of layers
            self.Bfree = Bfree          # Set of unstowed box IDs (e.g., {1, 3, 5})
            self.xcfree = xcfree        # Free container depth (xc remaining)
            self.v = 0                  # Packed volume, will be calculated externally
            self.container = container
            self.fitness = 0.0          # Initialize fitness

    # Generating npop stowage plans as the start generation using the basic heuristic (variant start):
    def create_empty_plan(self):
        """Return an empty stowage plan with all boxes free and full container depth."""
        return self.stowage_plan(
            container=self.container,
            L=[],
            Bfree={b.id for b in self.box_list}, # Corrected to use b.id from self.box_list
            xcfree=self.container.dc
        )

    # generate variant-dependent layer definining list Ldef1 for the first (additional) layer:
    def generate_Ldefn(self, s):
        """
        Generate a sorted list of feasible layer definitions (Ldef1) for the next layer.
        Args:
            s: Current stowage plan (with Bfree and xcfree).
        Returns:
            List of feasible layer definitions, sorted by:
            1. Volume of ldb (descending),
            2. x-dimension of ldb (descending),
            3. z-dimension of ldb (descending).
            nLdef1: The count of feasible layer definitions (|Ldef1|)
        """
        Ldef1 = []
        # safety_counter = 0  # Prevent infinite loops - This was not used

        tol = 1e-6 # Tolerance for floating point comparisons

        for box_id in s.Bfree:
            # Assuming box_list is 0-indexed and box_id is 1-indexed
            box_obj = self.box_list[box_id - 1]
            # box_obj = self.box_list[box_id - 1] # Old way
            box_obj = self.boxes_by_id.get(box_id)
            if not box_obj:
                # print(f"Warning: Box ID {box_id} from Bfree not found in boxes_by_id map during generate_Ldefn.")
                continue
            for rotation_id in range(1, 7):  # Check all 6 rotation variants
                if rotation_id not in box_obj.allowed_orientations: # New check
                    continue
                # print(f"→ Inside generate_Ldefn → Getting rotation for box {box_obj.id}")
                rotated_dims = self.get_rotated_dimensions(box_obj, rotation_id)
                dx_rot, dy_rot, dz_rot = rotated_dims  # dx_rot = depth, dy_rot = width, dz_rot = height for the layer
                # print(f"✅ Got rotated dimensions for box {box_obj.id}: {rotated_dims}")

                if dx_rot <= s.xcfree + tol and \
                   dy_rot <= self.container.wc + tol and \
                   dz_rot <= self.container.hc + tol:
                    Ldef1.append({
                        'ldb': box_obj,
                        'rvldb': self.rv(rotation_id),
                        'dx': dx_rot,    # This will be the depth of the layer
                        'dz': dz_rot     # Used for sorting criteria (original height in this rotation)
                    })

        nLdef1 = len(Ldef1)

        Ldef1_sorted = sorted(
            Ldef1,
            key=lambda x: (-x['ldb'].vb, -x['dx'], -x['dz'])
        )

        return Ldef1_sorted, nLdef1

    def basic_heuristic(self, sIn, variant, ns):

        # 1. Determine qldb parameters based on nbtype
        if self.nbtype <= 20:
            qldb1, qldb2, qldb3 = 100, 30, 33
        elif 20 < self.nbtype <= 50:
            qldb1, qldb2, qldb3 = 30, 10, 10
        elif 50 < self.nbtype <= 70:
            qldb1, qldb2, qldb3 = 10, 10, 10
        else:  # nbtype > 70
            qldb1, qldb2, qldb3 = 5, 5, 5

        # 2. Generate feasible layer definitions
        Ldef1_sorted, nLdef1 = self.generate_Ldefn(sIn)

        SGen = set()

        if nLdef1 > 0:
            definitions_to_use = []
            if variant == 'start':
                definitions_to_use = Ldef1_sorted
            elif variant == 'crossover':
                cutoff = max(1, int(nLdef1 * qldb1 / 100))
                definitions_to_use = Ldef1_sorted[:cutoff]
            elif variant == 'mutation':
                cutoff = max(1, int(nLdef1 * qldb3 / 100))
                if cutoff > 0 and Ldef1_sorted: # Ensure cutoff and list are valid
                    selected_idx = random.randint(0, cutoff - 1)
                    definitions_to_use = [Ldef1_sorted[selected_idx]]
                elif Ldef1_sorted: # Fallback if cutoff is problematic
                    definitions_to_use = [Ldef1_sorted[0]]


            for ldef in definitions_to_use:
                s = copy.deepcopy(sIn)
                l = self.fill_layer(
                    ldb=ldef['ldb'],
                    rvldb=ldef['rvldb'],
                    d=ldef['dx'],
                    Bfree=s.Bfree.copy(), # Pass a copy of Bfree
                    current_s=s
                )

                if l is None or not l.Pl: # If layer filling failed or layer is empty
                    continue

                # Update stowage plan
                s.nl += 1
                s.L.append(l)
                s.xcfree -= l.d
                s.Bfree -= l.B() # Corrected: Use layer's B() method
                # SGen.add(s) # SGen should store completed plans

                while True:
                    # print("→ Inside basic_heuristic → Calling generate_Ldefn for subsequent layers")
                    Ldefn_sorted, nLdefn = self.generate_Ldefn(s)

                    if nLdefn == 0:
                        break

                    definitions_to_process = []
                    if variant == 'start':
                        definitions_to_process = Ldefn_sorted[:1]
                    elif variant == 'crossover':
                        cutoff = max(1, int(nLdefn * qldb2 / 100))
                        definitions_to_process = Ldefn_sorted[:cutoff]
                    elif variant == 'mutation':
                        cutoff = max(1, int(nLdefn * qldb3 / 100))
                        if cutoff > 0 and Ldefn_sorted:
                            selected_idx = random.randint(0, cutoff - 1)
                            definitions_to_process = [Ldefn_sorted[selected_idx]]
                        elif Ldefn_sorted:
                             definitions_to_process = [Ldefn_sorted[0]]


                    Lnext_layers_filled = [] # Store successfully filled layers
                    for ldefn_next in definitions_to_process:
                        l_filled = self.fill_layer(
                            ldb=ldefn_next['ldb'],
                            rvldb=ldefn_next['rvldb'],
                            d=ldefn_next['dx'],
                            Bfree=s.Bfree.copy(), # Pass a copy
                            current_s=s
                        )
                        if l_filled and l_filled.Pl: # If layer filling was successful
                            Lnext_layers_filled.append(l_filled)

                    if not Lnext_layers_filled:
                        break # No suitable layer could be formed

                    lnext = max(Lnext_layers_filled, key=lambda layer_obj: layer_obj.vutil)

                    s.nl += 1
                    s.L.append(lnext)
                    s.xcfree -= lnext.d
                    s.Bfree -= lnext.B() # Corrected

                # Calculate total volume for the completed plan s
                # s.v = sum(self.box_list[pl.id-1].vb for layer_obj in s.L for pl in layer_obj.Pl) # Old way
                s.v = sum(self.boxes_by_id[pl.id].vb for layer_obj in s.L for pl in layer_obj.Pl)

                plan_sig = tuple(sorted((pl.id, pl.ox, pl.oy, pl.oz, pl.rv.rotation_id)
                                for layer_obj in s.L for pl in layer_obj.Pl))
                if plan_sig not in self._generated_signatures:
                    self._generated_signatures.add(plan_sig)
                    SGen.add(s)

            if SGen:
                top_plans = sorted(list(SGen), key=lambda sol_s: -sol_s.v)[:ns]
                SGen = set(top_plans)
            else: # If SGen is empty but sIn was processed
                # sIn.v = sum(self.box_list[pl.id-1].vb for layer_obj in sIn.L for pl in layer_obj.Pl) if sIn.L else 0 # Old way
                sIn.v = sum(self.boxes_by_id[pl.id].vb for layer_obj in sIn.L for pl in layer_obj.Pl) if sIn.L else 0
                SGen = {sIn}

        else: # No feasible first layers (nLdef1 == 0)
            # sIn.v = sum(self.box_list[pl.id-1].vb for layer_obj in sIn.L for pl in layer_obj.Pl) if sIn.L else 0 # Old way
            sIn.v = sum(self.boxes_by_id[pl.id].vb for layer_obj in sIn.L for pl in layer_obj.Pl) if sIn.L else 0
            SGen = {sIn}

        return SGen


    def get_rotated_dimensions(self, box_obj, rotation_id): # Changed box to box_obj for clarity
        # print(f"→ Inside get_rotated_dimensions → Box ID: {box_obj.id}, Rotation ID: {rotation_id}")
        if rotation_id == 1:
            return (box_obj.db, box_obj.wb, box_obj.hb)
        elif rotation_id == 2:
            return (box_obj.db, box_obj.hb, box_obj.wb)
        elif rotation_id == 3:
            return (box_obj.wb, box_obj.db, box_obj.hb)
        elif rotation_id == 4:
            return (box_obj.wb, box_obj.hb, box_obj.db)
        elif rotation_id == 5:
            return (box_obj.hb, box_obj.db, box_obj.wb)
        elif rotation_id == 6:
            return (box_obj.hb, box_obj.wb, box_obj.db)
        else:
            raise ValueError(f"Invalid rotation_id: {rotation_id}. Must be 1-6.")


    def fill_layer(self, ldb, rvldb, d, Bfree, current_s):
        l = self.layer(d=d, vutil=0.0, Pl=[])
        SStack = []
        daughter_placing = None # Initialize
        stype = None            # Initialize

        # Calculate the absolute starting X-coordinate of the current layer
        # This assumes current_s.xcfree is the remaining depth *before* this layer is placed.
        # And current_s.container.dc is the total container depth.
        layer_abs_x_start = current_s.container.dc - current_s.xcfree

        layer_residual = {
            'dimensions': (d, self.container.wc, self.container.hc), # Layer depth 'd', full container width and height
            'corner': (layer_abs_x_start, 0, 0), # Layer starts at this absolute X, and Y=0, Z=0
            'placed_box': ldb, # LDB is placed in the root residual space of the layer
            'placed_rotation': rvldb
        }
        SStack.append(layer_residual)

        updated_Bfree = Bfree - {ldb.id} # Remove LDB from free boxes for this layer attempt

        # 'd' is the layer_depth_d for this layer

        while SStack:
            scurr = SStack.pop()

            if scurr['placed_box'] is None: # If space is empty, try to fill it
                fit_found = any( # Check if any box in Bfree can fit in scurr in any rotation
                    any(
                        (dx_rot <= scurr['dimensions'][0] + 1e-6 and # Add tolerance
                         dy_rot <= scurr['dimensions'][1] + 1e-6 and
                         dz_rot <= scurr['dimensions'][2] + 1e-6)
                        for dx_rot, dy_rot, dz_rot in ( # Iterate through rotations for each box
                            self.get_rotated_dimensions(self.boxes_by_id[box_id], rot_id) # Use boxes_by_id
                            for rot_id in range(1, 7)
                        )
                    )
                    for box_id in updated_Bfree
                )

                if fit_found:
                    # Pass layer_abs_x_start and layer depth 'd' for boundary checks
                    current_placement_tuple, daughter_placement_tuple, _ = self.determine_placings(
                        scurr, updated_Bfree, self.current_generation, self.total_generations,
                        layer_abs_x_start, d
                    )

                    if current_placement_tuple:
                        bcurr, rvcurr = current_placement_tuple
                        scurr['placed_box'] = bcurr
                        scurr['placed_rotation'] = rvcurr
                        updated_Bfree.discard(bcurr.id) # Mark as used within this layer attempt

                    if daughter_placement_tuple:
                        bsucc, _, temp_stype = daughter_placement_tuple
                        stype = temp_stype # stype for arranging bsucc relative to bcurr
                        daughter_placing = daughter_placement_tuple
                        updated_Bfree.discard(bsucc.id) # Mark as used
                    else:
                        daughter_placing = None
                        stype = None
                # else: No box fits, scurr remains empty. It won't generate daughters.

            # Process the space if it's now filled (either LDB or by determine_placings)
            if scurr['placed_box'] is not None:
                # Add the placed box to the layer's list of placings
                # Ensure it's not already added (e.g. if a filled space was somehow re-evaluated)
                # A simple check: if this exact placing (id, corner, rotation) is already in l.Pl
                is_already_placed = any(
                    p.id == scurr['placed_box'].id and
                    p.ox == scurr['corner'][0] and
                    p.oy == scurr['corner'][1] and
                    p.oz == scurr['corner'][2] and
                    p.rv.rotation_id == scurr['placed_rotation'].rotation_id
                    for p in l.Pl
                )
                if not is_already_placed:
                    l.Pl.append(
                        self.box_placing(
                            id=scurr['placed_box'].id,
                            ox=scurr['corner'][0],
                            oy=scurr['corner'][1],
                            oz=scurr['corner'][2],
                            rotation_variant=scurr['placed_rotation']
                        )
                    )

                daughter_spaces = self.generate_daughter_spaces(scurr, scurr['placed_box'], scurr['placed_rotation'])
                sabove = daughter_spaces['sabove']
                sinfront = daughter_spaces['sinfront']
                sbeside = daughter_spaces['sbeside']

                # If bsucc (from daughter_placing) was determined for one of the daughter spaces
                if daughter_placing and stype:
                    bsucc_obj, rvsucc_obj, _ = daughter_placing # stype was already captured
                    target_space_for_bsucc = None
                    if stype == 'front': target_space_for_bsucc = sinfront
                    elif stype == 'beside': target_space_for_bsucc = sbeside
                    elif stype == 'above': target_space_for_bsucc = sabove

                    if target_space_for_bsucc:
                        # Before assigning, ensure bsucc actually fits this specific daughter space
                        # and respects absolute boundaries. This check is crucial.
                        # We can call _evaluate_placement for the target_space_for_bsucc with only bsucc_obj.
                        # For simplicity here, we assume determine_placings already validated bsucc for its intended role.
                        # A more robust solution would re-validate bsucc against target_space_for_bsucc here.
                        bsucc_dims = self.get_rotated_dimensions(bsucc_obj, rvsucc_obj.rotation_id)
                        ts_dx, ts_dy, ts_dz = target_space_for_bsucc['dimensions']
                        ts_ox, ts_oy, ts_oz = target_space_for_bsucc['corner']
                        tol = 1e-6

                        fits_locally = (bsucc_dims[0] <= ts_dx + tol and
                                        bsucc_dims[1] <= ts_dy + tol and
                                        bsucc_dims[2] <= ts_dz + tol)

                        if fits_locally:
                            bsucc_abs_end_x = ts_ox + bsucc_dims[0]
                            bsucc_abs_end_y = ts_oy + bsucc_dims[1]
                            bsucc_abs_end_z = ts_oz + bsucc_dims[2]

                            fits_globally = (
                                bsucc_abs_end_x <= layer_abs_x_start + d + tol and
                                bsucc_abs_end_y <= self.container.wc + tol and
                                bsucc_abs_end_z <= self.container.hc + tol and
                                ts_ox >= layer_abs_x_start - tol and
                                ts_oy >= 0 - tol and
                                ts_oz >= 0 - tol
                            )
                            if fits_globally:
                                target_space_for_bsucc['placed_box'] = bsucc_obj
                                target_space_for_bsucc['placed_rotation'] = rvsucc_obj
                            # else: bsucc didn't fit the designated daughter space, it remains empty.
                        # else: bsucc didn't fit locally.

                    daughter_placing = None # Reset after attempting to place bsucc
                    stype = None

                # Add valid daughter spaces to SStack (those with positive dimensions)
                # The order can influence packing; paper suggests specific orders.
                # Using a fixed order: above, then the one with larger base from (infront, beside)
                daughter_spaces_to_process = []
                if sabove['dimensions'][2] > 1e-6: daughter_spaces_to_process.append(sabove)

                # Compare base areas of sinfront and sbeside to decide order
                base_sinfront = sinfront['dimensions'][0] * sinfront['dimensions'][1]
                base_sbeside = sbeside['dimensions'][0] * sbeside['dimensions'][1]

                if base_sinfront >= base_sbeside:
                    if sinfront['dimensions'][0] > 1e-6 and sinfront['dimensions'][1] > 1e-6 : daughter_spaces_to_process.append(sinfront)
                    if sbeside['dimensions'][0] > 1e-6 and sbeside['dimensions'][1] > 1e-6 : daughter_spaces_to_process.append(sbeside)
                else:
                    if sbeside['dimensions'][0] > 1e-6 and sbeside['dimensions'][1] > 1e-6 : daughter_spaces_to_process.append(sbeside)
                    if sinfront['dimensions'][0] > 1e-6 and sinfront['dimensions'][1] > 1e-6 : daughter_spaces_to_process.append(sinfront)

                for new_space in reversed(daughter_spaces_to_process): # LIFO for stack
                    # Final check for negligible spaces before adding to stack
                    if new_space['dimensions'][0] < 1e-6 or \
                       new_space['dimensions'][1] < 1e-6 or \
                       new_space['dimensions'][2] < 1e-6:
                        continue
                    SStack.append(new_space) # No merging logic implemented here for simplicity

        l.npl = len(l.Pl)
        if l.Pl:
            # sum_vol = sum(self.box_list[pl.id-1].vb for pl in l.Pl) # Old way
            sum_vol = sum(self.boxes_by_id[pl.id].vb for pl in l.Pl)
            layer_total_volume = l.d * self.container.wc * self.container.hc
            if layer_total_volume > 1e-6: # Avoid division by zero for very thin/empty layers
                l.vutil = sum_vol / layer_total_volume
            else:
                l.vutil = 0
        else: # No boxes placed in the layer
            l.vutil = 0
            return None # Indicate layer filling failed or resulted in an empty layer

        return l

    def generate_daughter_spaces(self, scurr, bcurr, rvcurr):
        dx_b, dy_b, dz_b = self.get_rotated_dimensions(bcurr, rvcurr.rotation_id)
        px, py, pz = scurr['corner']
        pdx, pdy, pdz = scurr['dimensions']

        sabove = {
            'dimensions': (dx_b, dy_b, max(0, pdz - dz_b)),
            'corner': (px, py, pz + dz_b),
            'placed_box': None,
            'placed_rotation': None
        }

        # Variant 1: "in front of large" (split Y first for sinfront, then X for sbeside)
        # This interpretation might differ from paper's figure, common is to define s_rem_y, s_rem_x
        sinfront_v1 = {
            'dimensions': (dx_b, max(0, pdy - dy_b), pdz), # dx_b is depth of bcurr, use for sinfront depth
            'corner': (px, py + dy_b, pz),
            'placed_box': None, 'placed_rotation': None
        }
        sbeside_v1 = {
            'dimensions': (max(0, pdx - dx_b), pdy, pdz), # pdy is full width for sbeside
            'corner': (px + dx_b, py, pz),
            'placed_box': None, 'placed_rotation': None
        }
        base_v1_sf = sinfront_v1['dimensions'][0] * sinfront_v1['dimensions'][1]
        base_v1_sb = sbeside_v1['dimensions'][0] * sbeside_v1['dimensions'][1]
        max_base_v1 = max(base_v1_sf, base_v1_sb)


        # Variant 2: "beside large" (split X first for sbeside, then Y for sinfront)
        sbeside_v2 = {
            'dimensions': (max(0, pdx - dx_b), dy_b, pdz), # dy_b is width of bcurr, use for sbeside width
            'corner': (px + dx_b, py, pz),
            'placed_box': None, 'placed_rotation': None
        }
        sinfront_v2 = {
            'dimensions': (pdx, max(0, pdy - dy_b), pdz), # pdx is full depth for sinfront
            'corner': (px, py + dy_b, pz),
            'placed_box': None, 'placed_rotation': None
        }
        base_v2_sb = sbeside_v2['dimensions'][0] * sbeside_v2['dimensions'][1]
        base_v2_sf = sinfront_v2['dimensions'][0] * sinfront_v2['dimensions'][1]
        max_base_v2 = max(base_v2_sb, base_v2_sf)


        if max_base_v1 >= max_base_v2:
            selected_sinfront = sinfront_v1
            selected_sbeside = sbeside_v1
        else:
            selected_sinfront = sinfront_v2
            selected_sbeside = sbeside_v2

        return {
            'sabove': sabove,
            'sinfront': selected_sinfront,
            'sbeside': selected_sbeside
        }

    def determine_placings(self, scurr, Bfree, current_generation, total_generations,
                           layer_abs_x_start, layer_depth_d): # Added layer boundary params
        use_rule_31 = (current_generation / total_generations) <= 0.5 if total_generations > 0 else True
        best_placement_info = None
        best_eval_score = (-1, float('inf')) # (volume, priority_metric: smaller is better)

        box_ids = list(Bfree) # Consider only boxes currently free for this layer
        for i in range(len(box_ids)):
            # box1 = self.box_list[box_ids[i]-1] # Old way
            box1 = self.boxes_by_id.get(box_ids[i])
            if not box1: continue
            # Single box placement
            for rot1_id in range(1, 7):
                if rot1_id not in box1.allowed_orientations: # New check
                    continue
                placement = self._evaluate_placement(scurr, box1, None, self.rv(rot1_id), None, use_rule_31,
                                                     layer_abs_x_start, layer_depth_d)
                if placement:
                    current_score = (placement['volume'], placement['priority'])
                    if best_placement_info is None or \
                       current_score[0] > best_eval_score[0] or \
                       (abs(current_score[0] - best_eval_score[0]) < 1e-6 and current_score[1] < best_eval_score[1]):
                        best_eval_score = current_score
                        best_placement_info = placement
            # Paired box placement
            for j in range(i + 1, len(box_ids)):
                box2 = self.boxes_by_id.get(box_ids[j]) # Fetch box2 once for each j
                if not box2: continue
                
                for rot1_id in range(1, 7):
                    if rot1_id not in box1.allowed_orientations: # New check for box1
                        continue
                    for rot2_id in range(1, 7):
                        if rot2_id not in box2.allowed_orientations: # New check for box2
                            continue
                        placement = self._evaluate_placement(scurr, box1, box2, self.rv(rot1_id), self.rv(rot2_id), use_rule_31,
                                                             layer_abs_x_start, layer_depth_d)
                        if placement:
                            current_score = (placement['volume'], placement['priority'])
                            if best_placement_info is None or \
                               current_score[0] > best_eval_score[0] or \
                               (abs(current_score[0] - best_eval_score[0]) < 1e-6 and current_score[1] < best_eval_score[1]):
                                best_eval_score = current_score
                                best_placement_info = placement

        if not best_placement_info:
            return None, None, None # No valid placement found

        # Apply R4 allocation: Determine bcurr (current) and bsucc (successor for daughter space)
        b1_obj, b2_obj = best_placement_info['box1'], best_placement_info.get('box2')
        rv1_obj, rv2_obj = best_placement_info['rot1'], best_placement_info.get('rot2')
        stype = best_placement_info['stype']
        dims1 = best_placement_info['dims1'] # Rotated dims of b1_obj
        dims2 = best_placement_info.get('dims2', (0,0,0)) # Rotated dims of b2_obj

        current_placement_tuple = None  # (box_object, rv_object) for scurr
        daughter_placement_tuple = None # (box_object, rv_object, stype_for_daughter) for daughter space

        if stype == 'single' or not b2_obj:
            current_placement_tuple = (b1_obj, rv1_obj)
        elif stype in ['front', 'beside']: # R4.1
            # Place the taller box in scurr (bcurr), shorter in daughter (bsucc)
            if dims1[2] >= dims2[2] - 1e-6: # Compare rotated heights (dim[2])
                current_placement_tuple = (b1_obj, rv1_obj)
                daughter_placement_tuple = (b2_obj, rv2_obj, stype)
            else:
                current_placement_tuple = (b2_obj, rv2_obj)
                daughter_placement_tuple = (b1_obj, rv1_obj, stype)
        elif stype == 'above': # R4.2
            # Place box with larger base area in scurr (bcurr), smaller base in daughter (bsucc)
            base1_area = dims1[0] * dims1[1] # Rotated depth * rotated width
            base2_area = dims2[0] * dims2[1]
            if base1_area >= base2_area - 1e-6:
                current_placement_tuple = (b1_obj, rv1_obj) # b1 is bcurr (bottom)
                daughter_placement_tuple = (b2_obj, rv2_obj, stype) # b2 is bsucc (top)
            else:
                current_placement_tuple = (b2_obj, rv2_obj) # b2 is bcurr (bottom)
                daughter_placement_tuple = (b1_obj, rv1_obj, stype) # b1 is bsucc (top)
        
        # daughter_dims from best_placement_info is not directly used by fill_layer's current logic
        # for creating daughter spaces, as generate_daughter_spaces recalculates them.
        return current_placement_tuple, daughter_placement_tuple, best_placement_info.get('daughter_dims')


    def _spaces_can_merge(self, s1, s2):
        # Simplified: not implemented for brevity, assume no merging for now
        return False

    def _merge_spaces(self, s1, s2):
        # Simplified: not implemented
        return s1 # Or s2, or a new combined space dict

    def _generate_rotation_combos(self, box1, box2=None):
        # This method seems unused if _evaluate_placement handles rotations internally
        pass

    def _evaluate_placement(self, scurr, box1, box2, rv1, rv2, use_rule_31,
                            layer_abs_x_start, layer_depth_d): # Added layer boundary params
        dims1 = self.get_rotated_dimensions(box1, rv1.rotation_id)
        dims2 = self.get_rotated_dimensions(box2, rv2.rotation_id) if box2 else (0,0,0)
        s_dx, s_dy, s_dz = scurr['dimensions'] # Dimensions of the residual space
        s_ox, s_oy, s_oz = scurr['corner']     # Absolute origin of the residual space
        tol = 1e-6                             # Tolerance for float comparisons

        # Rule R3 (Rotation selection) is implicitly handled by `determine_placings` iterating all rotations.
        # This function evaluates a given pair of rotations.
        # If use_rule_31 is True, it could further filter rotations based on stability (not implemented here).

        valid_variants = []

        # Variant 1: Single box (box1 only)
        # Check 1: Local fit - Does box1 fit within the dimensions of scurr?
        fits_locally_s1 = (dims1[0] <= s_dx + tol and
                           dims1[1] <= s_dy + tol and
                           dims1[2] <= s_dz + tol)
        if fits_locally_s1:
            # Check 2: Absolute fit - Does box1, placed at scurr's origin, stay within layer/container boundaries?
            box1_abs_end_x = s_ox + dims1[0]
            box1_abs_end_y = s_oy + dims1[1]
            box1_abs_end_z = s_oz + dims1[2]

            fits_globally_s1 = (
                box1_abs_end_x <= layer_abs_x_start + layer_depth_d + tol and # Within layer depth
                box1_abs_end_y <= self.container.wc + tol and                 # Within container width
                box1_abs_end_z <= self.container.hc + tol and                 # Within container height
                s_ox >= layer_abs_x_start - tol and                           # Space starts at/after layer's X start
                s_oy >= 0 - tol and                                           # Space starts at/after Y=0
                s_oz >= 0 - tol                                               # Space starts at/after Z=0
            )
            if fits_globally_s1:
                priority_single = (s_dx - dims1[0]) + (s_dy - dims1[1]) + (s_dz - dims1[2]) # Smaller residual is better
                valid_variants.append({'stype': 'single', 'priority': priority_single, 'vol': box1.vb,
                                       'b1': box1, 'r1': rv1, 'd1': dims1,
                                       'b2': None, 'r2': None, 'd2': (0,0,0)})

        if box2: # If two boxes are being considered
            # Variant Beside (split scurr along its Y-axis)
            # Boxes are side-by-side, their combined width (dims1[1] + dims2[1]) must fit s_dy.
            # Their depths (dims1[0], dims2[0]) must fit s_dx. Heights (dims1[2], dims2[2]) must fit s_dz.
            fits_locally_beside = (max(dims1[0], dims2[0]) <= s_dx + tol and
                                   dims1[1] + dims2[1] <= s_dy + tol and
                                   max(dims1[2], dims2[2]) <= s_dz + tol)
            if fits_locally_beside:
                pair_abs_end_x = s_ox + max(dims1[0], dims2[0])
                pair_abs_end_y = s_oy + dims1[1] + dims2[1] # Combined width
                pair_abs_end_z = s_oz + max(dims1[2], dims2[2])

                fits_globally_beside = (
                    pair_abs_end_x <= layer_abs_x_start + layer_depth_d + tol and
                    pair_abs_end_y <= self.container.wc + tol and
                    pair_abs_end_z <= self.container.hc + tol and
                    s_ox >= layer_abs_x_start - tol and s_oy >= 0 - tol and s_oz >= 0 - tol
                )
                if fits_globally_beside:
                    # Priority: remaining width in scurr if this pair is placed.
                    # Or, simply s_dy as per paper's R2 (largest dimension of split).
                    priority_beside = s_dy - (dims1[1] + dims2[1])
                    valid_variants.append({'stype': 'beside', 'priority': priority_beside, 'vol': box1.vb + box2.vb,
                                           'b1': box1, 'r1': rv1, 'd1': dims1,
                                           'b2': box2, 'r2': rv2, 'd2': dims2})

            # Variant In Front (split scurr along its X-axis)
            # Boxes are one behind the other, combined depth (dims1[0] + dims2[0]) must fit s_dx.
            fits_locally_front = (dims1[0] + dims2[0] <= s_dx + tol and
                                  max(dims1[1], dims2[1]) <= s_dy + tol and
                                  max(dims1[2], dims2[2]) <= s_dz + tol)
            if fits_locally_front:
                pair_abs_end_x = s_ox + dims1[0] + dims2[0] # Combined depth
                pair_abs_end_y = s_oy + max(dims1[1], dims2[1])
                pair_abs_end_z = s_oz + max(dims1[2], dims2[2])

                fits_globally_front = (
                    pair_abs_end_x <= layer_abs_x_start + layer_depth_d + tol and
                    pair_abs_end_y <= self.container.wc + tol and
                    pair_abs_end_z <= self.container.hc + tol and
                    s_ox >= layer_abs_x_start - tol and s_oy >= 0 - tol and s_oz >= 0 - tol
                )
                if fits_globally_front:
                    priority_front = s_dx - (dims1[0] + dims2[0])
                    valid_variants.append({'stype': 'front', 'priority': priority_front, 'vol': box1.vb + box2.vb,
                                           'b1': box1, 'r1': rv1, 'd1': dims1,
                                           'b2': box2, 'r2': rv2, 'd2': dims2})

            # Variant Above (split scurr along its Z-axis)
            # Boxes are stacked, combined height (dims1[2] + dims2[2]) must fit s_dz.
            # Stacking condition: top box base must be <= bottom box base.
            # Check both (box1 on box2) and (box2 on box1) for local stacking feasibility.
            # For the purpose of fitting into scurr, we consider the max footprint.
            fits_locally_above = (max(dims1[0], dims2[0]) <= s_dx + tol and # Max footprint depth
                                  max(dims1[1], dims2[1]) <= s_dy + tol and # Max footprint width
                                  dims1[2] + dims2[2] <= s_dz + tol)       # Combined height
            
            # Stacking stability (R3.1 in some interpretations, or part of R1/R2)
            # Box1 on Box2: base of Box1 (d1,w1) <= base of Box2 (d2,w2)
            can_stack_b1_on_b2 = (dims1[0] <= dims2[0] + tol and dims1[1] <= dims2[1] + tol)
            # Box2 on Box1: base of Box2 (d2,w2) <= base of Box1 (d1,w1)
            can_stack_b2_on_b1 = (dims2[0] <= dims1[0] + tol and dims2[1] <= dims1[1] + tol)

            if fits_locally_above and (can_stack_b1_on_b2 or can_stack_b2_on_b1):
                # If either stacking order is stable, proceed with global check.
                # The actual order (who is on top) is decided by R4 in determine_placings.
                stack_abs_end_x = s_ox + max(dims1[0], dims2[0])
                stack_abs_end_y = s_oy + max(dims1[1], dims2[1])
                stack_abs_end_z = s_oz + dims1[2] + dims2[2] # Combined height

                fits_globally_above = (
                    stack_abs_end_x <= layer_abs_x_start + layer_depth_d + tol and
                    stack_abs_end_y <= self.container.wc + tol and
                    stack_abs_end_z <= self.container.hc + tol and
                    s_ox >= layer_abs_x_start - tol and s_oy >= 0 - tol and s_oz >= 0 - tol
                )
                if fits_globally_above:
                    priority_above = s_dz - (dims1[2] + dims2[2])
                    valid_variants.append({'stype': 'above', 'priority': priority_above, 'vol': box1.vb + box2.vb,
                                           'b1': box1, 'r1': rv1, 'd1': dims1,
                                           'b2': box2, 'r2': rv2, 'd2': dims2})

        if not valid_variants:
            return None

        # Select best variant: Maximize volume, then minimize priority metric (smaller is better)
        valid_variants.sort(key=lambda x: (-x['vol'], x['priority']))
        best_var = valid_variants[0]
        
        # The 'daughter_dims' key in the return dict is conceptual for R4.
        # It's not directly used by fill_layer to create daughter spaces, as
        # generate_daughter_spaces recalculates them based on bcurr.
        return {
            'box1': best_var['b1'], 'box2': best_var.get('b2'),
            'rot1': best_var['r1'], 'rot2': best_var.get('r2'),
            'stype': best_var['stype'],
            'dims1': best_var['d1'], 'dims2': best_var.get('d2', (0,0,0)),
            'daughter_dims': None, # Placeholder, actual daughter spaces created later
            'volume': best_var['vol'],
            'priority': best_var['priority']
        }

    def __init__(self, container, box_list, npop=50, nrep=10, pcross=0.67, pmut=0.33):
        self.container = container
        self.box_list = box_list
        self.npop = npop
        self.nrep = nrep
        self.pcross = pcross
        self.pmut = pmut
        self.population = []
        self.current_generation = 0
        self.total_generations = 500 # Default value for total generations
        self.nbtype = self.calculate_nbtype(self.box_list)
        self._generated_signatures = set()
        self.boxes_by_id = {b.id: b for b in self.box_list} # Create box lookup dictionary

    def initialize_population(self):
        empty_plan = self.create_empty_plan()
        generated_plans = self.basic_heuristic(
            sIn=empty_plan,
            variant='start',
            ns=self.npop
        )
        self.population = list(generated_plans)

        # Ensure population is of size npop, fill with copies if necessary
        while len(self.population) < self.npop:
            if self.population:
                self.population.append(copy.deepcopy(random.choice(self.population)))
            else: # Should not happen if heuristic returns at least one plan
                self.population.append(copy.deepcopy(empty_plan))
                if len(self.population) >= self.npop : break # Safety for npop=0 or 1

        self.population = self.population[:self.npop] # Trim if too many

        if len(self.population) != self.npop and self.npop > 0 :
             print( # Changed to print instead of raise for robustness in varied test cases
                f"Warning: Heuristic failed to generate {self.npop} plans. "
                f"Got {len(self.population)} instead."
            )

    def _get_plan_signature(self, plan):
        return tuple(sorted(
            (pl.id, pl.ox, pl.oy, pl.oz, pl.rv.rotation_id) # Corrected to use rotation_id
            for layer in plan.L
            for pl in layer.Pl
        ))

    def evaluate_fitness(self):
        if not self.population: return

        # Sort plans by descending packed volume
        
        sorted_plans = sorted(self.population,
                              # key=lambda s: -sum(self.box_list[pl.id-1].vb for layer in s.L for pl in layer.Pl)) # Old way
                              key=lambda s: -sum(self.boxes_by_id[pl.id].vb for layer in s.L for pl in layer.Pl))

        npop_current = len(sorted_plans)
        if npop_current == 0: return

        fmax = 2 / (npop_current + 1) if npop_current > 0 else 0
        d = 2 / (npop_current * (npop_current + 1)) if npop_current > 0 else 0

        for i, plan in enumerate(sorted_plans, start=1):
            plan.fitness = fmax - (i - 1) * d

    def select_parents(self, sorted_plans_ignored): # sorted_plans parameter not used as per implementation
        if not self.population: return None, None # Handle empty population
        
        # Normalize fitness values to probabilities
        total_fitness = sum(plan.fitness for plan in self.population)
        if total_fitness == 0: # Avoid division by zero if all fitnesses are 0
            # Fallback to uniform selection
            parent1 = random.choice(self.population)
            candidates = [p for p in self.population if p != parent1]
            parent2 = random.choice(candidates) if candidates else parent1
            return parent1, parent2

        selection_probs = [plan.fitness / total_fitness for plan in self.population]
        
        # Ensure probabilities sum to 1 (due to potential float issues)
        prob_sum = sum(selection_probs)
        if abs(prob_sum - 1.0) > 1e-6 : # If sum is not close to 1
            selection_probs = [p / prob_sum for p in selection_probs]


        parent1 = np.random.choice(self.population, p=selection_probs)

        candidates = [p for p in self.population if p != parent1]
        parent2 = np.random.choice(candidates) if candidates else parent1 # Ensure parent2 is different if possible

        return parent1, parent2


    def crossover(self, p1, p2):
        o = self.create_empty_plan() # Start with an empty plan

        # Transfer layers based on utilization
        # Combine layers from p1 and p2, sort by utilization
        combined_layers = sorted(p1.L + p2.L, key=lambda l: -l.vutil)
        
        temp_Bfree = {b.id for b in self.box_list}
        temp_xcfree = self.container.dc

        for lnext in combined_layers:
            # Check if layer can be added (boxes free, fits depth)
            can_add = True
            if lnext.d > temp_xcfree + 1e-6 : # Add tolerance
                can_add = False
            if not lnext.B().issubset(temp_Bfree):
                can_add = False

            if can_add:
                o.L.append(copy.deepcopy(lnext)) # Add a copy of the layer
                o.nl += 1
                temp_xcfree -= lnext.d
                temp_Bfree -= lnext.B()
        
        o.xcfree = temp_xcfree
        o.Bfree = temp_Bfree


        # Check if o.L matches either parent's layers exactly
        # This requires comparing sets of box IDs and their arrangements, which is complex.
        # A simpler check: if the sequence of layer depths and box counts is the same.
        # For now, assume crossoverOK is True if any layers were transferred.
        crossoverOK = bool(o.L) # True if at least one layer was transferred

        if crossoverOK:
            # Apply basic heuristic to complete the plan
            extended_plans = self.basic_heuristic(
                sIn=o,
                variant='crossover',
                ns=1
            )
            if extended_plans:
                o = list(extended_plans)[0] # Get the single extended plan
                # Final check if new o is identical to parents based on signature
                o_sig = self._get_plan_signature(o)
                p1_sig = self._get_plan_signature(p1)
                p2_sig = self._get_plan_signature(p2)
                if o_sig == p1_sig or o_sig == p2_sig:
                    crossoverOK = False # Mark as not OK if identical after extension
            else: # Heuristic failed to extend
                crossoverOK = False
        
        return crossoverOK, o


    def standard_mutation(self, p):
        o = self.create_empty_plan()

        if p.nl > 0:
            # Randomly select number of layers to transfer (1 to 50% of parent layers)
            nlp_to_transfer = np.random.randint(1, max(2, int(p.nl / 2)) +1) # Ensure at least 1
            
            # Transfer best nlp_to_transfer layers by utilization
            layers_to_consider = sorted(p.L, key=lambda l: -l.vutil)
            
            temp_Bfree = {b.id for b in self.box_list}
            temp_xcfree = self.container.dc
            transferred_count = 0

            for lnext in layers_to_consider:
                if transferred_count >= nlp_to_transfer: break
                
                can_add = True
                if lnext.d > temp_xcfree + 1e-6: can_add = False
                if not lnext.B().issubset(temp_Bfree): can_add = False

                if can_add:
                    o.L.append(copy.deepcopy(lnext))
                    o.nl +=1
                    temp_xcfree -= lnext.d
                    temp_Bfree -= lnext.B()
                    transferred_count +=1
            
            o.xcfree = temp_xcfree
            o.Bfree = temp_Bfree


        # Apply basic heuristic (variant='mutation', ns=1)
        mutated_plans = self.basic_heuristic(sIn=o, variant='mutation', ns=1)
        if mutated_plans:
            o = list(mutated_plans)[0]

        return o

    def merger_mutation(self, p):
        o = self.create_empty_plan()

        if p.nl >= 2: # Need at least two layers to remove two
            sorted_layers = sorted(p.L, key=lambda l: -l.vutil)
            
            # Select first random layer from all layers to exclude
            idx_exclude1 = random.randrange(p.nl)
            
            # Select second random layer from worst 50% layers (lower utilization)
            worst_half_indices = list(range(p.nl // 2, p.nl))
            if not worst_half_indices: # Handle case with few layers
                idx_exclude2 = (idx_exclude1 + 1) % p.nl if p.nl > 1 else idx_exclude1
            else:
                idx_exclude2_relative = random.choice(worst_half_indices)
                # Ensure idx_exclude2 is different from idx_exclude1
                while idx_exclude2_relative == idx_exclude1 and len(worst_half_indices) > 1:
                    idx_exclude2_relative = random.choice(worst_half_indices)
                idx_exclude2 = idx_exclude2_relative

            excluded_layers_actual = {p.L[idx_exclude1], p.L[idx_exclude2]}

            temp_Bfree = {b.id for b in self.box_list}
            temp_xcfree = self.container.dc

            for l_transfer in p.L:
                if l_transfer not in excluded_layers_actual:
                    can_add = True
                    if l_transfer.d > temp_xcfree + 1e-6 : can_add = False
                    if not l_transfer.B().issubset(temp_Bfree): can_add = False

                    if can_add:
                        o.L.append(copy.deepcopy(l_transfer))
                        o.nl += 1
                        temp_xcfree -= l_transfer.d
                        temp_Bfree -= l_transfer.B()
            
            o.xcfree = temp_xcfree
            o.Bfree = temp_Bfree


        # Apply basic heuristic (variant='mutation', ns=1)
        mutated_plans = self.basic_heuristic(sIn=o, variant='mutation', ns=1)
        if mutated_plans:
            o = list(mutated_plans)[0]

        return o

    def calculate_nbtype(self, box_list_param): # Added box_list_param
        unique_boxes = set()
        for box_obj in box_list_param: # Changed box to box_obj
            # Using a tuple of sorted dimensions to define a type
            box_dims_tuple = tuple(sorted((box_obj.db, box_obj.wb, box_obj.hb)))
            unique_boxes.add(box_dims_tuple)
        return len(unique_boxes)

    def evaluate_generation(self, next_generation_pop): # Changed param name
        original_population = self.population
        self.population = next_generation_pop # Use the passed generation
        self.evaluate_fitness()
        self.population = original_population # Restore (caller will assign new pop)

    def visualize_individual_layers(self, stowage_plan):
        """
        Visualizes each layer of the stowage plan in a separate 3D plot.
        """
        boxes_dict = {b.id: b for b in self.box_list}

        if not stowage_plan.L:
            print("No layers in the stowage plan to visualize.")
            return

        for i, layer_obj in enumerate(stowage_plan.L, start=1):
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            layer_start_x = "N/A"
            if layer_obj.Pl:
                # Assuming all boxes in a layer start at roughly the same x-depth
                # For title, just pick the first box's x-origin in that layer
                layer_start_x = f"{layer_obj.Pl[0].ox:.2f}"

            ax.set_title(f"Layer {i} Visualization (Layer Depth: {layer_obj.d:.2f}, Starts at X: {layer_start_x})")

            if not layer_obj.Pl:
                print(f"Layer {i} has no boxes to visualize.")
            else:
                for placing in layer_obj.Pl:
                    box_obj = boxes_dict.get(placing.id)
                    if not box_obj:
                        print(f"Warning: Box ID {placing.id} not found in box_list for layer {i}.")
                        continue
                        
                    rotation = placing.rv.rotation_id
                    dx, dy, dz = self.get_rotated_dimensions(box_obj, rotation)
                    x, y, z = placing.ox, placing.oy, placing.oz # These are absolute coordinates

                    # ... (rest of the plotting code for corners and faces as in your original file) ...
                    # For brevity, I'm not repeating the full Poly3DCollection part here,
                    # but it should be the same as in your `visualize_individual_layers` method.
                    # Ensure the box_color and ax.add_collection3d lines are present.
                    corners = [[x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z], [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]]
                    faces = [[corners[0], corners[1], corners[2], corners[3]], [corners[4], corners[5], corners[6], corners[7]], [corners[0], corners[1], corners[5], corners[4]], [corners[2], corners[3], corners[7], corners[6]], [corners[1], corners[2], corners[6], corners[5]], [corners[0], corners[3], corners[7], corners[4]]]
                    box_color = [random.random() for _ in range(3)]
                    ax.add_collection3d(Poly3DCollection(faces, facecolors=box_color, linewidths=1, edgecolors='k', alpha=0.7))

            ax.set_xlim([0, self.container.dc])
            ax.set_ylim([0, self.container.wc])
            ax.set_zlim([0, self.container.hc])
            ax.set_xlabel("Container Depth (X)")
            ax.set_ylabel("Container Width (Y)")
            ax.set_zlabel("Container Height (Z)")
            plt.tight_layout()
            plt.show() # Show each layer plot individually

    def visualize_stowage_plan(self, stowage_plan):
        boxes_dict = {b.id: b for b in self.box_list}
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Bin Packing Visualization")

        for layer_obj in stowage_plan.L: # Changed layer to layer_obj
            for placing in layer_obj.Pl:
                box_obj = boxes_dict[placing.id] # Changed box to box_obj
                rotation = placing.rv.rotation_id

                # Get rotated dimensions
                dx, dy, dz = self.get_rotated_dimensions(box_obj, rotation)

                x, y, z = placing.ox, placing.oy, placing.oz

                corners = [
                    [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                    [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
                ]
                faces = [
                    [corners[0], corners[1], corners[2], corners[3]],
                    [corners[4], corners[5], corners[6], corners[7]],
                    [corners[0], corners[1], corners[5], corners[4]],
                    [corners[2], corners[3], corners[7], corners[6]],
                    [corners[1], corners[2], corners[6], corners[5]],
                    [corners[3], corners[0], corners[4], corners[7]]
                ]
                color = [random.random() for _ in range(3)]
                ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=0.7))

        ax.set_xlim([0, self.container.dc])
        ax.set_ylim([0, self.container.wc])
        ax.set_zlim([0, self.container.hc])
        ax.set_xlabel("Depth (x)")
        ax.set_ylabel("Width (y)")
        ax.set_zlabel("Height (z)")

        plt.tight_layout()
        plt.show()
    
    def get_visualization_data_for_plan(self, stowage_plan):
        """
        Generates data suitable for frontend visualization of a stowage plan, layer by layer.
        """
        boxes_dict = {b.id: b for b in self.box_list}
        visualization_data = {
            "container_dims": {
                "dc": self.container.dc,
                "wc": self.container.wc,
                "hc": self.container.hc
            },
            "layers": []
        }

        if not stowage_plan or not stowage_plan.L:
            return visualization_data # Return empty structure if no plan or layers

        for i, layer_obj in enumerate(stowage_plan.L, start=1):
            layer_data = {
                "layer_number": i,
                "layer_depth_dim": layer_obj.d, # The actual depth of this layer
                "boxes": []
            }
            layer_start_x_ref = "N/A"
            if layer_obj.Pl:
                valid_ox_values = [p.ox for p in layer_obj.Pl if isinstance(p.ox, (int, float))]
                if valid_ox_values:
                    layer_start_x_ref = min(valid_ox_values) 
            layer_data["layer_start_x_ref"] = layer_start_x_ref

            for placing in layer_obj.Pl:
                box_obj = boxes_dict.get(placing.id)
                if not box_obj: continue
                
                rotation = placing.rv.rotation_id
                dx, dy, dz = self.get_rotated_dimensions(box_obj, rotation)
                hex_color = f'#{random.randint(0, 0xFFFFFF):06x}'

                layer_data["boxes"].append({"id": placing.id, "ox": placing.ox, "oy": placing.oy, "oz": placing.oz, "dx": dx, "dy": dy, "dz": dz, "color": hex_color})
            visualization_data["layers"].append(layer_data)
        return visualization_data

    def get_plan_metrics(self, stowage_plan):
        metrics = {
            "total_layers": 0,
            "unstowed_box_ids": [],
            "remaining_container_depth": 0,
            "total_packed_volume": 0,
            "container_volume": 0,
            "overall_volume_utilization": "0.00%"
        }
        if not stowage_plan:
            return metrics

        metrics["total_layers"] = stowage_plan.nl
        metrics["unstowed_box_ids"] = sorted(list(stowage_plan.Bfree))
        metrics["remaining_container_depth"] = stowage_plan.xcfree
        metrics["total_packed_volume"] = stowage_plan.v
        metrics["container_volume"] = stowage_plan.container.vc
        
        if stowage_plan.container.vc > 0:
            utilization = (stowage_plan.v / stowage_plan.container.vc) * 100
            metrics["overall_volume_utilization"] = f"{utilization:.2f}%"
        else:
            metrics["overall_volume_utilization"] = "N/A (container volume is zero)"
        
        layer_details = []
        for i, layer_obj in enumerate(stowage_plan.L, start=1):
            layer_details.append({
                "layer_number": i,
                "depth": layer_obj.d,
                "volume_utilization": f"{layer_obj.vutil * 100:.2f}%",
                "num_boxes": layer_obj.npl,
                "box_ids": sorted(list(layer_obj.B()))
            })
        metrics["layer_details"] = layer_details
        return metrics


# Function to run the GA and get the best plan details
def solve_loading_problem(container_data, boxes_data, ga_params):
    container_instance = hybrid_GA.container(
        dc=float(container_data['dc']),
        wc=float(container_data['wc']),
        hc=float(container_data['hc']),
        vc=float(container_data['dc']) * float(container_data['wc']) * float(container_data['hc'])
    )

    boxes_list_instances = [] # Renamed to avoid conflict with module-level boxes_list
    for b_data in boxes_data:
        db = float(b_data['db'])
        wb = float(b_data['wb'])
        hb = float(b_data['hb'])
        allowed_orientations = b_data.get('allowed_orientations', list(range(1,7))) # Get allowed_orientations
        # Ensure allowed_orientations are integers if provided and valid range
        if not (isinstance(allowed_orientations, list) and \
                all(isinstance(x, int) and 1 <= x <= 6 for x in allowed_orientations) and \
                allowed_orientations): # Ensure not empty
            allowed_orientations = list(range(1,7)) # Default if malformed or empty
        boxes_list_instances.append(hybrid_GA.box(
            id=int(b_data['id']), db=db, wb=wb, hb=hb, vb=db*wb*hb, allowed_orientations=allowed_orientations
        ))

    ga = hybrid_GA(
        container=container_instance,
        box_list=boxes_list_instances, # Use the new list name
        npop=ga_params.get('npop', 50),
        nrep=ga_params.get('nrep', 10),
        pcross=ga_params.get('pcross', 0.67),
        pmut=ga_params.get('pmut', 0.33)
        # nmerge is a class attribute, can be overridden if ga_params includes it and GA init supports it
    )
    ga.nmerge = ga_params.get('nmerge', ga.nmerge) # Allow override of nmerge
    ga.total_generations = ga_params.get('max_generations', 50)

    ga.initialize_population()

    if not ga.population:
        return {
            "error": "Failed to initialize GA population. Check box and container dimensions.",
            "metrics": ga.get_plan_metrics(None),
            "visualization_data": ga.get_visualization_data_for_plan(None)
        }

    ga.evaluate_fitness()

    start_time = time.time()
    generations_completed = 0
    max_generations = ga_params.get('max_generations', 50) 
    max_time_limit = ga_params.get('max_time_limit', 30)

    # --- Start of GA loop (copied and adapted from original __main__) ---
    while generations_completed < max_generations and (time.time() - start_time) < max_time_limit:
        ga.current_generation = generations_completed 
        for p_idx, plan in enumerate(ga.population):
            if not hasattr(plan, 'fitness'): plan.fitness = 0.0
        sorted_plans = sorted(ga.population, key=lambda p: p.fitness, reverse=True)
        next_generation_list = copy.deepcopy(sorted_plans[:ga.nrep])
        while len(next_generation_list) < ga.npop:
            op_choice = random.choices(population=["crossover", "mutation"], weights=[ga.pcross, ga.pmut], k=1)[0]
            p1, p2 = ga.select_parents(sorted_plans)
            if p1 is None:
                if ga.population: p1 = random.choice(ga.population)
                else: break 
            if p2 is None and p1: p2 = p1
            offspring = None
            if op_choice == "crossover":
                if p1 and p2:
                    crossoverOK, offspring_co = ga.crossover(p1, p2)
                    if crossoverOK: offspring = offspring_co
                    else: offspring = ga.standard_mutation(p1)
                elif p1: offspring = ga.standard_mutation(p1)
            else: 
                if p1: offspring = ga.standard_mutation(p1)
            if offspring: next_generation_list.append(offspring)
            elif p1 and len(next_generation_list) < ga.npop: next_generation_list.append(copy.deepcopy(p1))
            elif not p1 and len(next_generation_list) < ga.npop and ga.population: next_generation_list.append(copy.deepcopy(random.choice(ga.population)))
        for _ in range(ga.nmerge):
            if not ga.population: break
            p_merge_candidates = sorted_plans if sorted_plans else ga.population
            if not p_merge_candidates: continue
            p_merge, _ = ga.select_parents(p_merge_candidates)
            if p_merge is None: p_merge = random.choice(p_merge_candidates)
            offspring_merge = ga.merger_mutation(p_merge)
            temp_eval_list = next_generation_list + [offspring_merge]
            ga.evaluate_generation(temp_eval_list) 
            if len(next_generation_list) < ga.npop: next_generation_list.append(offspring_merge)
            else:
                next_generation_list.sort(key=lambda p_sort: p_sort.fitness, reverse=True)
                if offspring_merge.fitness > next_generation_list[-1].fitness: next_generation_list[-1] = offspring_merge
            next_generation_list.sort(key=lambda p_sort: p_sort.fitness, reverse=True)
        if not next_generation_list and ga.population: next_generation_list = copy.deepcopy(ga.population)
        elif not next_generation_list and not ga.population:
            return {"error": "GA evolution failed, population became empty.", "metrics": ga.get_plan_metrics(None), "visualization_data": ga.get_visualization_data_for_plan(None)}
        ga.evaluate_generation(next_generation_list)
        ga.population = sorted(next_generation_list, key=lambda p: p.fitness, reverse=True)[:ga.npop]
        generations_completed += 1
    # --- End of GA loop ---

    if not ga.population:
        return {
            "error": "No solution found after evolution, population is empty.",
            "metrics": ga.get_plan_metrics(None),
            "visualization_data": ga.get_visualization_data_for_plan(None)
        }

    sopt = max(ga.population, key=lambda p: p.v)

    return {
        "metrics": ga.get_plan_metrics(sopt),
        "visualization_data": ga.get_visualization_data_for_plan(sopt)
    }

    

# Sample Container (depth, width, height, volume)
container_instance = hybrid_GA.container(dc=10, wc=5, hc=8, vc=400) # Changed var name

# Sample Boxes (id, depth, width, height, volume)
boxes_list = [ # Changed var name
    hybrid_GA.box(id=1, db=2, wb=3, hb=1, vb=6, allowed_orientations=[1, 2, 3, 4, 5, 6]),
    hybrid_GA.box(id=2, db=1, wb=1, hb=4, vb=4, allowed_orientations=[1, 3, 5]),
    hybrid_GA.box(id=3, db=3, wb=2, hb=2, vb=12, allowed_orientations=[1, 2]),
    hybrid_GA.box(id=4, db=3, wb=6, hb=2, vb=36, allowed_orientations=None), # Defaults to all
    hybrid_GA.box(id=5, db=3, wb=2, hb=2, vb=12), # Defaults to all
    hybrid_GA.box(id=6, db=3, wb=3, hb=2, vb=18, allowed_orientations=[1,2,3,4,5,6]),
    hybrid_GA.box(id=7, db=3, wb=7, hb=2, vb=42, allowed_orientations=[1]),
    hybrid_GA.box(id=8, db=6, wb=3, hb=2, vb=36), # Defaults to all
    hybrid_GA.box(id=9, db=1, wb=2, hb=2, vb=4, allowed_orientations=[1,3,4,6]),
]

# Initialize GA with parameters
ga_instance = hybrid_GA( # Renamed to avoid conflict with module name 'hga'
        container=container_instance, # Use new var name
        box_list=boxes_list,          # Use new var name
        npop=50,
        pcross=0.67,
        pmut=0.33
)

# if __name__ == "__main__":
#     print("Before initializing population...")
#     ga.initialize_population()
#     print(f"After initializing population ({len(ga.population)} plans)... entering evolution loop")

#     ga.evaluate_fitness()

#     start_time = time.time()
#     generations_completed = 0
#     max_generations = 50 # Reduced for quick test; paper uses 500
#     max_time_limit = 30  # seconds; paper uses 500

#     print("before while loop")

#     while generations_completed < max_generations and (time.time() - start_time) < max_time_limit:
#         ga.current_generation = generations_completed # Update current generation for R3 rule

#         sorted_plans = sorted(ga.population, key=lambda p: -p.fitness if hasattr(p, 'fitness') else -float('inf'))

#         # Elitism: Keep top nrep plans
#         next_generation_list = copy.deepcopy(sorted_plans[:ga.nrep]) # Changed var name

#         # Generate remaining population via crossover and mutation
#         while len(next_generation_list) < ga.npop:
#             op_choice = random.choices( # Changed op to op_choice
#                 population=["crossover", "mutation"],
#                 weights=[ga.pcross, ga.pmut],
#                 k=1
#             )[0]

#             p1, p2 = ga.select_parents(sorted_plans)
#             if p1 is None: # Safety break if selection fails
#                 print("Parent selection failed, breaking generation.")
#                 break

#             offspring = None # Changed o to offspring
#             if op_choice == "crossover":
#                 if p2 is None: p2 = p1 # Handle case where only one parent could be selected
#                 crossoverOK, offspring_co = ga.crossover(p1, p2)
#                 if crossoverOK:
#                     offspring = offspring_co
#                 else: # If crossover failed (e.g. identical to parent), try mutation on p1
#                     offspring = ga.standard_mutation(p1)

#             else: # Mutation
#                 offspring = ga.standard_mutation(p1)

#             if offspring: # Ensure offspring is not None
#                  next_generation_list.append(offspring)
#             elif len(next_generation_list) < ga.npop : # Fill with a copy if operator failed
#                  next_generation_list.append(copy.deepcopy(p1))


#         # Add merger mutations (replace worst individuals if pop is full)
#         for _ in range(ga.nmerge):
#             p_merge, _ = ga.select_parents(sorted_plans)
#             if p_merge is None: continue
#             offspring_merge = ga.merger_mutation(p_merge)
#             if len(next_generation_list) < ga.npop:
#                 next_generation_list.append(offspring_merge)
#             else: # Replace one of the worst if merger is better
#                 if offspring_merge.fitness > next_generation_list[-1].fitness: # Assuming sorted by fitness desc
#                     next_generation_list[-1] = offspring_merge
#                     next_generation_list.sort(key=lambda p: -p.fitness if hasattr(p, 'fitness') else -float('inf'))


#         # Evaluate the fitness of the new generation (temporarily assign to self.population)
#         ga.evaluate_generation(next_generation_list) # Pass the list to evaluate

#         # Reduce to the npop best solutions
#         ga.population = sorted(next_generation_list, key=lambda p: -p.fitness if hasattr(p, 'fitness') else -float('inf'))[:ga.npop]

#         generations_completed += 1
#         if generations_completed % 10 == 0:
#             best_current_fitness = ga.population[0].fitness if ga.population else -1
#             print(f"Gen: {generations_completed}, Best Fitness: {best_current_fitness:.4f}, Pop Size: {len(ga.population)}")


#     # Determine the best plan from the final population
#     if ga.population:
#         sopt = max(ga.population, key=lambda p: p.v) # Select based on actual volume 'v'

#         print("\n===== Stowage Plan Summary =====\n")
#         print(f"Total layers: {sopt.nl}")
#         print(f"Unstowed box IDs: {sopt.Bfree}")
#         print(f"Remaining container depth (xcfree): {sopt.xcfree:.2f}")
#         print(f"Total packed volume: {sopt.v:.2f} / {sopt.container.vc:.2f}")
#         if sopt.container.vc > 0:
#              print(f"Overall volume utilization: {sopt.v / sopt.container.vc:.2%}\n")
#         else:
#              print("Overall volume utilization: N/A (container volume is zero)\n")


#         for i, layer_obj in enumerate(sopt.L, start=1): # Changed layer to layer_obj
#             print(f"--- Layer {i} ---")
#             print(f"Depth (d): {layer_obj.d}")
#             print(f"Volume utilization (vutil): {layer_obj.vutil:.2%}")
#             print(f"Number of boxes in layer (npl): {layer_obj.npl}")
#             print(f"Box IDs in this layer: {sorted(layer_obj.B())}")
#             print("Placings:")
#             for placing in layer_obj.Pl:
#                 print(f"  Box ID: {placing.id}, Pos: ({placing.ox:.2f}, {placing.oy:.2f}, {placing.oz:.2f}), Rotation ID: {placing.rv.rotation_id}")
#             print()

#         # Visualize the best stowage plan
#         # ga.visualize_stowage_plan(sopt) # Uncomment to visualize

#         print("\nVisualizing individual layers of the best plan...")
#         ga.visualize_individual_layers(sopt)
        
#     else:
#         print("No solution found after evolution.")




