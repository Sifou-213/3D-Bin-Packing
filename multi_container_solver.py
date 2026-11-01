from hga import hybrid_GA
import copy
import random
import time 

def solve_multi_container(boxes, container_dims, max_containers=5, generations=30, npop=30, nrep=10, time_limit_per_container=None): # Changed nrep_percentage to nrep
    container_results = []
    # 'boxes' is a list of hga.hybrid_GA.box objects, each representing a unique box instance.
    # The 'id' field in these hga.hybrid_GA.box objects is the temporary, sequential ID (1 to N)
    # assigned in app.py for this list.
    # The 'original_type_id' field in these hga.hybrid_GA.box objects holds the truly unique
    # instance identifier from the frontend (e.g., "SKU-XYZ_1").
    
    remaining_boxes_master_list = copy.deepcopy(boxes)
    # Tracks the instance 'id' of boxes already packed. 'original_type_id' is shared by every
    # instance of the same box type, so it cannot be used to tell instances apart.
    globally_packed_ids = set()

    for current_container_number in range(1, max_containers + 1):
        # Filter boxes from the master list that haven't been globally packed yet
        boxes_to_try_for_this_container = [
            b for b in remaining_boxes_master_list if b.id not in globally_packed_ids
        ]

        if not boxes_to_try_for_this_container:
            print(f"Container {current_container_number}: No more boxes left to pack.")
            break

        # For this specific GA run, we need a new set of 1-indexed IDs.
        # We also need a way to map these temporary HGA IDs back to the original box objects
        # to retrieve their 'original_type_id' and other details later.
        
        # temp_hga_id_to_original_box_map: maps temporary HGA ID -> original box object from boxes_to_try_for_this_container
        temp_hga_id_to_original_box_map = {}
        
        remapped_boxes_for_hga = []
        for temp_hga_id, original_box_obj in enumerate(boxes_to_try_for_this_container, start=1):
            temp_hga_id_to_original_box_map[temp_hga_id] = original_box_obj
            
            # Create a new box instance for HGA, using the temporary HGA ID
            hga_run_box = hybrid_GA.box(
                id=temp_hga_id, # Temporary, 1-indexed ID for this specific GA run
                db=original_box_obj.db, wb=original_box_obj.wb, hb=original_box_obj.hb, vb=original_box_obj.vb,
                allowed_orientations=original_box_obj.allowed_orientations,
                name=original_box_obj.name,
                original_type_id=original_box_obj.original_type_id, # Carry this along
                weight=original_box_obj.weight,
                load_bearing_strength=original_box_obj.load_bearing_strength
            )
            remapped_boxes_for_hga.append(hga_run_box)

        if not remapped_boxes_for_hga: # Should not happen if boxes_to_try_for_this_container was not empty
            print(f"Container {current_container_number}: No boxes remapped for HGA (this is unexpected).")
            continue

        container = hybrid_GA.container(*container_dims, vc=container_dims[0]*container_dims[1]*container_dims[2])
        
        # Use the provided npop, nrep, and generations directly
        ga = hybrid_GA(container=container, box_list=remapped_boxes_for_hga, npop=npop, nrep=nrep)
        ga.total_generations = generations
        ga.initialize_population()

        if not ga.population:
            print(f"Container {current_container_number}: GA population empty after initialization.")
            continue
            
        ga.evaluate_fitness()

        container_solve_start_time = time.time() # Start timer for this container's GA
        container_solve_end_time = container_solve_start_time

        for gen_count in range(generations):
            if time_limit_per_container and time.time() - container_solve_start_time >= time_limit_per_container:
                break
            # Standard GA loop: selection, crossover, mutation
            sorted_pop = sorted(ga.population, key=lambda p: p.fitness, reverse=True)
            new_gen = sorted_pop[:ga.nrep] # Elitism
            
            while len(new_gen) < ga.npop:
                p1, p2 = ga.select_parents(sorted_pop)
                if p1 is None: # Should not happen if population is not empty
                    if sorted_pop: p1 = random.choice(sorted_pop)
                    else: break # Cannot proceed
                if p2 is None: p2 = p1

                crossoverOK, offspring = ga.crossover(p1, p2)
                if not crossoverOK: # If crossover fails, mutate parent1
                    offspring = ga.standard_mutation(p1)
                new_gen.append(offspring)
            
            # Apply merger mutation to some individuals
            for _ in range(ga.nmerge): # nmerge is a class attribute of hybrid_GA
                if not new_gen: break
                p_to_merge = random.choice(new_gen) # Select from the new generation being built
                merged_offspring = ga.merger_mutation(p_to_merge)
                new_gen.append(merged_offspring)


            ga.evaluate_generation(new_gen) # Evaluate fitness of the new generation
            ga.population = sorted(new_gen, key=lambda p: p.fitness, reverse=True)[:ga.npop] # Select best npop

            container_solve_end_time = time.time() # End timer for this container's GA

        if not ga.population:
            print(f"Container {current_container_number}: No solution found, GA population empty after evolution.")
            continue

        best_plan_for_this_container = max(ga.population, key=lambda p: p.v)
        
        packed_ids_this_container_run = set()
        for layer in best_plan_for_this_container.L:
            for pl in layer.Pl: # pl.id is the temporary HGA ID
                original_box_obj_placed = temp_hga_id_to_original_box_map.get(pl.id)
                if original_box_obj_placed:
                    packed_ids_this_container_run.add(original_box_obj_placed.id)

        if not packed_ids_this_container_run:
            # Nothing fits in an empty container, so another container would not help either.
            print(f"Container {current_container_number}: none of the remaining boxes fit, stopping.")
            break

        globally_packed_ids.update(packed_ids_this_container_run)

        metrics = ga.get_plan_metrics(best_plan_for_this_container)
        metrics["time_taken_seconds"] = round(container_solve_end_time - container_solve_start_time, 3) # Add time taken
        viz_data = ga.get_visualization_data_for_plan(best_plan_for_this_container)

        for layer_detail in viz_data.get('layers', []):
            for box_info in layer_detail.get('boxes', []): # box_info['id'] is the temp HGA id
                original_box_obj_viz = temp_hga_id_to_original_box_map.get(box_info['id'])
                if original_box_obj_viz:
                    box_info['id'] = original_box_obj_viz.id
                    box_info['name'] = original_box_obj_viz.name
                    box_info['original_type_id'] = original_box_obj_viz.original_type_id
        
        detailed_unstowed = []
        # Unstowed box IDs in metrics are temporary HGA IDs from best_plan_for_this_container.Bfree
        for temp_hga_unstowed_id in metrics.get("unstowed_box_ids", []):
            original_box_obj_unstowed = temp_hga_id_to_original_box_map.get(temp_hga_unstowed_id)
            if original_box_obj_unstowed:
                # Only consider it truly unstowed if it hasn't been packed in *any* container yet
                if original_box_obj_unstowed.id not in globally_packed_ids:
                     detailed_unstowed.append({
                         'id': original_box_obj_unstowed.id, # This is the original sequential ID from app.py
                         'name': original_box_obj_unstowed.name,
                         'original_type_id': original_box_obj_unstowed.original_type_id # Unique instance ID
                     })
        metrics["unstowed_boxes_detailed"] = detailed_unstowed
        # Update unstowed_box_ids to be the list of original_type_ids for consistency
        metrics["unstowed_box_ids"] = [d['original_type_id'] for d in detailed_unstowed]

        container_results.append({
            "container_id": current_container_number,
            "metrics": metrics,
            "visualization_data": viz_data,
            "container_dimensions_used": {"dc": container.dc, "wc": container.wc, "hc": container.hc}
        })

        print(f"\nContainer {current_container_number} processed")
        print(f"Packed Volume: {metrics['total_packed_volume']} m^3")
        print(f"Volume Utilization: {metrics['overall_volume_utilization']}")
        print(f"Unpacked for next (Original IDs): {[b['original_type_id'] for b in metrics.get('unstowed_boxes_detailed', [])]}")
        print(f"Layers Used: {metrics['total_layers']}")

    return container_results


def generate_large_box_set():
    boxes = []
    box_types_proto = [
        {"name": "Alpha", "db": 3, "wb": 2, "hb": 2, "weight": 10, "lbs": 100, "allowed": [1,2,3,4,5,6]},
        {"name": "Beta",  "db": 2, "wb": 1.5,"hb": 1, "weight": 5,  "lbs": 50,  "allowed": [1,3,5]},
        {"name": "Gamma", "db": 1, "wb": 1,  "hb": 4, "weight": 2,  "lbs": 20,  "allowed": [1,2,6]},
    ]
    instance_counter = 1 # This will be the 'id' for hga.box, used for internal HGA list
    for type_idx, proto in enumerate(box_types_proto):
      for i in range(random.randint(5,10)): 
        vb = proto['db'] * proto['wb'] * proto['hb']
        # 'original_type_id' should be globally unique for each box instance
        globally_unique_instance_id = f"{proto['name']}_inst{i+1}" 
        boxes.append(hybrid_GA.box(
            id=instance_counter, 
            db=proto['db'], wb=proto['wb'], hb=proto['hb'], vb=vb,
            allowed_orientations=proto['allowed'],
            name=proto['name'],
            original_type_id=globally_unique_instance_id, 
            weight=proto['weight'],
            load_bearing_strength=proto['lbs']
        ))
        instance_counter += 1
    return boxes


if __name__ == "__main__":
    container_dims = (10, 5, 5)
    example_boxes = generate_large_box_set()
    print(f"Generated {len(example_boxes)} box instances for multi-container solving.")
    results = solve_multi_container(example_boxes, container_dims, max_containers=3, generations=20, npop=30)
    print(f"\n--- Overall Results ---")
    print(f"Number of containers used in solution: {len(results)}")
    for res_item in results:
        print(f"Container ID: {res_item['container_id']}, Packed Volume: {res_item['metrics']['total_packed_volume']}")
