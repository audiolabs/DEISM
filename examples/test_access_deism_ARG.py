import os
import time
import numpy as np

from deism.core_deism import DEISM
from deism.core_deism_arg import rotate_room_src_rec, find_wall_centers
from deism.data_loader import detect_conflicts
from deism.directivity_visualizer import Dir_Visualizer

def init_parameters_convex(params):
    """
    Initialize additional convex-room parameters for DEISM-ARG.
    Geometry and rotation are adapted from the original low-level example.
    """
    # Room vertices (simple 8-vertex convex room)
    vertices = np.array(
        [
            [0, 0, 0],  # Origin, but it does not have to be the origin
            [0, 0, 3.5],  # [x, y, z] coordinates of the room vertices
            [0, 3, 2.5],
            [0, 3, 0],
            [4, 0, 0],
            [4, 0, 3.5],
            [4, 3, 2.5],
            [4, 3, 0],
        ]
    )
    # --- Room rotation, if rotate the room w.r.t the origin ---
    if_rotate_room = 1
    # --- Room rotation angles using Z-X-Z Euler angles ---
    room_rotation = np.array([90, 90, 90])  # [alpha, beta, gamma] in degrees

    params["vertices"] = vertices
    params["wallCenters"] = find_wall_centers(vertices)
    params["if_rotate_room"] = if_rotate_room
    params["room_rotation"] = room_rotation

    # Apply room rotation to the room vertices and source/receiver positions
    if if_rotate_room:
        params = rotate_room_src_rec(params)

    return params


def main():
    # Use DEISM class for a convex (ARG) room
    model = DEISM("RTF", "convex")
    
    # Override geometry/rotation with this example's convex room
    model.params = init_parameters_convex(model.params)
    # Apply Conflict Checks
    detect_conflicts(model.params)
    
    # Standard convex workflow
    model.update_wall_materials()  
    model.update_freqs()       
    # For convex (ARG) rooms, image paths and reflection_matrix must be set
    # before initializing ARG directivities, so update_source_receiver comes first.
    model.update_source_receiver()       
    model.update_directivities()
    
    # Load SOFA data
    sofa_file = "./examples/data/sampled_directivity/sofa/mit_kemar_normal_pinna.sofa"
    use_recip = bool(model.params.get("ifReciprocal", 0))

    Dir_Visualizer.inject_sofa_into_deism(
        model, 
        sofa_path=sofa_file, 
        role="receiver",         
        use_reciprocal=use_recip
    )
    
    # Run deism
    print("Running DEISM-ARG with manually injected SOFA coefficients...")
    model.run_DEISM(if_clean_up=True)
    
    print(f"Done! Result shape: {model.params['RTF'].shape}")

if __name__ == "__main__":
    main()