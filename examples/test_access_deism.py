import numpy as np
from deism.core_deism import DEISM, vectorize_C_nm_s
from deism.directivity_visualizer import Dir_Visualizer

def main():
    # Instantiate DEISM in RTF/shoebox mode.
    model = DEISM("RTF", "shoebox")
    
    # Initialize, generate Wigner matrix
    model.update_directivities() 
    
    # Load SOFA data
    sofa_file = "./examples/data/sampled_directivity/sofa/mit_kemar_normal_pinna.sofa"
    target_freqs = model.params["freqs"]
    sh_order = model.params["receiverOrder"]
    use_recip = bool(model.params.get("ifReciprocal", 0))

    # Obtain SOFA coefficents
    cnm_sofa, r0_sofa = Dir_Visualizer.get_deism_sh_coeffs(
        sofa_file, 
        target_freqs, 
        max_order=sh_order, 
        use_reciprocal=use_recip
    )
    
    # Override initialized parameters
    model.params["C_nm_s"] = cnm_sofa
    model.params["radiusSource"] = r0_sofa
    
    # Update vectorized data
    model.params = vectorize_C_nm_s(model.params)
    
    # Run deism
    model.update_wall_materials()
    model.update_source_receiver()
    
    print("Running DEISM with manually injected SOFA coefficients...")
    model.run_DEISM(if_clean_up=True)
    
    print(f"Done! Result shape: {model.params['RTF'].shape}")

if __name__ == "__main__":
    main()