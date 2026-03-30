import numpy as np
from deism.core_deism import DEISM
from deism.directivity_visualizer import Dir_Visualizer

def main():
    # Instantiate DEISM in RTF/shoebox mode.
    model = DEISM("RTF", "shoebox")
    
    model.update_wall_materials()  
    model.update_freqs()           
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
    
    model.update_source_receiver()
    # Run deism
    print("Running DEISM with manually injected SOFA coefficients...")
    model.run_DEISM(if_clean_up=True)
    
    print(f"Done! Result shape: {model.params['RTF'].shape}")

if __name__ == "__main__":
    main()