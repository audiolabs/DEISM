from netCDF4 import Dataset
import os

def xray_sofa_raw(filepath):
    if not os.path.exists(filepath):
        print("no file exists！")
        return
        
    nc = Dataset(filepath, 'r')
        
    print("Part 1: Global Attributes")
    # nc.ncattrs() lists all global attributes
    for attr_name in nc.ncattrs():
        attr_value = nc.getncattr(attr_name)
        print(f" 🔹 {attr_name}: {attr_value}")
        
    print("\n" + "-"*50 + "\n")
        
    print("Part 2: Variables")
    for var_name, var_obj in nc.variables.items():
        # obtain shapes of variables
        shape = var_obj.shape
        # obtain descriptions of variables
        description = getattr(var_obj, 'long_name', getattr(var_obj, 'Description', ''))
        
        info = f" 🔸 {var_name}: shape {shape}"
        if description:
            info += f"  (description: {description})"
        print(info)
        
    nc.close()

def print_initial_angles(filepath):
    if not os.path.exists(filepath):
        print("no file exists！")
        return
        
    nc = Dataset(filepath, 'r')
    
    print("Part 3: initial angles (Measurement Index = 0)")
    
    # Abstract Source and Listener variables
    src_pos = nc.variables.get('SourcePosition')
    lis_pos = nc.variables.get('ListenerPosition')
    lis_view = nc.variables.get('ListenerView')
    lis_up = nc.variables.get('ListenerUp')
    emt_pos = nc.variables.get('EmitterPosition')
    
    # Get coordinate system type (Cartesian or spherical)
    src_type = getattr(src_pos, 'Type', 'Unknown') if src_pos else 'Unknown'
    
    # Print Listener attitude
    if lis_view is not None:
        print(f" ListenerView: {lis_view[0]}")
    if lis_up is not None:
        print(f" ListenerUp: {lis_up[0]}")
            
    # Print Source relative position
    if src_pos is not None:
        print(f" SourcePosition: {src_pos[0]}")
        if 'spherical' in src_type.lower():
            azimuth = src_pos[0][0]
            print(f" Conclusion: at initial measurement, the source is located at the azimuth of {azimuth} degrees.")

    print(f" ListenerPosition: {lis_pos[0]}")
    print(f" EmitterPosition: {emt_pos[0]}")

    nc.close()

if __name__ == "__main__":
    file_path = r"D:\Projects\DEISM\DEISM_main\DEISM\examples\data\sampled_directivity\sofa\BRIRs_from_a_room\A\023.sofa"
    xray_sofa_raw(file_path)
    print_initial_angles(file_path)
