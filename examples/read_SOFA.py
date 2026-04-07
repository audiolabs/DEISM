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

if __name__ == "__main__":
    file_path = r"D:\Projects\DEISM\DEISM_main\DEISM\examples\data\sampled_directivity\sofa\BRIRs_from_a_room\A\023.sofa"
    xray_sofa_raw(file_path)