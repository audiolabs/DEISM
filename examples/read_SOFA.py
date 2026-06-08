from netCDF4 import Dataset
import os
import numpy as np


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
    rec_pos = nc.variables.get('ReceiverPosition')
    
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

    rec_pos_lr = np.squeeze(rec_pos)
    print(f" ListenerPosition: {lis_pos[0]}")  
    print(f" EmitterPosition: {emt_pos[0]}")
    print(f" ReceiverPosition: {rec_pos_lr[0], rec_pos_lr[1]}")

    if "Data.Delay" in nc.variables:
        delay_samples = nc.variables["Data.Delay"][:]
        print(f"# of delay samples SOFA recorded: {delay_samples}")
    else:
        print("This SOFA file dosen't have Data.Delay")

    nc.close()

def print_sofa_mapping(filepath):
    if not os.path.exists(filepath):
        print("no file exists！")
        return
        
    nc = Dataset(filepath, 'r')
    M = nc.variables['Data.IR'].shape[0] # obtain the total number of measurements
    
    emitter_ids = nc.variables['EmitterID'][:]
    emitter_pos = nc.variables['EmitterPosition'][:]
    # receiver_ids = nc.variables['ReceiverID'][:]
    # receiver_pos = nc.variables['ReceiverPosition'][:]
    listener_pos = nc.variables['ListenerPosition'][:]
    listener_view = nc.variables['ListenerView'][:]
    
    print("Part 4: Measurement Mapping (Index -> LS ID & MP ID)")
    print(f"{'Index':<6} | {'Src ID':<6} | {'Rec ID':<6} | {'Source Pos (X, Y, Z)':<21} | {'Receiver Pos (X, Y, Z)'}")
     
    src_id = int(emitter_ids[0, 0])
    #rec_id = int(receiver_ids[0, 0])
    
    src_x, src_y, src_z = emitter_pos[0, 0, 0], emitter_pos[0, 1, 0], emitter_pos[0, 2, 0]
    lis_x, lis_y, lis_z = listener_pos[0, 0], listener_pos[0, 1], listener_pos[0, 2]

    for m in range(M):    
        view_az = listener_view[m, 0]
        view_el = listener_view[m, 1]

        #print(f"{m:<6} | Src{src_id:02d}  | Rec{rec_id:02d}  | [{src_x:.3f}, {src_y:.3f}, {src_z:.3f}] | [{rec_x:.3f}, {rec_y:.3f}, {rec_z:.3f}]")
        print(f"{m:<6} | Src{src_id:02}   | [{src_x:.3f}, {src_y:.3f}, {src_z:.3f}] | [{lis_x:.3f}, {lis_y:.3f}, {lis_z:.3f}] | [{view_az:>5.1f}°, {view_el:>4.1f}°]")
    
    nc.close()

if __name__ == "__main__":
    file_path = r"D:\Projects\DEISM\DEISM_main\DEISM\examples\data\sampled_directivity\sofa\01 single reflection (infinite plate)\BRIRs\scene1_BRIRs_Rigid.sofa"
    xray_sofa_raw(file_path)
    print_initial_angles(file_path)
    # print the mapping table of the positions of sources and receivers
    print_sofa_mapping(file_path)
