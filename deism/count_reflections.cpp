#include <cmath>

extern "C" {
    // Fast C++ implementation of reflection path counting
    // For fixed positions: source at (0,0,0) and receiver at (room_dims)
    
    long long count_reflections_shoebox_test(
        int order,
        double Lx, double Ly, double Lz,  // room dimensions
        double c,
        double T60
    ) {
        long long count = 0;
        
        // Pre-compute constants
        double max_distance_squared = (c * T60) * (c * T60);
        double Lx_sq = Lx * Lx;
        double Ly_sq = Ly * Ly;
        double Lz_sq = Lz * Lz;
        
        // Main loop: iterate over all parity and reflection order combinations
        for (int p_x = 0; p_x < 2; p_x++) {
            for (int p_y = 0; p_y < 2; p_y++) {
                for (int p_z = 0; p_z < 2; p_z++) {
                    // For each (p_x, p_y, p_z), generate all (q_x, q_y, q_z) that give valid reflection orders
                    for (int ref_order = 0; ref_order <= order; ref_order++) {
                        // Generate all combinations (i, j, k) such that |i| + |j| + |k| = ref_order
                        // where i = 2*q_x - p_x, j = 2*q_y - p_y, k = 2*q_z - p_z
                        
                        for (int i_abs = 0; i_abs <= ref_order; i_abs++) {
                            for (int j_abs = 0; j_abs <= (ref_order - i_abs); j_abs++) {
                                int k_abs = ref_order - i_abs - j_abs;
                                
                                // Generate all sign combinations for i, j, k
                                int i_values[2];
                                int i_count = (i_abs == 0) ? 1 : 2;
                                if (i_abs == 0) {
                                    i_values[0] = 0;
                                } else {
                                    i_values[0] = -i_abs;
                                    i_values[1] = i_abs;
                                }
                                
                                int j_values[2];
                                int j_count = (j_abs == 0) ? 1 : 2;
                                if (j_abs == 0) {
                                    j_values[0] = 0;
                                } else {
                                    j_values[0] = -j_abs;
                                    j_values[1] = j_abs;
                                }
                                
                                int k_values[2];
                                int k_count = (k_abs == 0) ? 1 : 2;
                                if (k_abs == 0) {
                                    k_values[0] = 0;
                                } else {
                                    k_values[0] = -k_abs;
                                    k_values[1] = k_abs;
                                }
                                
                                for (int i_idx = 0; i_idx < i_count; i_idx++) {
                                    int i = i_values[i_idx];
                                    for (int j_idx = 0; j_idx < j_count; j_idx++) {
                                        int j = j_values[j_idx];
                                        for (int k_idx = 0; k_idx < k_count; k_idx++) {
                                            int k = k_values[k_idx];
                                            
                                            // Parity check (matching Python structure)
                                            if ((i + p_x) % 2 == 0 &&
                                                (j + p_y) % 2 == 0 &&
                                                (k + p_z) % 2 == 0) {
                                                int q_x = (i + p_x) / 2;
                                                int q_y = (j + p_y) / 2;
                                                int q_z = (k + p_z) / 2;
                                                
                                                // Distance calculation optimized for fixed positions
                                                // Distance^2 = Lx^2*(2*q_x-1)^2 + Ly^2*(2*q_y-1)^2 + Lz^2*(2*q_z-1)^2
                                                double dx_factor = 2.0 * q_x - 1.0;
                                                double dy_factor = 2.0 * q_y - 1.0;
                                                double dz_factor = 2.0 * q_z - 1.0;
                                                
                                                double dist_squared = Lx_sq * dx_factor * dx_factor +
                                                                      Ly_sq * dy_factor * dy_factor +
                                                                      Lz_sq * dz_factor * dz_factor;
                                                
                                                // Check if within maximum distance
                                                if (dist_squared < max_distance_squared) {
                                                    count++;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return count;
    }
}

