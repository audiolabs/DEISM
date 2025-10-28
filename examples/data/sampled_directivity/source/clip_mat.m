cap_deg = 50;                        % degree range which are deleted is from pi - cap_rad to pi
cap_rad = deg2rad(cap_deg);         % convert to radians

theta = Dir_all(:,2);               % inclination θ
keepMask = theta <= (pi - cap_rad);
sum(keepMask)        % show how many directions are left
sum(~keepMask)       % show how many directions are deleted
Dir_all_trim = Dir_all(keepMask, :);           % [J_kept x 2]
Psh_trim     = Psh(:, keepMask);               % [491 x J_kept]
freqs_mesh_trim = freqs_mesh;
r0_trim = r0;

Psh        = Psh_trim;
Dir_all    = Dir_all_trim;
freqs_mesh = freqs_mesh_trim;
r0         = r0_trim;

save('Speaker_cuboid_cyldriver_trim50deg_source.mat', ...
     'Psh', 'Dir_all', 'freqs_mesh', 'r0');