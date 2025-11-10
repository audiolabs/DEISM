%% === User parameters ===
sofa_file   = 'D:\Projects\DEISM-main\DEISM-main\examples\data\sampled_directivity\sofa\P0001_FreeFieldComp_48kHz.sofa';  % path to your SOFA file
ear_select  = 1;          % 1 = left ear, 2 = right ear
target_freq = 12187.5;      % Hz (pick any frequency you want)

%% === Load SOFA HRTF ===
hrtf = SOFAload(sofa_file);

IR   = hrtf.Data.IR;            % size: [M, R, N]  (directions x ears x time)
fs   = double(hrtf.Data.SamplingRate);
dirs = hrtf.SourcePosition;     % [M x 3]: [az_deg, el_deg, r_m]

[M, R, N] = size(IR); %#ok<NASGU>

%% === FFT to get frequency response ===
H = fft(IR, [], 3);             % H: [M, R, N] complex
freq_axis = (0:N-1) * (fs / N); % Hz

% find bin closest to target_freq
[~, fi] = min(abs(freq_axis - target_freq));

% choose ear, get complex HRTF for all directions at that freq
H_ear = squeeze(H(:, ear_select, fi));   % [M x 1] complex

% magnitude
mag_lin = abs(H_ear);

% optional normalization so shape is comparable across freqs
mag_lin = mag_lin ./ max(mag_lin + eps);   % scale so max = 1

%% === Convert SOFA spherical coords -> Cartesian ===
% SOFA "spherical" SourcePosition convention:
%   azimuth (deg): angle in horizontal plane, 0° = front ( +X ), +90° = left ( +Y )
%   elevation (deg): angle up from horizontal plane, +90° = +Z (up)
%   radius (m)
%
% We'll convert each measurement direction to a unit vector (x,y,z),
% then scale by mag_lin, so stronger directions stick out more.

az_deg = dirs(:,1);
el_deg = dirs(:,2);
r0     = dirs(:,3);  %#ok<NASGU>  % measurement radius (often constant like 1.4 m)

az = deg2rad(az_deg);
el = deg2rad(el_deg);

% unit direction from SOFA spherical:
% x = cos(el)*cos(az)
% y = cos(el)*sin(az)
% z = sin(el)
ux = cos(el) .* cos(az);
uy = cos(el) .* sin(az);
uz = sin(el);

% now scale each direction vector by magnitude (balloon radius)
X = mag_lin .* ux;
Y = mag_lin .* uy;
Z = mag_lin .* uz;

%% === (A) 3D scatter balloon ===
figure;
scatter3(X, Y, Z, 20, mag_lin, 'filled');
axis equal;
grid on;
xlabel('X (front)');
ylabel('Y (left)');
zlabel('Z (up)');
title(sprintf('Directivity 3D balloon at %.0f Hz (ear %d)', target_freq, ear_select));
colorbar;
view(45,30);  % nicer angle

%% OPTIONAL === (B) Make a smooth surface / hull instead of just scatter
% This uses alphaShape (requires MATLAB's alphaShape + boundary/triangulation tools).
% If you don't have alphaShape, you can skip this block safely.

try
    shp = alphaShape(X, Y, Z, 0.5);  % alpha radius tune: smaller = tighter surface
    [tri, pts] = boundaryFacets(shp);

    figure;
    trisurf(tri, pts(:,1), pts(:,2), pts(:,3), ...
            'FaceColor',[0.4 0.2 0.8], ...
            'FaceAlpha',0.8, ...
            'EdgeColor','none');
    axis equal;
    grid on;
    xlabel('X (front)');
    ylabel('Y (left)');
    zlabel('Z (up)');
    title(sprintf('Smoothed directivity surface at %.0f Hz (ear %d)', target_freq, ear_select));
    view(45,30);
    camlight headlight; lighting gouraud;
catch ME
    warning('Surface plot skipped (alphaShape not available): %s', ME.message);
end
