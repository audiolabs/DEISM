import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.special import sph_harm
from deism.core_deism import *
from deism.data_loader import *
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
from tkinter import Tk, filedialog
import os
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    """Used to create 3D arrows"""

    # Initialize and save 3D coordinate information
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs  # Save 3D vertex coordinates

    # Convert 3D coordinates to 2D coordinates and call the parent class for drawing
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    # Tell matplotlib the depth of the arrows so they are rendered in the correct order
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        return zs[0]  # Returns the z value for depth ordering


def balloon_plot_with_slider(
    data_dir, sh_order=4, initial_freq=500, initial_r0_rec=0.5
):
    """
    Interactive balloon plot with sliders to change frequency and r0_rec.

    Parameters:
    data_dir : str
        Directory containing .mat files (must contain Psh, Dir_all, Freqs_mesh, r0)
    sh_order : int
        Spherical harmonics order
    initial_freq : float
        Initial frequency to display (Hz)
    initial_r0_rec: float
        Initial spherical radius during reconstruction
    """

    # Find all source files in directory
    source_files = [f for f in os.listdir(data_dir) if f.endswith("_source.mat")]
    if not source_files:
        raise ValueError(f"No source files found in {data_dir}")

    # Load first file to initialize data
    initial_file = os.path.join(data_dir, source_files[0])
    mat = loadmat(initial_file)
    Psh = mat["Psh"]
    Dir_all = mat["Dir_all"]
    freqs = mat["freqs_mesh"].squeeze()
    r0 = float(mat["r0"].squeeze())
    # params, _ = cmdArgsToDict()
    k_all = 2 * np.pi * freqs / 343

    # Create figure with controls
    fig = plt.figure(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.35, right=0.85)  # Make room for controls
    # Create axis for the 3D plot
    ax_recon = fig.add_subplot(111, projection="3d")
    # Add phase legend to the figure
    add_phase_legend(fig)
    # Create control axes
    ax_freq = plt.axes([0.2, 0.2, 0.6, 0.03])
    ax_r0 = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_file = plt.axes([0.2, 0.25, 0.6, 0.05])
    ax_browse = plt.axes([0.8, 0.25, 0.1, 0.05])

    # Initialize variables
    current_file = initial_file
    current_file_idx = 0

    # Resolution for the plots
    res = 50
    # Construct a spherical angle grid (azimuth and elevation) to prepare sampling points for drawing
    az = np.linspace(0, 2 * np.pi, 2 * res + 1)  # 2*res evenly spaced samples
    el = np.linspace(0, np.pi, res)
    az_m, el_m = np.meshgrid(az, el)  # az_m:(res, 2*res+1), el_m:(res, 2*res+1)
    # Expand the spherical angles into an array of angles (n_pts, 2), with each row containing one [φ, θ]
    dirs = np.stack(
        [az_m.ravel(), el_m.ravel()], axis=1
    )  # [phi, theta] each row represents point on the sphere, (res)*(2*res+1) points in total

    # precomputation of Ynm, hn, Pnm and Cnm
    Ynm_cache = {}
    for n in range(sh_order + 1):
        for m in range(-n, n + 1):
            Ynm_cache[(n, m)] = sph_harm(m, n, dirs[:, 0], dirs[:, 1])

    hn_cache = {}
    kr_values = [
        k_all[i] * r for i in range(len(freqs)) for r in np.arange(r0, 2.0 + 0.001, 0.1)
    ]  # range of rec_r0 : r0 to 2.0, step 0.1
    kr_values = list(set([round(v, 6) for v in kr_values]))
    for n in range(sh_order + 1):
        for kr in kr_values:
            hn_cache[(n, kr)] = sphankel2(n, kr)

    full_Pnm_cache = {}  # {(freq_idx, sh_order): full_Pnm}
    Cnm_s_cache = {}  # {(freq_idx, sh_order, r0): Cnm_s}
    for freq_idx in range(len(freqs)):
        full_Pnm_cache[(freq_idx, sh_order)] = SHCs_from_pressure_LS(
            Psh[freq_idx].reshape(1, -1), Dir_all, sh_order, np.array([freqs[freq_idx]])
        )
        k = k_all[freq_idx]
        Cnm_s_cache[(freq_idx, sh_order, r0)] = get_directivity_coefs(
            k, sh_order, full_Pnm_cache[(freq_idx, sh_order)], r0
        )

    def browse_file(event):
        # Create the Tkinter root window object
        root = Tk()
        root.withdraw()
        # 计算屏幕中央位置
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = 600  # 对话框预估宽度
        window_height = 400  # 对话框预估高度

        position_x = int((screen_width - window_width) / 2)
        position_y = int((screen_height - window_height) / 2)

        # 设置对话框位置（通过geometry）
        root.geometry(f"+{position_x}+{position_y}")
        # Mac-specific adjustments for proper dialog display
        # root.wm_attributes("-topmost", 1)
        root.focus_force()
        root.lift()
        root.update()

        try:
            # Open the file selection dialog box and configure the parameters:
            file_path = filedialog.askopenfilename(
                parent=root,
                initialdir=os.path.abspath(data_dir),
                title="Select MAT file",
                filetypes=[("MAT files", "*.mat")],
            )
            if file_path:
                # Normalized path format
                selected_dir = os.path.normpath(os.path.dirname(file_path))
                target_dir = os.path.normpath(os.path.abspath(data_dir))

                # Verify that the selected file is in the specified directory
                if selected_dir != target_dir:
                    print(f"Please select a file in the {target_dir} directory.")
                    return

                # Extract the pure file name (without path)
                filename = os.path.basename(file_path)
                # Check if the file is in the list of allowed source files
                if filename in source_files:
                    file_idx = source_files.index(filename)
                    load_file(file_idx)
                else:
                    print(f"Invalid file: {filename}")
        finally:
            # Destroy Tkinter window object to avoid memory leak
            root.destroy()

    # Create browse button
    browse_button = Button(
        ax=ax_browse, label="Browse...", color="lightblue", hovercolor="skyblue"
    )
    browse_button.on_clicked(browse_file)
    # Create a file display button
    file_button = Button(
        ax=ax_file,
        label=f"Current: {os.path.basename(current_file)}",
        color="lightgoldenrodyellow",
    )

    # Frequency slider
    freq_slider = Slider(
        ax=ax_freq,
        label="Frequency (Hz)",
        valmin=freqs.min(),
        valmax=freqs.max(),
        valinit=initial_freq,
        valstep=2,
    )  # Implementing continuous triggering
    # r0_rec slider (range from r0 to 2.0, step size: 0.1)
    r0_slider = Slider(
        ax=ax_r0,
        label="Reconstruction Radius (r0_rec)",
        valmin=r0,
        valmax=2.0,
        valinit=max(r0, initial_r0_rec),
        valstep=0.1,
    )

    def load_file(file_idx):
        nonlocal current_file, current_file_idx, r0, freqs, k_all, Psh, Dir_all, full_Pnm_cache, Cnm_s_cache

        # Clear old cache
        full_Pnm_cache.clear()
        Cnm_s_cache.clear()

        # load new datas
        current_file = os.path.join(data_dir, source_files[file_idx])
        mat = loadmat(current_file)
        Psh = mat["Psh"]
        Dir_all = mat["Dir_all"]
        freqs = mat["freqs_mesh"].squeeze()
        r0 = float(mat["r0"].squeeze())
        # params, _ = cmdArgsToDict()
        k_all = 2 * np.pi * freqs / 343

        # precomputation of full_Pnm and Cnm_s
        for freq_idx in range(len(freqs)):
            full_Pnm_cache[(freq_idx, sh_order)] = SHCs_from_pressure_LS(
                Psh[freq_idx].reshape(1, -1),
                Dir_all,
                sh_order,
                np.array([freqs[freq_idx]]),
            )
            k = k_all[freq_idx]
            Cnm_s_cache[(freq_idx, sh_order, r0)] = get_directivity_coefs(
                k, sh_order, full_Pnm_cache[(freq_idx, sh_order)], r0
            )

        # Update frequency slider
        freq_slider.valmin = freqs.min()
        freq_slider.valmax = freqs.max()
        # freq_slider.valstep = freqs[1]-freqs[0]
        freq_slider.set_val(freqs[len(freqs) // 2])  # Set to middle frequency
        # Update r0 slider
        r0_slider.set_val(max(r0, initial_r0_rec))
        # Update button label
        file_button.label.set_text(f"Current: {os.path.basename(current_file)}")

        update(None)  # Refresh plot

    # Function to update plot when sliders change
    def update(val):
        if current_file is None:
            return

        # get current freq and r0
        freq = freq_slider.val
        r0_rec = r0_slider.val

        freq_idx = np.argmin(np.abs(freqs - freq))
        actual_freq = freqs[freq_idx]

        # Clear previous plot
        ax_recon.cla()

        # get Cnm_s from Cnm_s_cache
        Cnm_s = Cnm_s_cache[(freq_idx, sh_order, r0)]

        # Reconstruct with current r0_rec
        # Recalculate the new Pnm using r0_rec according to formula: Pnm = Cnm_s * hn_r0
        Pnm_rec = np.zeros_like(Cnm_s, dtype=complex)
        for n in range(sh_order + 1):  # Change to different sh_order
            hn_r0_rec = hn_cache[(n, round(k * r0_rec, 6))]
            for m in range(-n, n + 1):
                Pnm_rec[:, n, m + n] = Cnm_s[:, n, m] * hn_r0_rec

        # Plot reconstructed
        plot_balloon_rec(
            ax_recon,
            sh_order,
            Pnm_rec[0],
            dirs,
            az_m,
            el_m,
            f"Reconstructed {int(actual_freq)} Hz, r0_rec={r0_rec:.1f}",
            Ynm_cache,
        )

        # Request to refresh the images when the GUI is idle to improve interaction efficiency
        fig.canvas.draw_idle()

    # Connect events
    freq_slider.on_changed(update)
    r0_slider.on_changed(update)

    # Draw initial plot
    update(None)

    plt.show()


def add_phase_legend(fig):
    """Add phase colorbar legend to the figure"""
    # Create a colormap object representing the HSV hue mapping
    cmap = plt.get_cmap("twilight")

    # Create a new axis to display the color bar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cb = mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, orientation="vertical"
    )

    cb.set_label("Phase (radians)")
    cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])


def plot_balloon_rec(ax, order, Pnm, dirs, az_m, el_m, title, Ynm_cache):
    """plot a reconstructed balloon plot"""
    # Initialize the direction response vector D
    D = np.zeros(dirs.shape[0], dtype=complex)

    # Iterate over each degree and order n, m; compute Ynm spherical harmonics function; Weight it using the SHCs; Obtain the directional response D (sound field) for all directions
    for n in range(order + 1):
        for m in range(-n, n + 1):
            Ynm = Ynm_cache[(n, m)]
            D += Pnm[n, m + n] * Ynm  # size = (res)*(2*res+1)

    # Magnitude
    D_abs = np.abs(D)
    D_plot = D_abs.reshape(
        el_m.shape
    )  # (elevation, azimuth), D_abs is one-dimensional, reshape to 2D (res, 2*res+1)
    # Phase (in radians, unwrapped)
    D_phase = np.angle(D)  # the range of np.angle(D) is [-π, π]
    D_phase_plot = D_phase.reshape(el_m.shape)

    # Amplitude and phase normalized to [0,1]
    norm_amp = D_plot / D_plot.max()
    norm_phase = (D_phase_plot + np.pi) / (2 * np.pi)

    # Spherical coordinates -> Cartesian coordinates
    x = norm_amp * np.sin(el_m) * np.cos(az_m)
    y = norm_amp * np.sin(el_m) * np.sin(az_m)
    z = norm_amp * np.cos(el_m)

    # color mapping
    cmap = plt.get_cmap("twilight")
    colors = cmap(norm_phase)

    # Plot
    ax.set_axis_on()
    lim = 1

    # Set the tick range and labels
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ticks = np.linspace(-lim, lim, 3)  # Generate 3 ticks
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])

    # Add axes indicators, Triangle Arrow
    arrow_len = 1.2
    for vec, label in zip(
        [(arrow_len, 0, 0), (0, arrow_len, 0), (0, 0, arrow_len)], ["x", "y", "z"]
    ):
        arrow = Arrow3D(
            [0, vec[0]],
            [0, vec[1]],
            [0, vec[2]],
            mutation_scale=20,
            lw=2,
            arrowstyle="-|>",
            color="gray",
        )
        ax.add_artist(arrow)
        ax.text(
            vec[0] * 1.05,
            vec[1] * 1.05,
            vec[2] * 1.05,
            label,
            color="black",
            fontsize=12,
        )

    ax.set_proj_type("ortho")  # Use orthographic projection
    ax.view_init(elev=30, azim=45)  # Set the default viewing angle


if __name__ == "__main__":
    balloon_plot_with_slider(
        data_dir=os.path.join(
            "examples", "data", "sampled_directivity", "source"
        ),  # Cross-platform path
        sh_order=6,
        initial_freq=500,
        initial_r0_rec=0.6,
        # if_show_info=True,  # If show the running information in a separate window
    )
