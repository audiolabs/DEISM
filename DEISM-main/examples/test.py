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
import tkinter as tk
from tkinter import Tk, filedialog, ttk
import os
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageTk
import io
import urllib.request
from scipy.interpolate import interp1d
import time
from netCDF4 import Dataset


def sofa_to_internal(sofa_path, target_freqs=None, ear="L"):
    """
    Convert a SOFA HRTF/HRIR file to the internal (Psh, Dir_all, freqs, r0) format

    Returns:
      Psh     : (F, J) complex    # frequency x directions(F = # of frequency point, J = # of sampling points)
      Dir_all : (J, 2) float rad  # columns: [az, inc], with inc = pi/2 - el
      freqs   : (F,) float        # frequency axis (Hz) after optional interpolation
      r0      : float             # reference radius (m)
    """
    ds = Dataset(sofa_path, "r")

    # 1) directions (deg) → radians, and convert elevation→inclination
    src = np.array(ds.variables["SourcePosition"])  # (J,3) [az_deg, el_deg, r_m]
    az_deg, el_deg, r_m = src[:, 0], src[:, 1], src[:, 2]
    az = np.deg2rad(az_deg)
    inc = np.deg2rad(90.0 - el_deg)  # inc = 90° - el
    Dir_all = np.c_[az, inc].astype(float)

    # radius: often constant in SOFA; take median as r0
    r0 = float(np.median(r_m))

    # 2) time-domain HRIR → frequency response H(f)
    ir = np.array(ds.variables["Data.IR"])  # typical shape (J, R, N)
    fs = float(np.array(ds.variables["Data.SamplingRate"]).squeeze())
    ds.close()

    # choose ear (0=left, 1=right)
    ear_idx = 0 if str(ear).upper().startswith("L") else 1
    if ir.ndim != 3 or ir.shape[1] < 2:
        # fallback: if dataset is mono or layout differs, just take channel 0
        ear_idx = 0
    J, _, N = ir.shape if ir.ndim == 3 else (ir.shape[0], 1, ir.shape[-1])
    ir_ear = ir[:, ear_idx, :] if ir.ndim == 3 else ir

    # rFFT per direction → shape (J, F)
    H_dirF = np.fft.rfft(ir_ear, axis=-1)  # (J, F_sofa)
    f_sofa = np.fft.rfftfreq(N, 1 / fs)  # (F_sofa,)

    # 3) arrange to (F, J)
    H_FJ = H_dirF.T.astype(complex)  # (F_sofa, J)

    # --- drop DC (0 Hz), which makes k=0 and h_n^(2)(0) singular ---
    if f_sofa[0] == 0.0:
        H_FJ = H_FJ[1:, :]
        f_sofa = f_sofa[1:]

    # 4) interpolate complex response to target_freqs
    if target_freqs is not None:
        tf = np.asarray(target_freqs, float).ravel()
        # separate real/imag for safe interpolation
        Hr = np.empty((tf.size, H_FJ.shape[1]), dtype=float)
        Hi = np.empty_like(Hr)
        for j in range(H_FJ.shape[1]):
            Hr[:, j] = np.interp(tf, f_sofa, H_FJ.real[:, j])
            Hi[:, j] = np.interp(tf, f_sofa, H_FJ.imag[:, j])
        H_FJ = Hr + 1j * Hi
        freqs = tf
    else:
        freqs = f_sofa

    Psh = H_FJ  # naming aligned
    return Psh, Dir_all, freqs, r0


# Audiolabs color scheme
AUDIOLABS_ORANGE = "#F15A24"
AUDIOLABS_GRAY = "#5B6770"
AUDIOLABS_LIGHT_GRAY = "#E6E6E6"
AUDIOLABS_DARK_GRAY = "#333333"
AUDIOLABS_WHITE = "#FFFFFF"
AUDIOLABS_BLACK = "#000000"


def configure_global_styles():
    """Used to unify interface style"""

    if not hasattr(configure_global_styles, "_called"):
        style = ttk.Style()
        style.theme_use("clam")

        # Progressbar style
        style.configure(
            "Audiolabs.Horizontal.TProgressbar",
            troughcolor=AUDIOLABS_LIGHT_GRAY,
            background=AUDIOLABS_ORANGE,
            bordercolor=AUDIOLABS_GRAY,
        )

        # label style
        style.configure(
            "TLabel",
            background=AUDIOLABS_WHITE,
            foreground=AUDIOLABS_BLACK,
            font=("Arial", 10),
        )

        configure_global_styles._called = True


def load_audiolabs_logo():
    """Load Audiolabs logo while preserving aspect ratio"""
    try:
        logo_path = os.path.join("examples", "audiolabs_logo.png")
        logo_image = Image.open(logo_path)

        # calculate original ratio of width and height
        original_width, original_height = logo_image.size
        aspect_ratio = original_width / original_height

        # Scale by height, keeping proportions
        new_height = 40
        new_width = int(new_height * aspect_ratio)
        logo_image = logo_image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )

        return logo_image
    except Exception as e:
        print(f"Logo loading failed: {e}")
        return Image.new("RGB", (100, 40), color=AUDIOLABS_WHITE)


class InitializationWindow:
    """Used to display the Progress-Window"""

    def __init__(self, title="Progress"):
        configure_global_styles()

        self.root = tk.Toplevel()
        self.root.title(title)
        self.root.geometry("400x150")
        self.root.configure(bg=AUDIOLABS_WHITE)

        self.label = ttk.Label(
            self.root,
            text="",
            font=("Arial", 12),
            foreground=AUDIOLABS_BLACK,
            background=AUDIOLABS_WHITE,
        )
        self.label.pack(pady=10)

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=300,
            mode="determinate",
            style="Audiolabs.Horizontal.TProgressbar",
        )
        self.progress.pack(pady=10)

        self.status_label = ttk.Label(
            self.root,
            text="",
            font=("Arial", 10),
            foreground=AUDIOLABS_DARK_GRAY,
            background=AUDIOLABS_WHITE,
        )
        self.status_label.pack(pady=5)

    def update_progress(self, value, message, status=""):
        self.progress["value"] = value
        self.label.config(width=60, anchor="center", text=message)
        self.status_label.config(width=60, anchor="center", text=status)
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()


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


def interpolate_Cnm(freqs, Cnm_s_cache, sh_order, r0, target_freq):
    """Used to interpolate for Cnm"""

    freq_array = np.array(freqs)
    Cnm_array = np.array(
        [Cnm_s_cache[(i, sh_order, r0)] for i in range(len(freqs))]
    )  # shape: (n_freq, N+1, 2N+1)

    # Cubic spline interpolation; Cnm is complex, so interpolate real and imag separately
    interp_real = interp1d(freq_array, Cnm_array.real, axis=0, kind="cubic")
    interp_imag = interp1d(freq_array, Cnm_array.imag, axis=0, kind="cubic")

    Cnm_interp = interp_real(target_freq) + 1j * interp_imag(target_freq)
    return Cnm_interp


def is_receiver_file(name: str) -> bool:
    """Used to judge if the data file is receiver file"""
    return name.endswith("_receiver.mat")


def balloon_plot_with_slider(
    source_dir,
    receiver_dir,
    sh_order=4,
    initial_freq=500,
    initial_r0_rec=0.5,
    progress_window=None,
    params=None,
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
    is_initial_draw = True
    is_sofa_file = False
    Psh_raw = None

    # Show initialization window
    progress_window.update_progress(25, "Initializing...", "Scanning files")
    time.sleep(0.5)

    # Find all source and receiver files in directory
    abs_source_dir = os.path.abspath(source_dir)
    abs_receiver_dir = os.path.abspath(receiver_dir)
    abs_sofa_dir = os.path.abspath(
        os.path.join("examples", "data", "sampled_directivity", "sofa")
    )
    ALLOWED_DIRS = {abs_source_dir, abs_receiver_dir, abs_sofa_dir}
    BASE_DIR = os.path.commonpath(list(ALLOWED_DIRS))

    def list_files(d, exts):
        return [f for f in os.listdir(d) if f.lower().endswith(exts)]

    source_files = [
        f for f in list_files(abs_source_dir, (".mat",)) if f.endswith("_source.mat")
    ]
    receiver_files = [
        f
        for f in list_files(abs_receiver_dir, (".mat",))
        if f.endswith("_receiver.mat")
    ]
    sofa_files = [f for f in list_files(abs_sofa_dir, (".sofa",))]

    allowed_files = source_files + receiver_files + sofa_files

    # basename -> full path
    file_lookup = {}
    for f in source_files:
        file_lookup[f] = os.path.join(abs_source_dir, f)
    for f in receiver_files:
        file_lookup[f] = os.path.join(abs_receiver_dir, f)
    for f in sofa_files:
        file_lookup[f] = os.path.join(abs_sofa_dir, f)

    if not allowed_files:
        progress_window.close()
        raise ValueError(f"No source/receiver files found.")

    # Load first file to initialize data
    first_name = source_files[0] if source_files else allowed_files[0]
    initial_file = file_lookup[first_name]
    progress_window.update_progress(
        35, "Initializing...", f"Loading initial file: {first_name}"
    )
    time.sleep(0.5)

    mat = loadmat(initial_file)
    Psh = mat["Psh"]
    Psh_raw = Psh.copy()  # keep pristine copy
    Dir_all = mat["Dir_all"]
    freqs = mat["freqs_mesh"].squeeze()
    r0 = float(mat["r0"].squeeze())
    max_sh_order = sh_order
    k_all = 2 * np.pi * freqs / 343

    current_file = initial_file
    current_file_idx = allowed_files.index(first_name)
    current_is_receiver = is_receiver_file(first_name)
    S = params["pointSrcStrength"] if current_is_receiver else None

    # Create figure with Audiolabs styling
    plt.style.use("seaborn-v0_8")  # Start with a clean style
    fig = plt.figure(figsize=(6, 8), facecolor=AUDIOLABS_WHITE)
    plt.subplots_adjust(bottom=0.35, right=0.85)
    # Create axis for the 3D plot
    ax_recon = fig.add_subplot(111, projection="3d")
    ax_recon.view_init(elev=30, azim=45)  # Set the initial viewing angle
    # Add phase legend
    add_phase_legend(fig)
    # Create control axes with Audiolabs styling ([left, bottom, width, height])
    control_bg_color = AUDIOLABS_LIGHT_GRAY
    ax_freq = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor=control_bg_color)
    ax_freq_left = plt.axes([0.812, 0.182, 0.02, 0.02], facecolor=control_bg_color)
    ax_freq_right = plt.axes([0.832, 0.182, 0.02, 0.02], facecolor=control_bg_color)
    ax_r0 = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=control_bg_color)
    ax_sh = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=control_bg_color)
    ax_browse = plt.axes([0.2, 0.25, 0.6, 0.05], facecolor=control_bg_color)

    ax_freq_input = plt.axes([0.81, 0.25, 0.085, 0.03])
    fmt = lambda f: f"{f:.1f}"
    text_box_freq = mpl.widgets.TextBox(ax_freq_input, "", initial=fmt(initial_freq))
    fig.text(0.895, 0.257, "Hz", fontsize=10, color=AUDIOLABS_BLACK)

    # Add checkbox for receiver directivity normalization
    ax_norm = plt.axes([0.2, 0.04, 0.3, 0.03], facecolor=AUDIOLABS_LIGHT_GRAY)
    norm_checkbox = mpl.widgets.CheckButtons(
        ax_norm, ["Normalize Receiver"], [bool(params.get("ifReceiverNormalize", 0))]
    )

    def toggle_normalize(label):
        """Used to judge if Psh should be normalized"""
        nonlocal Psh, full_Pnm_cache, Cnm_s_cache, is_sofa_file

        if is_sofa_file:
            fig.canvas.draw_idle()
            return

        # Flip flag
        params["ifReceiverNormalize"] = 1 - params.get("ifReceiverNormalize", 0)

        # Only meaningful for receiver files
        if not current_is_receiver:
            fig.canvas.draw_idle()
            return

        # Rebuild caches from pristine Psh_raw to avoid double-dividing
        Psh = Psh_raw.copy()

        # Handle S as scalar or per-frequency array
        S_val = params.get("pointSrcStrength", 1.0)
        S_arr = np.array(S_val).squeeze()

        full_Pnm_cache.clear()
        Cnm_s_cache.clear()

        for freq_idx in range(len(freqs)):
            # Apply normalization only if flag is ON
            if params.get("ifReceiverNormalize", 0):
                if S_arr.ndim == 0:
                    Psh_use = Psh[freq_idx] / S_arr
                else:
                    Psh_use = Psh[freq_idx] / S_arr[freq_idx]
            else:
                Psh_use = Psh[freq_idx]

            full_Pnm_cache[(freq_idx, sh_order)] = SHCs_from_pressure_LS(
                Psh_use.reshape(1, -1), Dir_all, sh_order, np.array([freqs[freq_idx]])
            )
            k = k_all[freq_idx]
            Cnm_s_cache[(freq_idx, sh_order, r0)] = get_directivity_coefs(
                k, sh_order, full_Pnm_cache[(freq_idx, sh_order)], r0
            )

        update(None)  # now the redraw uses the rebuilt caches

    norm_checkbox.on_clicked(toggle_normalize)

    # When enter a number in the input box and press Enter, this callback function is called
    def on_text_submit(text):
        try:
            val = float(text)
            if freqs.min() <= val <= freqs.max():
                freq_slider.set_val(val)  # trigger update()
            else:
                print("Input frequency out of range")
        except:
            print("Please input a valid frequency")

    text_box_freq.on_submit(on_text_submit)

    # Resolution for the plots
    res = 50
    # Construct a spherical angle grid (azimuth and elevation) to prepare sampling points for drawing
    az = np.linspace(0, 2 * np.pi, 2 * res + 1)  # 2*res evenly spaced samples
    el = np.linspace(0, np.pi, res)
    az_m, el_m = np.meshgrid(az, el)  # az_m:(res, 2*res+1), el_m:(res, 2*res+1)
    # Expand the spherical angles into an array of angles (n_pts, 2), with each row containing one [φ, θ]
    # [phi, theta] each row represents point on the sphere, (res)*(2*res+1) points in total
    dirs = np.stack([az_m.ravel(), el_m.ravel()], axis=1)

    # precomputation of Ynm, hn, Pnm and Cnm
    start_step = time.perf_counter()
    Ynm_cache = {}
    for n in range(sh_order + 1):
        for m in range(-n, n + 1):
            Ynm_cache[(n, m)] = sph_harm(m, n, dirs[:, 0], dirs[:, 1])
    ynm_max_order = sh_order
    elapsed_step = time.perf_counter() - start_step
    progress_window.update_progress(45, "Precomputing Ynm...", f"{elapsed_step:.3f} s")
    time.sleep(0.5)

    start_step = time.perf_counter()
    hn_cache = {}
    kr_values = [
        k_all[i] * r for i in range(len(freqs)) for r in np.arange(r0, 2.0 + 0.001, 0.1)
    ]  # range of rec_r0 : r0 to 2.0, step 0.1
    kr_values = list(set([round(v, 6) for v in kr_values]))
    for n in range(sh_order + 1):
        for kr in kr_values:
            hn_cache[(n, kr)] = sphankel2(n, kr)
    elapsed_step = time.perf_counter() - start_step
    progress_window.update_progress(55, "Precomputing hn...", f"{elapsed_step:.3f} s")
    time.sleep(0.5)

    full_Pnm_cache = {}  # {(freq_idx, sh_order): full_Pnm}
    Cnm_s_cache = {}  # {(freq_idx, sh_order, r0): Cnm_s}
    for freq_idx in range(len(freqs)):
        start_step = time.perf_counter()

        if current_is_receiver and params.get("ifReceiverNormalize", 0):
            S_val = params.get("pointSrcStrength", 1.0)
            S_arr = np.array(S_val).squeeze()
            if S_arr.ndim == 0:
                Psh_use = Psh_raw[freq_idx] / S_arr
            else:
                Psh_use = Psh_raw[freq_idx] / S_arr[freq_idx]
        else:
            Psh_use = Psh_raw[freq_idx]

        full_Pnm_cache[(freq_idx, sh_order)] = SHCs_from_pressure_LS(
            Psh_use.reshape(1, -1), Dir_all, sh_order, np.array([freqs[freq_idx]])
        )
        k = k_all[freq_idx]
        Cnm_s_cache[(freq_idx, sh_order, r0)] = get_directivity_coefs(
            k, sh_order, full_Pnm_cache[(freq_idx, sh_order)], r0
        )
        elapsed_step = time.perf_counter() - start_step
        progress_window.update_progress(
            65 + freq_idx / len(freqs) * 30,
            f"Precomputing Pnm & Cnm: freq {freqs[freq_idx]:.1f} Hz...",
            f"{elapsed_step:.3f} s",
        )

    # Close initialization window when done
    progress_window.update_progress(
        100, "Precomputation complete!", "Initialization complete!"
    )
    time.sleep(0.5)
    progress_window.close()

    def browse_file(event):
        # Create the Tkinter root window object
        root = tk.Tk()
        root.withdraw()
        # macOS compatible
        root.lift()
        root.focus_force()

        try:
            # Open the file selection dialog box and configure the parameters:
            file_path = filedialog.askopenfilename(
                # parent=root,
                initialdir=BASE_DIR,
                title="Select MAT/SOFA file",
                filetypes=[
                    ("MAT/SOFA files", "*.mat *.sofa"),
                    ("MAT files", "*.mat"),
                    ("SOFA files", "*.sofa"),
                ],
            )
            if file_path:
                # Normalized path format
                selected_dir = os.path.normpath(os.path.dirname(file_path))

                # Verify that the selected file is in the specified directory
                if selected_dir not in ALLOWED_DIRS:
                    print("Please select a file under one of these directories:")
                    print(" -", abs_source_dir)
                    print(" -", abs_receiver_dir)
                    return

                # Extract the pure file name (without path)
                filename = os.path.basename(file_path)
                # Check if the file is in the list of allowed files
                if filename in allowed_files:
                    file_idx = allowed_files.index(filename)
                    load_file(file_idx)
                else:
                    print(f"Invalid file: {filename}")
        finally:
            # Destroy Tkinter window object to avoid memory leak
            root.destroy()

    def on_freq_left(event):
        new_val = freq_slider.val - 2
        if new_val >= freq_slider.valmin:
            freq_slider.set_val(new_val)

    def on_freq_right(event):
        new_val = freq_slider.val + 2
        if new_val <= freq_slider.valmax:
            freq_slider.set_val(new_val)

    # Create buttons with Audiolabs styling
    btn_browse = Button(
        ax=ax_browse,
        label=f"Current: {os.path.basename(current_file)}",
        color=AUDIOLABS_LIGHT_GRAY,
        hovercolor=AUDIOLABS_ORANGE,
    )
    btn_browse.on_clicked(browse_file)

    # update displayed text (on mouse hover)
    def on_hover(event):
        if ax_browse.contains(event)[0]:
            ax_browse.patch.set_facecolor(AUDIOLABS_ORANGE)
            btn_browse.label.set_text("Browse...")
        else:
            ax_browse.patch.set_facecolor(AUDIOLABS_LIGHT_GRAY)
            btn_browse.label.set_text(f"Current: {os.path.basename(current_file)}")
        ax_browse.figure.canvas.draw_idle()

    # Listening for mouse movement events
    btn_browse.ax.figure.canvas.mpl_connect("motion_notify_event", on_hover)

    btn_freq_left = Button(
        ax_freq_left, label="<", color=AUDIOLABS_GRAY, hovercolor=AUDIOLABS_ORANGE
    )
    btn_freq_right = Button(
        ax_freq_right, label=">", color=AUDIOLABS_GRAY, hovercolor=AUDIOLABS_ORANGE
    )
    btn_freq_left.on_clicked(on_freq_left)
    btn_freq_right.on_clicked(on_freq_right)

    # Create sliders with Audiolabs styling
    slider_params = {
        "facecolor": AUDIOLABS_ORANGE,
        "track_color": AUDIOLABS_LIGHT_GRAY,
        "edgecolor": AUDIOLABS_GRAY,
        "alpha": 0.8,
    }

    freq_slider = Slider(
        ax=ax_freq,
        label="",
        valmin=freqs.min(),
        valmax=freqs.max(),
        valinit=initial_freq,
        valstep=2,
        **slider_params,
    )
    fig.text(
        0.2,
        0.197,
        "Frequency (Hz)",
        ha="left",
        va="top",
        fontsize=10,
        color=AUDIOLABS_BLACK,
    )

    r0_slider = Slider(
        ax=ax_r0,
        label="",
        valmin=r0,
        valmax=2.0,
        valinit=max(r0, initial_r0_rec),
        valstep=0.1,
        **slider_params,
    )
    fig.text(
        0.2,
        0.147,
        "Reconstruction Radius (m)",
        ha="left",
        va="top",
        fontsize=10,
        color=AUDIOLABS_BLACK,
    )

    sh_slider = Slider(
        ax=ax_sh,
        label="",
        valmin=0,
        valmax=max_sh_order,
        valinit=sh_order,
        valstep=1,
        **slider_params,
    )
    fig.text(
        0.2,
        0.097,
        "Spherical Harmonics Order",
        ha="left",
        va="top",
        fontsize=10,
        color=AUDIOLABS_BLACK,
    )

    # Add Audiolabs logo to the figure
    try:
        logo_img = load_audiolabs_logo()
        logo_width, logo_height = logo_img.size
        logo_display_height = 0.06
        logo_display_width = logo_display_height * (logo_width / logo_height)

        logo_ax = fig.add_axes([0.02, 0.85, logo_display_width, logo_display_height])
        logo_ax.imshow(logo_img)
        logo_ax.axis("off")
    except Exception as e:
        print(f"Could not load logo: {e}")

    def load_file(file_idx):
        nonlocal current_file, current_file_idx, r0, freqs, k_all, Psh, Dir_all, hn_cache, Ynm_cache, full_Pnm_cache
        nonlocal Cnm_s_cache, current_is_receiver, S, Psh_raw, is_initial_draw, max_sh_order, ynm_max_order, is_sofa_file

        current_file = file_lookup[allowed_files[file_idx]]
        current_file_idx = file_idx

        is_sofa_file = os.path.splitext(current_file)[1].lower() == ".sofa"

        is_initial_draw = True

        # Create progress window
        progress_win = InitializationWindow("Loading File Progress")
        progress_win.update_progress(
            0, "Starting...", f"Loading new file: {allowed_files[file_idx]}"
        )
        time.sleep(0.5)

        # Clear old cache
        full_Pnm_cache.clear()
        Cnm_s_cache.clear()

        # load new datas
        ext = os.path.splitext(current_file)[1].lower()
        if ext == ".sofa":
            # SOFA branch
            target_f = None  # using freq range of SOFA file
            Psh, Dir_all, freqs, r0 = sofa_to_internal(
                current_file, target_freqs=target_f, ear="L"
            )
            Psh_raw = Psh.copy()
            k_all = 2 * np.pi * freqs / 343.0
            current_is_receiver = True
            params["ifReceiverNormalize"] = 0

            # after loading the new file, limit max order based on J
            J = Dir_all.shape[0]
            max_by_J = int(np.floor(np.sqrt(J)) - 1)  # make sure (N+1)^2 <= J
            max_by_kr = int(np.floor((k_all.max() * r0)) + 2)  # experience cap of kr
            max_sh_order = max(0, min(max_by_J, max_by_kr))
            # update cap of sh_slider
            sh_slider.valmax = max_sh_order
            sh_slider.ax.set_xlim(0, max_sh_order)
            if sh_slider.val > max_sh_order:
                sh_slider.set_val(max_sh_order)

            # recalculate hn function
            hn_cache.clear()
            kr_values = [
                (k_all[i]) * r
                for i in range(len(freqs))
                for r in np.arange(r0, 2.0 + 1e-6, 0.1)
            ]
            kr_values = list(set([round(v, 6) for v in kr_values]))
            for n in range(max_sh_order + 1):
                for kr in kr_values:
                    hn_cache[(n, kr)] = sphankel2(n, kr)

            if max_sh_order > ynm_max_order:
                # Only fill in the new order to avoid repeated calculations
                for n in range(ynm_max_order + 1, max_sh_order + 1):
                    for m in range(-n, n + 1):
                        Ynm_cache[(n, m)] = sph_harm(m, n, dirs[:, 0], dirs[:, 1])
                ynm_max_order = max_sh_order
        else:
            # MAT branch
            mat = loadmat(current_file)
            Psh = mat["Psh"]
            Psh_raw = Psh.copy()
            Dir_all = mat["Dir_all"]
            freqs = mat["freqs_mesh"].squeeze()
            r0 = float(mat["r0"].squeeze())
            k_all = 2 * np.pi * freqs / 343.0
            current_is_receiver = is_receiver_file(allowed_files[file_idx])
            S = params["pointSrcStrength"] if current_is_receiver else None

        # precomputation of full_Pnm and Cnm_s
        for freq_idx in range(len(freqs)):
            start_step = time.perf_counter()
            if current_is_receiver and params.get("ifReceiverNormalize", 0):
                S_val = params.get("pointSrcStrength", 1.0)
                S_arr = np.array(S_val).squeeze()
                if S_arr.ndim == 0:
                    Psh_use = Psh_raw[freq_idx] / S_arr
                else:
                    Psh_use = Psh_raw[freq_idx] / S_arr[freq_idx]
            else:
                Psh_use = Psh_raw[freq_idx]

            full_Pnm_cache[(freq_idx, max_sh_order)] = SHCs_from_pressure_LS(
                Psh_use.reshape(1, -1),
                Dir_all,
                max_sh_order,
                np.array([freqs[freq_idx]]),
            )
            k = k_all[freq_idx]
            Cnm_s_cache[(freq_idx, max_sh_order, r0)] = get_directivity_coefs(
                k, max_sh_order, full_Pnm_cache[(freq_idx, max_sh_order)], r0
            )
            elapsed_step = time.perf_counter() - start_step
            progress_win.update_progress(
                35 + freq_idx / len(freqs) * 60,
                f"Precomputing Pnm & Cnm: freq {freqs[freq_idx]:.1f} Hz...",
                f"{elapsed_step:.3f} s",
            )

        progress_win.update_progress(
            95,
            "Updating parameters on the interface...",
            "updating sliders and buttons",
        )
        # Update frequency slider
        freq_slider.valmin = freqs.min()
        freq_slider.valmax = freqs.max()
        freq_slider.ax.set_xlim(freqs.min(), freqs.max())
        # Set to middle frequency
        freq_slider.set_val(freqs[len(freqs) // 2])
        # Update r0 slider
        r0_slider.set_val(max(r0, initial_r0_rec))
        # Update button label
        btn_browse.label.set_text(f"Current: {os.path.basename(current_file)}")
        time.sleep(0.5)

        progress_win.update_progress(100, "Ready!", "File loading complete!")
        time.sleep(0.5)
        progress_win.close()

        update(None)

    def update(val):
        nonlocal is_initial_draw

        if current_file is None:
            return

        # get current freq, r0 and sh_order
        freq = freq_slider.val
        r0_rec = r0_slider.val
        current_sh_order = int(sh_slider.val)

        freq_idx = np.argmin(np.abs(freqs - freq))

        # Clear previous plot
        ax_recon.cla()

        if is_initial_draw:
            # get Cnm_s from Cnm_s_cache
            Cnm_s = Cnm_s_cache[(freq_idx, max_sh_order, r0)][
                :, : current_sh_order + 1, :
            ]
        else:
            # get Cnm_s with interpolation, (1, sh_order + 1, 2 * sh_order + 1)
            Cnm_s = interpolate_Cnm(freqs, Cnm_s_cache, max_sh_order, r0, freq)
            Cnm_s = Cnm_s[:, : current_sh_order + 1, :]

        # Reconstruct with current r0_rec
        # Recalculate the new Pnm using r0_rec according to formula: Pnm = Cnm_s * hn_r0
        Pnm_rec = np.zeros(
            (1, current_sh_order + 1, 2 * current_sh_order + 1), dtype=complex
        )
        k = k_all[freq_idx]

        for n in range(current_sh_order + 1):
            hn_r0_rec = hn_cache[(n, round(k * r0_rec, 6))]
            for m in range(-n, n + 1):
                Pnm_rec[:, n, m + n] = Cnm_s[:, n, m] * hn_r0_rec

        # Plot reconstructed
        plot_balloon_rec(
            ax_recon,
            current_sh_order,
            Pnm_rec[0],
            dirs,
            az_m,
            el_m,
            f"Reconstructed balloon plot",
            Ynm_cache,
        )

        # Request to refresh the images when the GUI is idle to improve interaction efficiency
        fig.canvas.draw_idle()
        # Synchronize text box value
        text_box_freq.set_val(fmt(freq))

        is_initial_draw = False

    # Connect events
    freq_slider.on_changed(update)
    r0_slider.on_changed(update)
    sh_slider.on_changed(update)

    # Draw initial plot
    update(None)

    plt.show()


def add_phase_legend(fig):
    """Add phase colorbar legend to the figure"""
    # Create a colormap object representing the HSV hue mapping
    cmap = plt.get_cmap("twilight")

    # Create a new axis to display the color bar
    cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.6])
    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cb = mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, orientation="vertical"
    )

    cb.set_label("Phase (radians)", color=AUDIOLABS_BLACK)
    cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

    # Set colorbar styling
    cbar_ax.tick_params(colors=AUDIOLABS_BLACK)
    cbar_ax.yaxis.label.set_color(AUDIOLABS_BLACK)
    cbar_ax.spines["top"].set_color(AUDIOLABS_GRAY)
    cbar_ax.spines["bottom"].set_color(AUDIOLABS_GRAY)
    cbar_ax.spines["left"].set_color(AUDIOLABS_GRAY)
    cbar_ax.spines["right"].set_color(AUDIOLABS_GRAY)


def plot_balloon_rec(ax, order, Pnm, dirs, az_m, el_m, title, Ynm_cache):
    """plot a reconstructed balloon plot with Audiolabs styling"""
    # Initialize the direction response vector D
    D = np.zeros(dirs.shape[0], dtype=complex)

    # Iterate over each degree and order n, m; compute Ynm spherical harmonics function; Weight it using the SHCs; Obtain the directional response D (sound field) for all directions
    for n in range(order + 1):
        for m in range(-n, n + 1):
            Ynm = Ynm_cache[(n, m)]
            D += Pnm[n, m + n] * Ynm  # size = (res)*(2*res+1)

    # Magnitude
    D_abs = np.abs(D)
    # (elevation, azimuth), D_abs is one-dimensional, reshape to 2D (res, 2*res+1)
    D_plot = D_abs.reshape(el_m.shape)
    # Phase (in radians, unwrapped), the range of np.angle(D) is [-π, π]
    D_phase = np.angle(D)
    D_phase_plot = D_phase.reshape(el_m.shape)

    # Amplitude and phase normalized to [0,1]
    maxval = D_plot.max()
    if maxval < 1e-12:  # zero guard to avoid 0/0 → NaN
        maxval = 1.0
    norm_amp = D_plot / maxval
    norm_phase = (D_phase_plot + np.pi) / (2 * np.pi)

    # Spherical coordinates -> Cartesian coordinates
    x = norm_amp * np.sin(el_m) * np.cos(az_m)
    y = norm_amp * np.sin(el_m) * np.sin(az_m)
    z = norm_amp * np.cos(el_m)

    # color mapping
    cmap = plt.get_cmap("twilight")
    colors = cmap(norm_phase)

    # Plot; Set the tick range and labels
    lim = 1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ticks = np.linspace(-lim, lim, 3)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_xlabel("X", color=AUDIOLABS_BLACK)
    ax.set_ylabel("Y", color=AUDIOLABS_BLACK)
    ax.set_zlabel("Z", color=AUDIOLABS_BLACK)

    ax.set_facecolor(AUDIOLABS_WHITE)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor(AUDIOLABS_LIGHT_GRAY)
        pane.set_edgecolor(AUDIOLABS_GRAY)

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
    ax.set_title(title, fontsize=10, color=AUDIOLABS_BLACK)
    ax.set_box_aspect([1, 1, 1])

    # Add axes indicators
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
            color=AUDIOLABS_GRAY,
        )
        ax.add_artist(arrow)
        ax.text(
            vec[0] * 1.05,
            vec[1] * 1.05,
            vec[2] * 1.05,
            label,
            color=AUDIOLABS_BLACK,
            fontsize=12,
        )

    # Use orthographic projection
    ax.set_proj_type("ortho")


if __name__ == "__main__":
    # Create the main Tk window and hide it
    root = tk.Tk()
    root.withdraw()

    params, cmdArgs = cmdArgsToDict()

    # Create progress window
    init_window = InitializationWindow("Initialization Progress")
    init_window.update_progress(
        0, "Starting initialization...", "configure global styles"
    )
    time.sleep(0.5)

    configure_global_styles()

    balloon_plot_with_slider(
        source_dir=os.path.join("examples", "data", "sampled_directivity", "source"),
        receiver_dir=os.path.join(
            "examples", "data", "sampled_directivity", "receiver"
        ),
        sh_order=6,
        initial_freq=500,
        initial_r0_rec=0.6,
        progress_window=init_window,
        params=params,
    )
