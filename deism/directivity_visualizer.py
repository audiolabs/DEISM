# DEISM Directivity Visualizer Components

import numpy as np 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrowPatch
from scipy.io import loadmat
from scipy.special import sph_harm
from scipy.interpolate import interp1d

from deism.core_deism import *
from deism.data_loader import *
import tkinter as tk
from tkinter import filedialog, ttk
import os
import time
from mpl_toolkits.mplot3d import proj3d
from PIL import Image
from netCDF4 import Dataset


class Dir_Visualizer:
    """
    Class for visualizing directivity patterns with interactive controls
    Supports MAT files (source/receiver) and SOFA files (HRTF/HRIR)
    """
    
    # Audiolabs color scheme
    AUDIOLABS_ORANGE = "#F15A24"
    AUDIOLABS_GRAY = "#5B6770"
    AUDIOLABS_LIGHT_GRAY = "#E6E6E6"
    AUDIOLABS_DARK_GRAY = "#333333"
    AUDIOLABS_WHITE = "#FFFFFF"
    AUDIOLABS_BLACK = "#000000"

    @staticmethod
    def sofa_to_internal(sofa_path, ear="L", ref_dirs=None, if_fill_missing_dirs=True):
        """
        Convert a SOFA HRTF/HRIR file to the internal (Psh, Dir_all, freqs, r0) format

        Parameters:
        -----------
        sofa_path : str
            Path to SOFA file
        ear : str, optional
            Ear selection: "L" for left, "R" for right (default: "L")
        ref_dirs : ndarray, optional
            Reference directions for interpolation (J_ref, 2) in radians [az, inc]

        Returns:
        --------
        Psh : ndarray
            Complex pressure field (F, J) - frequency x directions
        Dir_all : ndarray
            Directions in radians (J, 2) - [azimuth, inclination]
        freqs : ndarray
            Frequency axis (Hz)
        r0 : float
            Reference radius (m)
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
        ir = np.array(ds.variables["Data.IR"])  # typical shape (J, R, N), R: # of receivers(ears), N: time samples 
        fs = float(np.array(ds.variables["Data.SamplingRate"]).squeeze())
        ds.close()

        # choose ear (0=left, 1=right)
        ear_idx = 0 if str(ear).upper().startswith("L") else 1
        if ir.ndim != 3 or ir.shape[1] < 2:
            # fallback: if dataset is mono or layout differs, just take channel 0
            ear_idx = 0
        J, _, N = ir.shape if ir.ndim == 3 else (ir.shape[0], 1, ir.shape[-1])
        ir_ear = ir[:, ear_idx, :] if ir.ndim == 3 else ir # impluse response of one ear

        # rFFT per direction → shape (J, F)
        H_dirF = np.fft.rfft(ir_ear, axis=-1)  # (J, F_sofa)
        f_sofa = np.fft.rfftfreq(N, 1 / fs)  # (F_sofa,)

        # 3) arrange to (F, J)
        H_FJ = H_dirF.T.astype(complex)  # (F_sofa, J)

        # --- drop DC (0 Hz), which makes k=0 and h_n^(2)(0) singular ---
        if f_sofa[0] == 0.0:
            H_FJ = H_FJ[1:, :]
            f_sofa = f_sofa[1:]

        freqs = f_sofa
        Psh = H_FJ  # naming aligned

        if ref_dirs is not None and if_fill_missing_dirs:
            # ref_dirs: (J_ref, 2) az/inc from source.mat (uniform sphere)
            Dir_ref = ref_dirs.astype(float)
            az_ref = Dir_ref[:, 0]
            inc_ref = Dir_ref[:, 1]

            az_src = Dir_all[:, 0]
            inc_src = Dir_all[:, 1]

            # convert to 3D unit vectors
            def sph2cart(az, inc):
                x = np.sin(inc) * np.cos(az)
                y = np.sin(inc) * np.sin(az)
                z = np.cos(inc)
                return np.stack([x, y, z], axis=-1)

            vec_src = sph2cart(az_src, inc_src)      # (J_sofa,3)
            vec_ref = sph2cart(az_ref, inc_ref)      # (J_ref,3)

            # great-circle distance cosine
            dot = vec_src @ vec_ref.T                 # (J_sofa, J_ref)
            dot = np.clip(dot, -1.0, 1.0)
            dist = np.arccos(dot)                     # spherical distance (rad)

            # choose K nearest neighbours
            K = 8
            idx = np.argpartition(dist, K, axis=0)[:K, :]  # (K, J_ref)
            dist_K = dist[idx, np.arange(dist.shape[1])]

            # weights = 1 / (d + eps)
            eps = 1e-6
            w = 1.0 / (dist_K + eps)
            w = w / np.sum(w, axis=0, keepdims=True)  # normalize

            # allocate new Psh
            F = Psh.shape[0]
            J_ref = Dir_ref.shape[0]
            Psh_new = np.zeros((F, J_ref), dtype=complex)

            # interpolate each freq
            for fi in range(F):
                P = Psh[fi]  # (J_sofa,)
                Pn = P[idx]  # (K, J_ref)
                # weighted average
                Psh_new[fi] = np.sum(w * Pn, axis=0)

            # replace old
            Psh = Psh_new
            Dir_all = Dir_ref

        return Psh, Dir_all, freqs, r0

    @staticmethod
    def get_directivity_coefs_sofa(k, maxSHorder, Pmnr0, r0):
        """
        Calculate source directivity coefficients C_nm^s or receiver directivity coefficients C_vu^r for SOFA files
        
        Parameters:
        -----------
        k : ndarray
            Wave numbers
        maxSHorder : int
            Maximum spherical harmonics order
        Pmnr0 : ndarray
            Spherical harmonic coefficients
        r0 : float
            Reference radius
        
        Returns:
        --------
        C_nm_s : ndarray
            Directivity coefficients
        """
        C_nm_s = np.zeros([k.size, maxSHorder + 1, 2 * maxSHorder + 1], dtype="complex")
        for n in range(maxSHorder + 1):
            hn_r0_all = sphankel2(n, k * r0)
            for m in range(-n, n + 1):
                # The source directivity coefficients
                C_nm_s[:, n, m] = 1j * ((-1) ** m) / k * Pmnr0[:, n, m + n] / hn_r0_all
        return C_nm_s

    @classmethod
    def get_deism_sh_coeffs(cls, sofa_path, target_freqs, max_order=6, use_reciprocal=True):
        """
        interface for DEISM：
        Convert the SOFA file into a C_nm coefficient matrix that can be used directly by DEISM
        """
        # Load SOFA original datas (Psh, Dir_all, f_sofa, r0)
        # target_freqs is the freq defined by DEISM 
        Psh, Dir_all, f_sofa, r0 = cls.load_directivity(sofa_path, if_fill_missing_dirs=True)
        
        # calculate Cnm
        cache, _ = cls.build_cnm_cache(Psh, Dir_all, f_sofa, r0, max_order, use_reciprocal)
        
        # convert cache (dict) to numpy array (F_sofa, N+1, 2N+1)
        cnm_sofa_array = np.array([cache[(i, max_order, r0)] for i in range(len(f_sofa))])
        
        # interpolate
        f_interp_real = interp1d(f_sofa, cnm_sofa_array.real, axis=0, kind='cubic', fill_value="extrapolate")
        f_interp_imag = interp1d(f_sofa, cnm_sofa_array.imag, axis=0, kind='cubic', fill_value="extrapolate")
        
        cnm_deism = f_interp_real(target_freqs) + 1j * f_interp_imag(target_freqs)

        # confirm the result should be 3-dimensional (F, N+1, 2N+1)
        if cnm_deism.ndim == 2:
            cnm_deism = cnm_deism[np.newaxis, :, :]
        elif cnm_deism.ndim == 4: 
            cnm_deism = np.squeeze(cnm_deism, axis=-1)
        
        return cnm_deism.astype(np.complex64), r0

    @classmethod
    def inject_sofa_into_deism(cls, model, sofa_path, role="receiver", use_reciprocal=False):
        """
        interface: Injects SOFA-directed data into the DEISM model with a single click.        
        Parameters:
            model: DEISM instance
            sofa_path: SOFA file path
            role: "source" or "receiver"
            use_reciprocal: Whether to apply reciprocity
        """
        # Get frequency from the model
        target_freqs = model.params["freqs"]
        
        # Determine the max level based on the role
        sh_order = model.params["sourceOrder"] if role == "source" else model.params["receiverOrder"]
            
        # Get the revised C_nm coefficient matrix
        cnm_sofa, r0_sofa = cls.get_deism_sh_coeffs(
            sofa_path, target_freqs, max_order=sh_order, use_reciprocal=use_reciprocal
        )
        
        # Inject the datas based on the role and vevtorize
        if role == "source":
            model.params["C_nm_s"] = cnm_sofa
            model.params["radiusSource"] = r0_sofa
            model.params["sourceType"] = "SOFA_Imported"
            model.params = vectorize_C_nm_s(model.params)
            print(f"Successfully injected into SOURCE directivity.")
            
        elif role == "receiver":
            model.params["C_vu_r"] = cnm_sofa
            model.params["radiusReceiver"] = r0_sofa
            model.params["receiverType"] = "SOFA_Imported"            
            model.params = vectorize_C_vu_r(model.params)
            print(f"Successfully injected into RECEIVER directivity.")
            
        else:
            raise ValueError("Role must be 'source' or 'receiver'")

    def __init__(self):
        """Initialize the Dir_Visualizer class"""
        self.is_initial_draw = True
        self.is_sofa_file = False
        self.Psh_raw = None
        self.current_file = None
        self.current_file_idx = 0
        self.current_is_receiver = False
        self.S = None
        self.ref_dirs_source = None
        self.params = {}
        
        # Initialize tkinter for file dialogs
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()
        
        # Configure global styles
        self.configure_global_styles()
    
    def configure_global_styles(self):
        """Configure global styles for the visualizer interface"""
        if not hasattr(self, "_styles_configured"):
            style = ttk.Style()
            style.theme_use("clam")

            # Progressbar style
            style.configure(
                "Audiolabs.Horizontal.TProgressbar",
                troughcolor=self.AUDIOLABS_LIGHT_GRAY,
                background=self.AUDIOLABS_ORANGE,
                bordercolor=self.AUDIOLABS_GRAY,
            )

            # label style
            style.configure(
                "TLabel",
                background=self.AUDIOLABS_WHITE,
                foreground=self.AUDIOLABS_BLACK,
                font=("Arial", 10),
            )
            self._styles_configured = True

    def load_audiolabs_logo(self):
        """Load Audiolabs logo while preserving aspect ratio"""
        try:
            logo_path = os.path.join("examples", "audiolabs_logo.png")
            logo_image = Image.open(logo_path)

            # Calculate original ratio of width and height
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
            return Image.new("RGB", (100, 40), color=self.AUDIOLABS_WHITE)

    class InitializationWindow:
        """Window to display progress during initialization"""

        def __init__(self, title="Progress"):
            self.root = tk.Toplevel()
            self.root.title(title)
            self.root.geometry("400x150")
            self.root.configure(bg=Dir_Visualizer.AUDIOLABS_WHITE)

            self.label = ttk.Label(
                self.root,
                text="",
                font=("Arial", 12),
                foreground=Dir_Visualizer.AUDIOLABS_BLACK,
                background=Dir_Visualizer.AUDIOLABS_WHITE,
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
                foreground=Dir_Visualizer.AUDIOLABS_DARK_GRAY,
                background=Dir_Visualizer.AUDIOLABS_WHITE,
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
        """Create 3D arrows for visualization"""

        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs  # Save 3D vertex coordinates

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            return zs[0]  # Returns the z value for depth ordering

    @staticmethod
    def interpolate_Cnm(freqs, Cnm_s_cache, sh_order, r0, target_freq):
        """Interpolate Cnm coefficients to target frequency"""
        freq_array = np.array(freqs)
        Cnm_array = np.array(
            [Cnm_s_cache[(i, sh_order, r0)] for i in range(len(freqs))]
        )  # shape: (n_freq, N+1, 2N+1)

        # Cubic spline interpolation; Cnm is complex, so interpolate real and imag separately
        interp_real = interp1d(freq_array, Cnm_array.real, axis=0, kind="cubic")
        interp_imag = interp1d(freq_array, Cnm_array.imag, axis=0, kind="cubic")

        Cnm_interp = interp_real(target_freq) + 1j * interp_imag(target_freq)
        return Cnm_interp

    @staticmethod
    def is_receiver_file(name: str) -> bool:
        """Check if the data file is a receiver file"""
        return name.endswith("_receiver.mat")

    def _point_src_strength_array(self):
        """Return `pointSrcStrength` as a squeezed numpy array (scalar or 1D)."""
        S_val = self.params.get("pointSrcStrength", 1.0)
        return np.array(S_val).squeeze()

    def _psh_use_for_freq(self, freq_idx: int):
        """
        Return the pressure field used for SH fitting at `freq_idx`.

        - Uses pristine `self.Psh_raw`
        - Applies receiver normalization when enabled (ignored for SOFA)
        """
        if self.Psh_raw is None:
            raise ValueError("Psh_raw is not initialized.")

        if self.is_sofa_file:
            return self.Psh_raw[freq_idx]

        if self.current_is_receiver and self.params.get("ifReceiverNormalize", 0):
            S_arr = self._point_src_strength_array()
            if S_arr.ndim == 0:
                return self.Psh_raw[freq_idx] / S_arr
            return self.Psh_raw[freq_idx] / S_arr[freq_idx]

        return self.Psh_raw[freq_idx]

    def _rebuild_directivity_caches(
        self,
        order: int,
        progress_window=None,
        progress_start: float = 0.0,
        progress_span: float = 100.0,
        message_prefix: str = "Precomputing Pnm & Cnm",
    ):
        """(Re)compute `full_Pnm_cache` and `Cnm_s_cache` for all frequencies."""
        self.full_Pnm_cache.clear()
        self.Cnm_s_cache.clear()

        n_freq = len(self.freqs)
        for freq_idx in range(n_freq):
            start_step = time.perf_counter()
            Psh_use = self._psh_use_for_freq(freq_idx)
            self._compute_and_store_cnm(freq_idx, order, Psh_use)

            if progress_window is not None:
                elapsed_step = time.perf_counter() - start_step
                progress_window.update_progress(
                    progress_start + (freq_idx + 1) / n_freq * progress_span,
                    f"{message_prefix}: {self.freqs[freq_idx]:.1f} Hz...",
                    f"{elapsed_step:.3f} s",
                )

    @staticmethod
    def _list_files(directory, exts):
        return [f for f in os.listdir(directory) if f.lower().endswith(exts)]

    def _discover_allowed_files(self, source_dir, receiver_dir):
        """
        Discover available MAT (source/receiver) and SOFA files and build a basename->path lookup.

        Returns:
            allowed_files: list[str]
            file_lookup: dict[str,str]
            first_name: str
        """
        abs_source_dir = os.path.abspath(source_dir)
        abs_receiver_dir = os.path.abspath(receiver_dir)
        abs_sofa_dir = os.path.abspath(
            os.path.join("examples", "data", "sampled_directivity", "sofa")
        )

        source_files = [
            f
            for f in self._list_files(abs_source_dir, (".mat",))
            if f.endswith("_source.mat")
        ]
        receiver_files = [
            f
            for f in self._list_files(abs_receiver_dir, (".mat",))
            if f.endswith("_receiver.mat")
        ]
        sofa_files = [f for f in self._list_files(abs_sofa_dir, (".sofa",))]

        allowed_files = source_files + receiver_files + sofa_files
        if not allowed_files:
            raise ValueError("No source/receiver/SOFA files found.")

        file_lookup = {}
        for f in source_files:
            file_lookup[f] = os.path.join(abs_source_dir, f)
        for f in receiver_files:
            file_lookup[f] = os.path.join(abs_receiver_dir, f)
        for f in sofa_files:
            file_lookup[f] = os.path.join(abs_sofa_dir, f)

        # Prefer MAT on first load (so we always have a reference grid for later SOFA interpolation)
        first_name = (source_files or receiver_files or sofa_files)[0]
        return allowed_files, file_lookup, first_name

    def balloon_plot_with_slider(
        self,
        source_dir=os.path.join("examples", "data", "sampled_directivity", "source"),
        receiver_dir=os.path.join("examples", "data", "sampled_directivity", "receiver"),
        sh_order=6,
        initial_freq=500,
        initial_r0_rec=0.6,
        params=None,
    ):
        """
        Interactive balloon plot with sliders to change frequency and r0_rec.

        Parameters:
        -----------
        source_dir : str
            Directory containing source .mat files
        receiver_dir : str
            Directory containing receiver .mat files
        sh_order : int
            Spherical harmonics order
        initial_freq : float
            Initial frequency to display (Hz)
        initial_r0_rec: float
            Initial spherical radius during reconstruction
        params : dict
            Parameters dictionary
        """        
        # Create progress window
        progress_window = self.InitializationWindow("Initialization Progress")
        progress_window.update_progress(
            0, "Starting initialization...", "configure global styles"
        )
        time.sleep(0.5)

        # get parameters from cmdArgsToDict 
        if params is None:
            params, cmdArgs = cmdArgsToDict()
        self.params = params

        self.is_initial_draw = True
        self.is_sofa_file = False
        self.Psh_raw = None
        self.initial_r0_rec = initial_r0_rec
        self.initial_sh_order = int(sh_order)

        # Show initialization window
        progress_window.update_progress(25, "Initializing...", "Scanning files")
        time.sleep(0.5)

        # Find all source/receiver MAT files and SOFA files
        allowed_files, file_lookup, first_name = self._discover_allowed_files(
            source_dir, receiver_dir
        )

        # Load first file to initialize data
        self.current_file = file_lookup[first_name]
        progress_window.update_progress(
            35, "Initializing...", f"Loading initial file: {first_name}"
        )
        time.sleep(0.5)

        mat = loadmat(self.current_file)
        Psh = mat["Psh"]
        self.Psh_raw = Psh.copy()  # keep pristine copy
        Dir_all = mat["Dir_all"]
        freqs = mat["freqs_mesh"].squeeze()
        r0 = float(mat["r0"].squeeze())

        # Use the first source file's directions as reference grid for SOFA interpolation
        self.ref_dirs_source = Dir_all.copy()

        max_sh_order = sh_order
        k_all = 2 * np.pi * freqs / 343

        self.current_file_idx = allowed_files.index(first_name)
        self.current_is_receiver = self.is_receiver_file(first_name)
        self.S = params["pointSrcStrength"] if self.current_is_receiver else None
        self.params = params

        # Create figure with Audiolabs styling
        plt.style.use("seaborn-v0_8")  # Start with a clean style    
        self.fig = plt.figure(figsize=(6, 8), facecolor=self.AUDIOLABS_WHITE)  
        plt.subplots_adjust(bottom=0.35, right=0.85)
        
        # status box
        self.status_text = self.fig.text( 
            0.21, 0.02, "", ha="left", va="center", fontsize=9, color=self.AUDIOLABS_ORANGE, 
            bbox=dict(boxstyle="round", fc=self.AUDIOLABS_LIGHT_GRAY, ec=self.AUDIOLABS_ORANGE, alpha=0.8) 
        )
        
        # Create axis for the 3D plot
        self.ax_recon = self.fig.add_subplot(111, projection="3d")
        self.ax_recon.view_init(elev=30, azim=45)  # Set the initial viewing angle
        
        # Add phase legend
        self.add_phase_legend()
        
        # Create control axes with Audiolabs styling ([left, bottom, width, height])
        control_bg_color = self.AUDIOLABS_LIGHT_GRAY
        self.ax_freq = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor=control_bg_color)
        self.ax_freq_left = plt.axes([0.812, 0.182, 0.02, 0.02], facecolor=control_bg_color)
        self.ax_freq_right = plt.axes([0.832, 0.182, 0.02, 0.02], facecolor=control_bg_color)
        self.ax_r0 = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=control_bg_color)
        self.ax_sh = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=control_bg_color)
        self.ax_browse = plt.axes([0.2, 0.25, 0.6, 0.05], facecolor=control_bg_color)
        self.ax_freq_input = plt.axes([0.81, 0.25, 0.085, 0.03])
        
        fmt = lambda f: f"{f:.1f}"
        self.text_box_freq = mpl.widgets.TextBox(self.ax_freq_input, "", initial=fmt(initial_freq))
        self.fig.text(0.895, 0.257, "Hz", fontsize=10, color=self.AUDIOLABS_BLACK)
        
        # Display frequency range
        self.fmm_text = self.fig.text(
            0.81, 0.235,
            f"[{fmt(freqs.min())}, {fmt(freqs.max())}]",
            fontsize=9, color=self.AUDIOLABS_GRAY, ha="left", va="center"
        )

        # Add checkbox for receiver directivity normalization
        self.ax_norm = plt.axes([0.2, 0.04, 0.31, 0.035], facecolor=self.AUDIOLABS_LIGHT_GRAY)
        self.norm_checkbox = mpl.widgets.CheckButtons(
            self.ax_norm, ["Normalize Receiver"], [bool(params.get("ifReceiverNormalize", 0))]
        )  
        
        # Add checkbox for the reciprocal relation 
        self.ax_recip = plt.axes([0.53, 0.04, 0.20, 0.035], facecolor=self.AUDIOLABS_LIGHT_GRAY)
        self.recip_checkbox = mpl.widgets.CheckButtons(
            self.ax_recip, ["Reciprocity"], [bool(params.get("ifReciprocal", 0))]
        )

        # Make sure artists have valid positions
        self.fig.canvas.draw()  # important before reading bbox

        # Move the label text left/right in the checkbox axes (0..1 coordinates)
        LABEL_X = 0.18  
        for txt in self.norm_checkbox.labels:
            txt.set_transform(self.ax_norm.transAxes)
            x_old, y_old = txt.get_position()
            txt.set_position((LABEL_X, y_old))
            txt.set_ha("left")
            txt.set_va("center")
            txt.set_clip_on(False)

        for txt in self.recip_checkbox.labels:
            txt.set_transform(self.ax_recip.transAxes)
            x_old, y_old = txt.get_position()
            txt.set_position((LABEL_X, y_old))
            txt.set_ha("left")
            txt.set_va("center")
            txt.set_clip_on(False)
        
        self.ax_help = plt.axes([0.475, 0.055, 0.005, 0.005])
        self.ax_help2 = plt.axes([0.7, 0.055, 0.005, 0.005])
        self.ax_help.axis("off")
        self.ax_help2.axis("off")
        
        self.ax_help.text(
            0.5,
            0.5,
            "?",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="circle", facecolor=self.AUDIOLABS_DARK_GRAY, alpha=0.8),
            color="white",
        )
        
        self.ax_help2.text(
            0.5,
            0.5,
            "?",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="circle", facecolor=self.AUDIOLABS_DARK_GRAY, alpha=0.8),
            color="white",
        )
        
        self.tooltip = self.ax_help.annotate(
            "Normalize Receiver:\nDivide receiver Psh by point-source strength S(f)\n"
            "Only for FEM-based receiver data; ignored for SOFA.\n"
            "'x' means normalization",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            xytext=(20, 20),
            textcoords="offset points",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round", fc="w", ec=self.AUDIOLABS_GRAY, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=self.AUDIOLABS_GRAY),
            visible=False,
        )
        
        self.tooltip2 = self.ax_help2.annotate(
            "Reciprocity:\n"
            "Toggle using the reciprocal relation\n"
            "for computing the directivity coefficients.\n"
            "Affects all file types.",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            xytext=(20, 20),
            textcoords="offset points",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round", fc="w", ec=self.AUDIOLABS_GRAY, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=self.AUDIOLABS_GRAY),
            visible=False,
        )
        
        # Store data for later use
        self.allowed_files = allowed_files
        self.file_lookup = file_lookup
        self.freqs = freqs
        self.r0 = r0
        self.k_all = k_all
        self.max_sh_order = max_sh_order
        self.Dir_all = Dir_all
        self.Psh = Psh
        
        # Initialize caches
        self.Ynm_cache = {}
        self.hn_cache = {}
        self.full_Pnm_cache = {}  # {(freq_idx, sh_order): full_Pnm}
        self.Cnm_s_cache = {}  # {(freq_idx, sh_order, r0): Cnm_s}
        
        # Resolution for the plots
        res = 50
        # Construct a spherical angle grid (azimuth and elevation) to prepare sampling points for drawing
        az = np.linspace(0, 2 * np.pi, 2 * res + 1)  # 2*res evenly spaced samples
        el = np.linspace(0, np.pi, res)
        self.az_m, self.el_m = np.meshgrid(az, el)  # az_m:(res, 2*res+1), el_m:(res, 2*res+1)
        # Expand the spherical angles into an array of angles (n_pts, 2), with each row containing one [φ, θ]
        self.dirs = np.stack([self.az_m.ravel(), self.el_m.ravel()], axis=1)
        self.ynm_max_order = sh_order

        # Continue with precomputation
        self._continue_initialization(progress_window, sh_order)

    def _continue_initialization(self, progress_window, sh_order):
        """Continue initialization after UI setup"""
        # precomputation of Ynm, hn, Pnm and Cnm
        start_step = time.perf_counter()
        for n in range(sh_order + 1):
            for m in range(-n, n + 1):
                self.Ynm_cache[(n, m)] = sph_harm(m, n, self.dirs[:, 0], self.dirs[:, 1])
        elapsed_step = time.perf_counter() - start_step
        progress_window.update_progress(45, "Precomputing Ynm...", f"{elapsed_step:.3f} s")
        time.sleep(0.5)

        start_step = time.perf_counter()
        kr_values = [
            self.k_all[i] * r for i in range(len(self.freqs)) for r in np.arange(self.r0, 2.0 + 0.001, 0.1)
        ]  # range of rec_r0 : r0 to 2.0, step 0.1
        kr_values = list(set([round(v, 6) for v in kr_values]))
        for n in range(sh_order + 1):
            for kr in kr_values:
                self.hn_cache[(n, kr)] = sphankel2(n, kr)
        elapsed_step = time.perf_counter() - start_step
        progress_window.update_progress(55, "Precomputing hn...", f"{elapsed_step:.3f} s")
        time.sleep(0.5)

        # Precompute caches at `self.max_sh_order` for consistent lookup/slicing later
        self._rebuild_directivity_caches(
            order=self.max_sh_order,
            progress_window=progress_window,
            progress_start=65,
            progress_span=30,
            message_prefix="Precomputing Pnm & Cnm",
        )

        # Close initialization window when done
        progress_window.update_progress(
            100, "Precomputation complete!", "Initialization complete!"
        )
        time.sleep(0.5)
        progress_window.close()

        self.set_status(f"Frequency range: {self.freqs.min():.1f}–{self.freqs.max():.1f} Hz")

        # Setup UI elements
        self._setup_ui_elements()

        # Draw initial plot
        self.update(None)

        plt.show()

    def _compute_and_store_cnm(self, freq_idx, order, Psh_use):
        """
        Compute and store Pnm and Cnm coefficients
        
        Depending on the switch selection, determine whether to apply the reciprocal relation.
        Regardless of file type (SOFA/MAT, source/receiver)
        Compute Pnm and Cnm
        """
        self.full_Pnm_cache[(freq_idx, order)] = SHCs_from_pressure_LS(
            Psh_use.reshape(1, -1), self.Dir_all, order, np.array([self.freqs[freq_idx]])
        )
        k = self.k_all[freq_idx]

        if self.params.get("ifReciprocal", 0):
            self.Cnm_s_cache[(freq_idx, order, self.r0)] = self.get_directivity_coefs_sofa(
                k, order, self.full_Pnm_cache[(freq_idx, order)], self.r0
            )
        else:
            self.Cnm_s_cache[(freq_idx, order, self.r0)] = get_directivity_coefs(
                k, order, self.full_Pnm_cache[(freq_idx, order)], self.r0
            )

    def _ui_on_text_submit(self, text):
        """Callback function for the frequency input box"""
        # Enter a number in the input box and press Enter
        try:
            val = float(text)
            fmin = float(self.freqs.min())
            fmax = float(self.freqs.max())
            if fmin <= val <= fmax:
                self.freq_slider.set_val(val)  # triggers update()
                self.set_status(f"Frequency set to {val:.1f} Hz")
            else:
                self.set_status(
                    f"Out of range: [{fmin:.1f}, {fmax:.1f}] Hz", color="crimson"
                )
                self.text_box_freq.set_val(f"{np.clip(val, fmin, fmax):.1f}")
        except Exception:
            self.set_status("Please input a valid number for frequency.", color="crimson")

    def _ui_toggle_recip(self, label):
        """Toggle reciprocal relation (all file types) and rebuild caches."""
        self.params["ifReciprocal"] = 1 - self.params.get("ifReciprocal", 0)

        win = self.InitializationWindow(
            "Applying reciprocal relation..."
            if self.params["ifReciprocal"]
            else "Removing reciprocal relation..."
        )
        win.update_progress(0, "Rebuilding caches...", "")

        self._rebuild_directivity_caches(
            order=self.max_sh_order,
            progress_window=win,
            progress_start=5,
            progress_span=90,
            message_prefix="Rebuilding caches",
        )

        win.update_progress(100, "Done!", "")
        time.sleep(0.2)
        win.close()
        self.update(None)
        self.set_status(
            "Reciprocal relation: ON"
            if self.params["ifReciprocal"]
            else "Reciprocal relation: OFF"
        )

    def _ui_toggle_normalize(self, label):
        """Toggle receiver normalization (MAT receiver only) and rebuild caches."""

        # Flip flag
        self.params["ifReceiverNormalize"] = 1 - self.params.get("ifReceiverNormalize", 0)

        # Only meaningful for receiver files
        if self.is_sofa_file or not self.current_is_receiver:
            self.fig.canvas.draw_idle()
            return

        win = self.InitializationWindow(
            "Normalizing receiver…"
            if self.params["ifReceiverNormalize"]
            else "Denormalizing receiver…"
        )
        win.update_progress(0, "Starting...", "")

        self._rebuild_directivity_caches(
            order=self.max_sh_order,
            progress_window=win,
            progress_start=5,
            progress_span=90,
            message_prefix="Rebuilding caches",
        )

        win.update_progress(100, "Recomputation Complete!", "")
        time.sleep(0.3)
        win.close()

        self.update(None)
        self.set_status(
            "Receiver normalization."
            if self.params.get("ifReceiverNormalize", 0)
            else "Receiver denormalization."
        )

    def _ui_browse_file(self, event):
        file_path = filedialog.askopenfilename(
            initialdir=os.path.commonpath(
                [
                    os.path.abspath(
                        os.path.join("examples", "data", "sampled_directivity", "source")
                    ),
                    os.path.abspath(
                        os.path.join(
                            "examples", "data", "sampled_directivity", "receiver"
                        )
                    ),
                    os.path.abspath(
                        os.path.join("examples", "data", "sampled_directivity", "sofa")
                    ),
                ]
            ),
            title="Select MAT/SOFA file",
            filetypes=[
                ("MAT/SOFA files", "*.mat *.sofa"),
                ("MAT files", "*.mat"),
                ("SOFA files", "*.sofa"),
            ],
        )
        if not file_path:
            return

        filename = os.path.basename(file_path)
        if filename in self.allowed_files:
            file_idx = self.allowed_files.index(filename)
            self.load_file(file_idx)
        else:
            print(f"Invalid file: {filename}")

    def _ui_on_freq_left(self, event):
        new_val = self.freq_slider.val - 2
        if new_val >= self.freq_slider.valmin:
            self.freq_slider.set_val(new_val)

    def _ui_on_freq_right(self, event):
        new_val = self.freq_slider.val + 2
        if new_val <= self.freq_slider.valmax:
            self.freq_slider.set_val(new_val)

    def _ui_on_hover(self, event):
        if self.ax_browse.contains(event)[0]:
            self.ax_browse.patch.set_facecolor(self.AUDIOLABS_ORANGE)
            self.btn_browse.label.set_text("Browse...")
        else:
            self.ax_browse.patch.set_facecolor(self.AUDIOLABS_LIGHT_GRAY)
            self.btn_browse.label.set_text(
                f"Current: {os.path.basename(self.current_file)}"
            )
        self.ax_browse.figure.canvas.draw_idle()

    def _ui_on_move(self, event):
        self.tooltip.set_visible(event.inaxes is self.ax_help)
        self.tooltip2.set_visible(event.inaxes is self.ax_help2)
        event.canvas.draw_idle()

    def _setup_ui_elements(self):
        """Setup UI elements and event handlers"""
        self.text_box_freq.on_submit(self._ui_on_text_submit)
        self.recip_checkbox.on_clicked(self._ui_toggle_recip)
        self.norm_checkbox.on_clicked(self._ui_toggle_normalize)

        # Create buttons with Audiolabs styling
        self.btn_browse = Button(
            ax=self.ax_browse,
            label=f"Current: {os.path.basename(self.current_file)}",
            color=self.AUDIOLABS_LIGHT_GRAY,
            hovercolor=self.AUDIOLABS_ORANGE,
        )
        self.btn_browse.on_clicked(self._ui_browse_file)

        # Listening for mouse movement events
        self.btn_browse.ax.figure.canvas.mpl_connect("motion_notify_event", self._ui_on_hover)

        self.btn_freq_left = Button(
            self.ax_freq_left, label="<", color=self.AUDIOLABS_GRAY, hovercolor=self.AUDIOLABS_ORANGE
        )
        self.btn_freq_right = Button(
            self.ax_freq_right, label=">", color=self.AUDIOLABS_GRAY, hovercolor=self.AUDIOLABS_ORANGE
        )
        self.btn_freq_left.on_clicked(self._ui_on_freq_left)
        self.btn_freq_right.on_clicked(self._ui_on_freq_right)

        # Create sliders with Audiolabs styling
        slider_params = {
            "facecolor": self.AUDIOLABS_ORANGE,
            "track_color": self.AUDIOLABS_LIGHT_GRAY,
            "edgecolor": self.AUDIOLABS_GRAY,
            "alpha": 0.8,
        }

        self.freq_slider = Slider(
            ax=self.ax_freq,
            label="",
            valmin=self.freqs.min(),
            valmax=self.freqs.max(),
            valinit=500,
            valstep=2,
            **slider_params,
        )
        self.fig.text(
            0.2,
            0.197,
            "Frequency (Hz)",
            ha="left",
            va="top",
            fontsize=10,
            color=self.AUDIOLABS_BLACK,
        )

        self.r0_slider = Slider(
            ax=self.ax_r0,
            label="",
            valmin=self.r0,
            valmax=2.0,
            valinit=max(self.r0, self.initial_r0_rec),
            valstep=0.1,
            **slider_params,
        )
        self.fig.text(
            0.2,
            0.147,
            "Reconstruction Radius (m)",
            ha="left",
            va="top",
            fontsize=10,
            color=self.AUDIOLABS_BLACK,
        )

        self.sh_slider = Slider(
            ax=self.ax_sh,
            label="",
            valmin=0,
            valmax=self.max_sh_order,
            valinit=self.initial_sh_order,
            valstep=1,
            **slider_params,
        )
        self.fig.text(
            0.2,
            0.097,
            "Spherical Harmonics Order",
            ha="left",
            va="top",
            fontsize=10,
            color=self.AUDIOLABS_BLACK,
        )

        # Add Audiolabs logo to the figure
        try:
            logo_img = self.load_audiolabs_logo()
            logo_width, logo_height = logo_img.size
            logo_display_height = 0.06
            logo_display_width = logo_display_height * (logo_width / logo_height)

            logo_ax = self.fig.add_axes([0.02, 0.85, logo_display_width, logo_display_height])
            logo_ax.imshow(logo_img)
            logo_ax.axis("off")
        except Exception as e:
            print(f"Could not load logo: {e}")

        # Connect events
        self.freq_slider.on_changed(self.update)
        self.r0_slider.on_changed(self.update)
        self.sh_slider.on_changed(self.update)

        # Connect mouse movement for tooltips
        self.fig.canvas.mpl_connect("motion_notify_event", self._ui_on_move)

    def load_file(self, file_idx):
        """Load a new file and update the visualization"""
        self.current_file = self.file_lookup[self.allowed_files[file_idx]]
        self.current_file_idx = file_idx
        self.is_sofa_file = os.path.splitext(self.current_file)[1].lower() == ".sofa"
        self.is_initial_draw = True

        # Create progress window
        progress_win = self.InitializationWindow("Loading File Progress")
        progress_win.update_progress(
            0, "Starting...", f"Loading new file: {self.allowed_files[file_idx]}"
        )
        time.sleep(0.5)

        # Clear old cache
        self.full_Pnm_cache.clear()
        self.Cnm_s_cache.clear()

        # load new datas
        ext = os.path.splitext(self.current_file)[1].lower()
        if ext == ".sofa":
            # SOFA branch
            Psh, Dir_all, freqs, r0 = self.sofa_to_internal(
                self.current_file, ear="L", ref_dirs=self.ref_dirs_source, if_fill_missing_dirs=True
            )
            self.Psh_raw = Psh.copy()
            self.freqs = freqs
            self.r0 = r0
            self.k_all = 2 * np.pi * freqs / 343.0
            self.current_is_receiver = True
            self.params["ifReceiverNormalize"] = 0

            self.max_sh_order = 6

            # update cap of sh_slider
            self.sh_slider.valmax = self.max_sh_order
            self.sh_slider.ax.set_xlim(0, self.max_sh_order)
            if self.sh_slider.val > self.max_sh_order:
                self.sh_slider.set_val(self.max_sh_order)

            # recalculate hn function
            self.hn_cache.clear()
            kr_values = [
                (self.k_all[i]) * r
                for i in range(len(self.freqs))
                for r in np.arange(r0, 2.0 + 1e-6, 0.1)
            ]
            kr_values = list(set([round(v, 6) for v in kr_values]))
            for n in range(self.max_sh_order + 1):
                for kr in kr_values:
                    self.hn_cache[(n, kr)] = sphankel2(n, kr)

            if self.max_sh_order > self.ynm_max_order:
                # Only fill in the new order to avoid repeated calculations
                for n in range(self.ynm_max_order + 1, self.max_sh_order + 1):
                    for m in range(-n, n + 1):
                        self.Ynm_cache[(n, m)] = sph_harm(m, n, self.dirs[:, 0], self.dirs[:, 1])
                self.ynm_max_order = self.max_sh_order
        else:
            # MAT branch
            mat = loadmat(self.current_file)
            Psh = mat["Psh"]
            self.Psh_raw = Psh.copy()
            self.Dir_all = mat["Dir_all"]
            self.freqs = mat["freqs_mesh"].squeeze()
            self.r0 = float(mat["r0"].squeeze())
            self.k_all = 2 * np.pi * self.freqs / 343.0
            self.current_is_receiver = self.is_receiver_file(self.allowed_files[file_idx])
            self.S = self.params["pointSrcStrength"] if self.current_is_receiver else None

        # Precompute caches at `self.max_sh_order` (lookup expects max order then slice)
        self._rebuild_directivity_caches(
            order=self.max_sh_order,
            progress_window=progress_win,
            progress_start=35,
            progress_span=60,
            message_prefix="Precomputing Pnm & Cnm",
        )

        progress_win.update_progress(
            95,
            "Updating parameters on the interface...",
            "updating sliders and buttons",
        )
        
        # Update frequency slider
        self.freq_slider.valmin = self.freqs.min()
        self.freq_slider.valmax = self.freqs.max()
        self.freq_slider.ax.set_xlim(self.freqs.min(), self.freqs.max())
        self.fmm_text.set_text(f"[{self.freqs.min():.1f}, {self.freqs.max():.1f}]")
        self.set_status(f"Frequency range: {self.freqs.min():.1f}–{self.freqs.max():.1f} Hz")
        # Set to middle frequency
        self.freq_slider.set_val(self.freqs[len(self.freqs) // 2])
        # Update r0 slider
        self.r0_slider.set_val(max(self.r0, self.initial_r0_rec))
        # Update button label
        self.btn_browse.label.set_text(f"Current: {os.path.basename(self.current_file)}")
        time.sleep(0.5)

        progress_win.update_progress(100, "Ready!", "File loading complete!")
        time.sleep(0.5)
        progress_win.close()

        self.update(None)

    def update(self, val):
        """Update the visualization based on current slider values"""
        if self.current_file is None:
            return

        # get current freq, r0 and sh_order
        freq = self.freq_slider.val
        r0_rec = self.r0_slider.val
        current_sh_order = int(self.sh_slider.val)

        freq_idx = np.argmin(np.abs(self.freqs - freq))

        # Clear previous plot
        self.ax_recon.cla()

        if self.is_initial_draw:
            # get Cnm_s from Cnm_s_cache
            Cnm_s = self.Cnm_s_cache[(freq_idx, self.max_sh_order, self.r0)][
                :, : current_sh_order + 1, :
            ]
        else:
            # get Cnm_s with interpolation, (1, sh_order + 1, 2 * sh_order + 1)
            Cnm_s = self.interpolate_Cnm(self.freqs, self.Cnm_s_cache, self.max_sh_order, self.r0, freq)
            Cnm_s = Cnm_s[:, : current_sh_order + 1, :]

        # Reconstruct with current r0_rec
        # Recalculate the new Pnm using r0_rec according to formula: Pnm = Cnm_s * hn_r0
        Pnm_rec = np.zeros(
            (1, current_sh_order + 1, 2 * current_sh_order + 1), dtype=complex
        )
        k = self.k_all[freq_idx]

        with np.errstate(invalid="raise", divide="raise", over="raise"):
            try:
                for n in range(current_sh_order + 1):
                    _key = (int(n), round(float(k * r0_rec), 6))
                    hn_r0_rec = self.hn_cache.get(_key)
                    if hn_r0_rec is None:
                        hn_r0_rec = sphankel2(int(n), float(_key[1]))
                        self.hn_cache[_key] = hn_r0_rec
                    for m in range(-n, n + 1):
                        Pnm_rec[:, n, m + n] = Cnm_s[:, n, m] * hn_r0_rec
            except FloatingPointError:
                self.set_status("Numerical instability at current (frequency, radius, order). Try lower order or larger radius.", color="crimson")
                self.ax_recon.cla()
                self.fig.canvas.draw_idle()
                return

        # Plot reconstructed
        self.plot_balloon_rec(
            current_sh_order,
            Pnm_rec[0],
            f"Reconstructed balloon plot"
        )

        # Request to refresh the images when the GUI is idle to improve interaction efficiency
        self.fig.canvas.draw_idle()
        # Synchronize text box value
        self.text_box_freq.set_val(f"{freq:.1f}")

        self.is_initial_draw = False

    def set_status(self, msg, color=None):
        """Display a status message at the bottom of the interface"""
        if color is None:
            color = self.AUDIOLABS_ORANGE
        self.status_text.set_text(msg)
        self.status_text.set_color(color)
        self.fig.canvas.draw_idle()

    def add_phase_legend(self):
        """Add phase colorbar legend to the figure"""
        # Create a colormap object representing the HSV hue mapping
        cmap = plt.get_cmap("twilight")

        # Create a new axis to display the color bar
        cbar_ax = self.fig.add_axes([0.85, 0.3, 0.02, 0.6])
        norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb = mpl.colorbar.ColorbarBase(
            cbar_ax, cmap=cmap, norm=norm, orientation="vertical"
        )

        cb.set_label("Phase (radians)", color=self.AUDIOLABS_BLACK)
        cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

        # Set colorbar styling
        cbar_ax.tick_params(colors=self.AUDIOLABS_BLACK)
        cbar_ax.yaxis.label.set_color(self.AUDIOLABS_BLACK)
        cbar_ax.spines["top"].set_color(self.AUDIOLABS_GRAY)
        cbar_ax.spines["bottom"].set_color(self.AUDIOLABS_GRAY)
        cbar_ax.spines["left"].set_color(self.AUDIOLABS_GRAY)
        cbar_ax.spines["right"].set_color(self.AUDIOLABS_GRAY)

    def plot_balloon_rec(self, order, Pnm, title):
        """Plot a reconstructed balloon plot with Audiolabs styling"""
        # Initialize the direction response vector D
        D = np.zeros(self.dirs.shape[0], dtype=complex)

        # Iterate over each degree and order n, m; compute Ynm spherical harmonics function; Weight it using the SHCs; Obtain the directional response D (sound field) for all directions
        for n in range(order + 1):
            for m in range(-n, n + 1):
                Ynm = self.Ynm_cache[(n, m)]
                D += Pnm[n, m + n] * Ynm  # size = (res)*(2*res+1)

        # Magnitude
        D_abs = np.abs(D)
        # (elevation, azimuth), D_abs is one-dimensional, reshape to 2D (res, 2*res+1)
        D_plot = D_abs.reshape(self.el_m.shape)
        # Phase (in radians, unwrapped), the range of np.angle(D) is [-π, π]
        D_phase = np.angle(D)
        D_phase_plot = D_phase.reshape(self.el_m.shape)

        # Amplitude and phase normalized to [0,1]
        maxval = D_plot.max()
        if maxval < 1e-12:  # zero guard to avoid 0/0 → NaN
            maxval = 1.0
        norm_amp = D_plot / maxval
        norm_phase = (D_phase_plot + np.pi) / (2 * np.pi)

        # Spherical coordinates -> Cartesian coordinates
        x = norm_amp * np.sin(self.el_m) * np.cos(self.az_m)
        y = norm_amp * np.sin(self.el_m) * np.sin(self.az_m)
        z = norm_amp * np.cos(self.el_m)

        # color mapping
        cmap = plt.get_cmap("twilight")
        colors = cmap(norm_phase)

        # Plot; Set the tick range and labels
        lim = 1
        self.ax_recon.set_xlim([-lim, lim])
        self.ax_recon.set_ylim([-lim, lim])
        self.ax_recon.set_zlim([-lim, lim])
        ticks = np.linspace(-lim, lim, 3)
        self.ax_recon.set_xticks(ticks)
        self.ax_recon.set_yticks(ticks)
        self.ax_recon.set_zticks(ticks)

        self.ax_recon.set_xlabel("X", color=self.AUDIOLABS_BLACK)
        self.ax_recon.set_ylabel("Y", color=self.AUDIOLABS_BLACK)
        self.ax_recon.set_zlabel("Z", color=self.AUDIOLABS_BLACK)

        self.ax_recon.set_facecolor(self.AUDIOLABS_WHITE)
        for pane in [self.ax_recon.xaxis.pane, self.ax_recon.yaxis.pane, self.ax_recon.zaxis.pane]:
            pane.set_facecolor(self.AUDIOLABS_LIGHT_GRAY)
            pane.set_edgecolor(self.AUDIOLABS_GRAY)

        self.ax_recon.plot_surface(
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
        self.ax_recon.set_title(title, fontsize=10, color=self.AUDIOLABS_BLACK)
        self.ax_recon.set_box_aspect([1, 1, 1])

        # Add axes indicators
        arrow_len = 1.2
        for vec, label in zip(
            [(arrow_len, 0, 0), (0, arrow_len, 0), (0, 0, arrow_len)], ["x", "y", "z"]
        ):
            arrow = self.Arrow3D(
                [0, vec[0]],
                [0, vec[1]],
                [0, vec[2]],
                mutation_scale=20,
                lw=2,
                arrowstyle="-|>",
                color=self.AUDIOLABS_GRAY,
            )
            self.ax_recon.add_artist(arrow)
            self.ax_recon.text(
                vec[0] * 1.05,
                vec[1] * 1.05,
                vec[2] * 1.05,
                label,
                color=self.AUDIOLABS_BLACK,
                fontsize=12,
            )

        # Use orthographic projection
        self.ax_recon.set_proj_type("ortho")

    # ==============================================================================
    # Comparison & Analysis Methods (Integrated from comparison_analysis.py)
    # ==============================================================================

    @staticmethod
    def mat_to_internal(mat_path):
        """Convert *_source.mat / *_receiver.mat to (Psh, Dir_all, freqs, r0)"""
        mat = loadmat(mat_path)
        Psh = mat["Psh"]
        Dir_all = mat["Dir_all"]
        freqs = mat["freqs_mesh"].squeeze()
        r0 = float(mat["r0"].squeeze())
        return Psh, Dir_all, freqs, r0

    @classmethod
    def load_directivity(cls, path, ear="L", if_fill_missing_dirs=True):
        """Unified interface for .sofa or .mat"""
        if path.lower().endswith(".sofa"):
            initial_file = os.path.join(
                "examples",
                "data",
                "sampled_directivity",
                "source",
                "Speaker_cuboid_cyldriver_source.mat",
            )
            mat = loadmat(initial_file)
            Dir_all = mat["Dir_all"]
            ref_dirs_source = Dir_all.copy()

            Psh, Dir_all, freqs, r0 = cls.sofa_to_internal(
                path, ear=ear, ref_dirs=ref_dirs_source, if_fill_missing_dirs=if_fill_missing_dirs
            )
            Psh = np.asarray(Psh)
            if Psh.shape[0] != len(freqs):  # In case shape is (J, F) instead of (F, J)
                Psh = Psh.T
        elif path.lower().endswith(".mat"):
            Psh, Dir_all, freqs, r0 = cls.mat_to_internal(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        return Psh, Dir_all, freqs, r0

    @classmethod
    def build_cnm_cache(cls, Psh, Dir_all, freqs, r0, max_order, use_reciprocal):
        """For the current file, compute a Cnm_s_cache copy over all frequencies"""
        k_all = 2 * np.pi * freqs / 343.0  # c = 343 m/s
        cache = {}

        for fi, k in enumerate(k_all):
            Psh_use = Psh[fi]

            Pnm = SHCs_from_pressure_LS(
                Psh_use.reshape(1, -1),
                Dir_all,
                max_order,
                np.array([freqs[fi]]),
            )

            if use_reciprocal:
                Cnm = cls.get_directivity_coefs_sofa(k, max_order, Pnm, r0)
            else:
                Cnm = get_directivity_coefs(k, max_order, Pnm, r0)

            cache[(fi, max_order, r0)] = Cnm

        return cache, k_all

    @staticmethod
    def reconstruct_pressure_field(Cnm_cache, k_all, dirs, r0, r0_rec, max_order):
        """Reconstruct 3D pressure field on discrete directions from Cnm_cache"""
        n_freq = len(k_all)
        n_dir = dirs.shape[0]
        P_field = np.zeros((n_freq, n_dir), dtype=complex)

        az = dirs[:, 0]
        inc = dirs[:, 1]
        Ynm_cache = {}

        for n in range(max_order + 1):
            for m in range(-n, n + 1):
                Ynm_cache[(n, m)] = sph_harm(m, n, az, inc)

        for fi, k in enumerate(k_all):
            Cnm_s = Cnm_cache[(fi, max_order, r0)]

            Pnm_rec = np.zeros((1, max_order + 1, 2 * max_order + 1), dtype=complex)
            for n in range(max_order + 1):
                kr = k * r0_rec
                hn_r0_rec = sphankel2(int(n), float(kr))
                for m in range(-n, n + 1):
                    Pnm_rec[:, n, m + n] = Cnm_s[:, n, m] * hn_r0_rec

            D = np.zeros(n_dir, dtype=complex)
            for n in range(max_order + 1):
                for m in range(-n, n + 1):
                    Ynm = Ynm_cache[(n, m)]
                    D += Pnm_rec[0, n, m + n] * Ynm

            P_field[fi, :] = D

        return P_field

    @classmethod
    def build_pressure_field_with_reciprocity(cls, path, ear="L", max_order=6, r0_rec=None, if_fill_missing_dirs=True):
        """Load directivity, build cache with reciprocity, and reconstruct 3D field"""
        print(f"Loading file: {path}")
        Psh, Dir_all, freqs, r0 = cls.load_directivity(path, ear=ear, if_fill_missing_dirs=if_fill_missing_dirs)
        if r0_rec is None:
            r0_rec = r0

        print("  Building Cnm cache with reciprocity ...")
        cnm_rec, k_all = cls.build_cnm_cache(
            Psh, Dir_all, freqs, r0, max_order, use_reciprocal=True
        )

        print("  Reconstructing pressure field ...")
        P_field = cls.reconstruct_pressure_field(cnm_rec, k_all, Dir_all, r0, r0_rec, max_order)

        return P_field, Dir_all, freqs

    @staticmethod
    def load_olhead_eq_response(eq_sofa_path, ear, freqs_target):
        """Read single-ear EQ IR, compute H_eq(f), and interpolate onto freqs_target"""
        ds = Dataset(eq_sofa_path, "r")
        ir_all = ds.variables["Data.IR"][:]
        shape = ir_all.shape

        ear_idx = 0 if ear.upper() == "L" else 1

        if len(shape) == 3:
            ir = ir_all[0, ear_idx, :]
        elif len(shape) == 2:
            ir = ir_all[ear_idx, :]
        else:
            ds.close()
            raise RuntimeError(f"Unexpected Data.IR shape {shape} in {eq_sofa_path}")

        if "Data.SamplingRate" in ds.variables:
            fs = float(ds.variables["Data.SamplingRate"][:].squeeze())
        else:
            fs = float(ds.getncattr("Data.SamplingRate"))

        ds.close()

        ir = np.asarray(ir, dtype=float)
        N = len(ir)
        N_fft = 2 ** int(np.ceil(np.log2(N * 2)))
        H_full = np.fft.rfft(ir, n=N_fft)
        freqs_full = np.fft.rfftfreq(N_fft, d=1.0 / fs)

        H_real = np.interp(freqs_target, freqs_full, H_full.real)
        H_imag = np.interp(freqs_target, freqs_full, H_full.imag)
        H_eq = H_real + 1j * H_imag

        return H_eq

    @classmethod
    def build_pressure_field_olhead(cls, case, hrir_path, ff_eq_path, diff_eq_path, ear="L", max_order=6, r0_rec=None, if_fill_missing_dirs=True):
        """Build pressure field with optional Olhead EQ"""
        Psh_raw, Dir_all, freqs, r0 = cls.load_directivity(hrir_path, ear=ear, if_fill_missing_dirs=if_fill_missing_dirs)
        if r0_rec is None:
            r0_rec = r0

        if case.lower() == "raw":
            Psh_use = Psh_raw
        elif case.lower() == "free":
            H_eq = cls.load_olhead_eq_response(ff_eq_path, ear, freqs)
            Psh_use = Psh_raw * H_eq[:, np.newaxis]
        elif case.lower() == "diff":
            H_eq = cls.load_olhead_eq_response(diff_eq_path, ear, freqs)
            Psh_use = Psh_raw * H_eq[:, np.newaxis]
        else:
            raise ValueError(f"Unknown case '{case}'")

        print(f"  Building Cnm cache with reciprocity for case '{case}' ...")
        cnm_cache, k_all = cls.build_cnm_cache(
            Psh_use, Dir_all, freqs, r0, max_order, use_reciprocal=True
        )

        print("  Reconstructing pressure field ...")
        P_field = cls.reconstruct_pressure_field(
            cnm_cache, k_all, Dir_all, r0, r0_rec, max_order
        )

        return P_field, Dir_all, freqs

    @staticmethod
    def compute_field_differences(P_ref, P_cmp):
        """Compute magnitude and phase differences between two fields"""
        mag_ref = np.abs(P_ref)
        mag_cmp = np.abs(P_cmp)
        dmag = np.abs(mag_cmp - mag_ref)

        phase_ref = np.angle(P_ref)
        phase_cmp = np.angle(P_cmp)
        dphase = phase_cmp - phase_ref
        dphase_abs = dphase

        mean_dmag = dmag.mean(axis=1)
        mean_dphase = dphase_abs.mean(axis=1)

        return dmag, dphase_abs, mean_dmag, mean_dphase

    @staticmethod
    def plot_differences(freqs, Dir_all, dmag, dphase_abs, mean_dmag, mean_dphase, title_prefix=""):
        """Plot frequency vs mean differences and 2D scatter mollweide plots"""
        fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
        ax1.plot(freqs, mean_dmag)
        ax1.set_ylabel("Mean |ΔP|")
        ax1.grid(True, alpha=0.3)

        ax2.plot(freqs, mean_dphase)
        ax2.set_ylabel("Mean |Δphase| [rad]")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.grid(True, alpha=0.3)

        fig1.suptitle(f"{title_prefix} (mean differences)")

        az = Dir_all[:, 0]
        inc = Dir_all[:, 1]
        lat = np.pi / 2 - inc
        lon = az - np.pi

        example_freqs = [2000, 4000, 8000, 12000]
        idxs = [int(np.argmin(np.abs(freqs - f))) for f in example_freqs]

        n_cols = len(idxs)
        fig2 = plt.figure(figsize=(3.4 * n_cols, 6.5), constrained_layout=True)
        gs = fig2.add_gridspec(2, n_cols, hspace=0.015, wspace=0.15)
        fig2.suptitle(f"{title_prefix} (selected frequencies)")

        lon_ticks_deg = np.arange(-135, 136, 45)
        lat_ticks_deg = np.arange(-60, 61, 30)

        def _style_mollweide(ax):
            ax.set_xticks(np.deg2rad(lon_ticks_deg))
            ax.set_xticklabels([f"{deg:d}°" for deg in lon_ticks_deg], fontsize=9)
            ax.set_yticks(np.deg2rad(lat_ticks_deg))
            ax.set_yticklabels([f"{deg:d}°" for deg in lat_ticks_deg], fontsize=9)
            ax.tick_params(axis="both", pad=6, colors="black", labelcolor="black")
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
            ax.set_xlabel("")
            ax.set_ylabel("")

        for col, (fi, f_target) in enumerate(zip(idxs, example_freqs)):
            ax_mag = fig2.add_subplot(gs[0, col], projection="mollweide")
            sc1 = ax_mag.scatter(lon, lat, c=dmag[fi, :], s=15, cmap="plasma")
            ax_mag.set_title(f"{f_target:.0f} Hz |ΔP|")
            _style_mollweide(ax_mag)
            fig2.colorbar(sc1, ax=ax_mag, orientation="horizontal", pad=0.08, fraction=0.05, aspect=30)

            ax_ph = fig2.add_subplot(gs[1, col], projection="mollweide")
            sc2 = ax_ph.scatter(lon, lat, c=dphase_abs[fi, :], s=15, vmin=0, vmax=np.pi, cmap="plasma")
            ax_ph.set_title(f"{f_target:.0f} Hz |Δphase|")
            _style_mollweide(ax_ph)
            fig2.colorbar(sc2, ax=ax_ph, orientation="horizontal", pad=0.08, fraction=0.05, aspect=30)

        fig1.tight_layout()

    @classmethod
    def analyze_reciprocity(cls, path, ear="L", max_order=6, r0_rec=None, if_fill_missing_dirs=True):
        """Compare reciprocity on/off for a given file"""
        print(f"Loading file: {path}")
        Psh, Dir_all, freqs, r0 = cls.load_directivity(path, ear=ear, if_fill_missing_dirs=if_fill_missing_dirs)
        if r0_rec is None:
            r0_rec = r0

        print("Building Cnm caches ...")
        cnm_off, k_all = cls.build_cnm_cache(Psh, Dir_all, freqs, r0, max_order, use_reciprocal=False)
        cnm_on, _ = cls.build_cnm_cache(Psh, Dir_all, freqs, r0, max_order, use_reciprocal=True)

        print("Reconstructing pressure fields ...")
        P_off = cls.reconstruct_pressure_field(cnm_off, k_all, Dir_all, r0, r0_rec, max_order)
        P_on = cls.reconstruct_pressure_field(cnm_on, k_all, Dir_all, r0, r0_rec, max_order)

        dmag, dphase_abs, mean_dmag, mean_dphase = cls.compute_field_differences(P_off, P_on)
        cls.plot_differences(freqs, Dir_all, dmag, dphase_abs, mean_dmag, mean_dphase, title_prefix="reciprocity on vs off")

    @classmethod
    def compare_two_files(cls, path_ref, path_cmp, label_ref, label_cmp, ear="L", max_order=6, r0_rec=None, if_fill_missing_dirs=True):
        """Compare two files under reciprocity-on condition"""
        print("=" * 80)
        print(f"Reference: {label_ref}")
        P_ref, Dir_ref, freqs_ref = cls.build_pressure_field_with_reciprocity(
            path_ref, ear=ear, max_order=max_order, r0_rec=r0_rec, if_fill_missing_dirs=if_fill_missing_dirs
        )

        print(f"Compared : {label_cmp}")
        P_cmp, Dir_cmp, freqs_cmp = cls.build_pressure_field_with_reciprocity(
            path_cmp, ear=ear, max_order=max_order, r0_rec=r0_rec, if_fill_missing_dirs=if_fill_missing_dirs
        )
        # 1) Direction grids must be identical (otherwise the difference is not interpretable)
        if Dir_ref.shape != Dir_cmp.shape or not np.allclose(Dir_ref, Dir_cmp):
            raise ValueError("Direction grids of the two files do not match!")
        # 2) Align frequency axis: only compare on common frequency bins
        freqs_ref = np.asarray(freqs_ref).ravel()
        freqs_cmp = np.asarray(freqs_cmp).ravel()

        common_freqs = np.intersect1d(freqs_ref, freqs_cmp)
        if common_freqs.size == 0:
            raise ValueError("No overlapping frequencies between the two files!")

        idx_ref = np.nonzero(np.isin(freqs_ref, common_freqs))[0]
        idx_cmp = np.nonzero(np.isin(freqs_cmp, common_freqs))[0]
        # Aligned frequency axis and pressure fields
        freqs_use = freqs_ref[idx_ref]
        P_ref_use = P_ref[idx_ref, :]
        P_cmp_use = P_cmp[idx_cmp, :]

        print(f"  Using {len(freqs_use)} common frequency bins from {freqs_use[0]:.1f} Hz to {freqs_use[-1]:.1f} Hz")

        dmag, dphase_abs, mean_dmag, mean_dphase = cls.compute_field_differences(P_ref_use, P_cmp_use)
        title = f"{label_cmp} vs. {label_ref}"
        cls.plot_differences(freqs_use, Dir_ref, dmag, dphase_abs, mean_dmag, mean_dphase, title_prefix=title)

    @classmethod
    def compare_olhead_eq(cls, hrir_path, ff_eq_path, diff_eq_path, ear="L", max_order=6, r0_rec=None, if_fill_missing_dirs=True):
        """Compare Raw, Free-field, and Diffuse-field cases for Olhead"""
        print("=" * 80)
        print("Case: RAW")
        P_raw, Dir_all, freqs = cls.build_pressure_field_olhead("raw", hrir_path, ff_eq_path, diff_eq_path, ear=ear, max_order=max_order, r0_rec=r0_rec, if_fill_missing_dirs=if_fill_missing_dirs)

        print("=" * 80)
        print("Case: FREE-FIELD")
        P_free, Dir2, freqs2 = cls.build_pressure_field_olhead("free", hrir_path, ff_eq_path, diff_eq_path, ear=ear, max_order=max_order, r0_rec=r0_rec, if_fill_missing_dirs=if_fill_missing_dirs)

        print("=" * 80)
        print("Case: DIFFUSE-FIELD")
        P_diff, Dir3, freqs3 = cls.build_pressure_field_olhead("diff", hrir_path, ff_eq_path, diff_eq_path, ear=ear, max_order=max_order, r0_rec=r0_rec, if_fill_missing_dirs=if_fill_missing_dirs)

        assert np.allclose(Dir_all, Dir2) and np.allclose(Dir_all, Dir3)
        assert np.allclose(freqs, freqs2) and np.allclose(freqs, freqs3)

        dmag_rf, dph_rf, mean_mag_rf, mean_ph_rf = cls.compute_field_differences(P_raw, P_free)
        cls.plot_differences(freqs, Dir_all, dmag_rf, dph_rf, mean_mag_rf, mean_ph_rf, title_prefix="Free-field vs Raw")

        dmag_rd, dph_rd, mean_mag_rd, mean_ph_rd = cls.compute_field_differences(P_raw, P_diff)
        cls.plot_differences(freqs, Dir_all, dmag_rd, dph_rd, mean_mag_rd, mean_ph_rd, title_prefix="Diffuse-field vs Raw")

    @classmethod
    def experiments(cls, exp=None, filenames=None, if_fill_missing_dirs=True):

        base = os.path.join("examples", "data", "sampled_directivity", "sofa")
        paths = [os.path.join(base, f) for f in filenames]

        if exp == "if_reciprocity":
            """
            Experiment 1:
            - any SOFA file
            - Analyze reciprocity on/off
            """
            cls.analyze_reciprocity(paths[0], ear="L", max_order=6, r0_rec=None, if_fill_missing_dirs=if_fill_missing_dirs)
            plt.show()
            
        elif exp == "compare_2files":
            """
            Experiment 2:
            - Compare 2 SOFA files
            e.g.- SONICOM Raw vs Free-field
                - SONICOM Raw vs Free-field MinPhase
            """
            cls.compare_two_files(paths[0], paths[1], label_ref="Raw", label_cmp="Free-field", ear="L", max_order=6, r0_rec=None, if_fill_missing_dirs=if_fill_missing_dirs)
            plt.show()

        elif exp == "compare_olhead":
            """
            Experiment 3:
            - OlHeaD BuK-ED:
                Raw vs Free-field & Diffuse-field
            """
            cls.compare_olhead_eq(paths[0], paths[1], paths[2], ear="L", max_order=6, r0_rec=None, if_fill_missing_dirs=if_fill_missing_dirs)
            plt.show()

        