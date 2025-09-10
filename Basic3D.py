import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle, RegularPolygon, Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # For custom 3D polygons

# (All original scientific and helper functions like generate_luts_for_method,
# create_inverse_interpolators, VScrollFrame, etc., remain unchanged)

# ----------------------------- Materials / Resins -----------------------------
MATERIALS = {
    "Si":   {"n": 3.43, "lambda_range_um": (1.2, 14.0)},
    "SiN":  {"n": 2.00, "lambda_range_um": (0.4, 5.0)},
    "CaF2": {"n": 1.40, "lambda_range_um": (0.2, 12.0)},
    "SOI":  {"n": 3.43, "lambda_range_um": (1.2, 8.0)},
    "ZnSe": {"n": 2.40, "lambda_range_um": (0.6, 16.0)},
}
RESINS = {
    "IP-Dip + TiO2":  {"n_eff": 1.80},
    "IP-Dip + SiO2":  {"n_eff": 1.55},
    "ORMOCER + ZrO2": {"n_eff": 1.68},
    "NOA + TiO2":     {"n_eff": 1.70},
}

# ----------------------------- Manufacturing methods -----------------------------
METHODS = {
    "Single-photon AM": {
        "compatible_resins": {"NOA + TiO2", "ORMOCER + ZrO2"},
        "d_frac": (0.30, 0.80),  # 30–80% of pitch
        "quant_um": 0.10,
        "h2pi_scale": 0.85,
    },
    "fs two-photon AM": {
        "compatible_resins": {"IP-Dip + TiO2", "IP-Dip + SiO2", "ORMOCER + ZrO2"},
        "d_frac": (0.20, 0.90),  # 20–90% of pitch
        "quant_um": 0.05,
        "h2pi_scale": 1.00,
    },
}

# ----------------------------- LUT scaled to pitch + method -----------------------------
def generate_luts_for_method(substrate, resin, wavelength_um, pitch_um, method):
    neff = RESINS[resin]["n_eff"]
    m = METHODS[method]
    d_lo, d_hi = m["d_frac"]
    q = m["quant_um"]

    d_um = np.linspace(d_lo * pitch_um, d_hi * pitch_um, 120)
    if q and q > 0:
        d_um = np.round(d_um / q) * q
        d_um = np.unique(d_um)
    diameters_nm = d_um * 1000.0

    # Smooth phase curve; span mildly increases with n_eff
    amp = 1.5 + 0.4 * (neff - 1.6)
    phases = np.mod(amp * np.pi * np.sin(np.pi * (d_um - d_lo * pitch_um) / ((d_hi - d_lo) * pitch_um)), 2 * np.pi)

    fab_param = 1000 + 120 * (d_um - d_lo * pitch_um)
    fab_name = "Fabrication Proxy (a.u.)"
    return {
        "phase_lut": {"diameters_nm": diameters_nm, "phases_rad": phases},
        "fab_lut":   {"diameters_nm": diameters_nm, "param_values": fab_param, "param_name": fab_name},
    }

def create_inverse_interpolators(phase_lut, fab_lut):
    # phase -> diameter (periodic)
    ph = np.mod(phase_lut["phases_rad"], 2 * np.pi)
    dia = phase_lut["diameters_nm"]
    order = np.argsort(ph)
    ph_u, idx = np.unique(ph[order], return_index=True)
    dia_u = dia[order][idx]
    ph_aug = np.concatenate([ph_u, ph_u[:1] + 2 * np.pi])
    dia_aug = np.concatenate([dia_u, dia_u[:1]])
    phase_to_dia = interp1d(ph_aug, dia_aug, bounds_error=False, fill_value="extrapolate")

    # diameter -> phase  (for realized phase)
    dia_to_phase = interp1d(dia, phase_lut["phases_rad"], bounds_error=False,
                            fill_value=(phase_lut["phases_rad"][0], phase_lut["phases_rad"][-1]))

    # diameter -> fabrication proxy
    dia_to_fab = interp1d(
        fab_lut["diameters_nm"], fab_lut["param_values"],
        bounds_error=False, fill_value=(fab_lut["param_values"][0], fab_lut["param_values"][-1])
    )
    return phase_to_dia, dia_to_phase, dia_to_fab

# ----------------------------- Helpers -----------------------------
class VScrollFrame(ttk.Frame):
    """A vertical scrollable frame that plays nicely with Matplotlib."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._wheelbound = False

        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.body.bind("<Enter>", self._bind_mousewheel)
        self.body.bind("<Leave>", self._unbind_mousewheel)

    def _on_body_configure(self, _evt):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, evt):
        self.canvas.itemconfigure(self._win, width=evt.width)

    def _on_wheel(self, evt):
        self.canvas.yview_scroll(int(-evt.delta/120), "units")

    def _on_wheel_up(self, _evt):   self.canvas.yview_scroll(-1, "units")
    def _on_wheel_down(self, _evt): self.canvas.yview_scroll(+1, "units")

    def _bind_mousewheel(self, _evt):
        if self._wheelbound: return
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel_up)
        self.canvas.bind_all("<Button-5>", self._on_wheel_down)
        self._wheelbound = True

    def _unbind_mousewheel(self, _evt):
        if not self._wheelbound: return
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
        self._wheelbound = False

    def scroll_to_top(self):
        self.canvas.yview_moveto(0.0)

def choice_map(pairs): return {a: b for a, b in pairs}

# --- Constants ---
WL_CHOICES   = [("λ = 9 μm", 9.0), ("λ = 10 μm", 10.0)]
DIAM_CHOICES = [("50 µm", 0.05), ("75 µm", 0.075), ("100 µm", 0.10),
                ("1.5 mm", 1.5), ("10 mm", 10.0), ("15 mm", 15.0), ("20 mm", 20.0), ("25 mm", 25.0)]
PITCH_CHOICES= [("5 µm", 5.0), ("7 µm", 7.0), ("9 µm", 9.0), ("12 µm", 12.0)]
WD_CHOICES   = [("5 mm", 5.0), ("25 mm", 25.0), ("50 mm", 50.0), ("100 mm", 100.0), ("200 mm", 200.0)]
METHOD_LIST  = list(METHODS.keys())

LAYOUT_KEYS = [
    "wavelength_um","substrate","resin",
    "shape","sides",
    "lens_diameter","pitch_um","working_distance","perf_pct",
    "method"
]

# ----------------------------- App -----------------------------
class PDK_App:
    def __init__(self, root):
        self.root = root
        self.root.title("Metalens PDK")

        self.REALIZED_UPSAMPLE = 3

        # -------- Defaults (Frustum) --------
        self.config = {
            "wavelength_um": WL_CHOICES[0][1],
            "substrate": list(MATERIALS.keys())[0],
            "resin": list(RESINS.keys())[0],
            "shape": "frustum",
            "sides": 10,
            "lens_diameter": DIAM_CHOICES[3][1],
            "pitch_um": PITCH_CHOICES[2][1],
            "working_distance": WD_CHOICES[1][1],
            "perf_pct": 80.0,
            "method": METHOD_LIST[1],
        }
        self.current_layout_data = {}
        self._layout_key = None
        self.right_panel_widget = None # Holds the current right-side widget (e.g., canvas)

        # -------- Frames --------
        left_sf = VScrollFrame(root, width=500, bootstyle=SECONDARY)
        left_sf.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left = left_sf.body
        left_sf.scroll_to_top()

        self.right_frame = ttk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        # -------- Left Panel Widgets --------
        ttk.Label(left, text="Metalens PDK", font=("Inter", 18, "bold")).pack(anchor="w", pady=(0, 5), padx=5)

        self.info = tk.StringVar(value="")
        self.warn = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.info, wraplength=420, bootstyle=INFO).pack(anchor="w", padx=5)
        ttk.Label(left, textvariable=self.warn, wraplength=420, bootstyle=DANGER).pack(anchor="w", pady=(0, 10), padx=5)

        self._build_all_controls(left)

        # -------- Right Panel Matplotlib Figure (initially empty) --------
        plt.style.use("dark_background")
        mpl.rcParams.update({
            "figure.facecolor": "#222b31", # from ttkbootstrap superhero theme
            "axes.facecolor": "#222b31",
            "text.color": "white",
            "axes.labelcolor": "lightgray",
            "xtick.color": "lightgray",
            "ytick.color": "lightgray",
            "font.size": 8 # Smaller font size for plots
        })
        
        self._show_welcome_message() # NEW: Show welcome message instead of plots
        self._on_any_change(from_init=True)
        
    def _clear_right_panel(self):
        """Destroys any widget currently in the right-side frame."""
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        self.right_panel_widget = None

    def _show_welcome_message(self):
        """Displays a welcome message in the right panel."""
        self._clear_right_panel()
        welcome_label = ttk.Label(
            self.right_frame,
            text="Welcome to the Metalens PDK\n\nSelect your parameters on the left and click one of the action buttons.",
            font=("Inter", 14),
            justify="center",
            anchor="center"
        )
        welcome_label.pack(fill="both", expand=True)
        self.right_panel_widget = welcome_label

    def _create_control_group(self, parent, title):
        frame = ttk.Labelframe(parent, text=title, bootstyle=PRIMARY, padding=(10, 5))
        frame.pack(fill=tk.X, expand=True, padx=5, pady=(0, 10))
        return frame

    def _add_option_menu(self, parent, label, var, choices, callback):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(5, 2))
        menu = ttk.OptionMenu(parent, var, var.get(), *choices, command=callback, bootstyle=SECONDARY)
        menu.pack(fill=tk.X, expand=True)

    def _build_all_controls(self, parent):
        # --- Optical Setup ---
        optical_frame = self._create_control_group(parent, "Optical Setup")
        self.var_wl = tk.StringVar(value="λ = 9 μm")
        self._add_option_menu(optical_frame, "Wavelength λ", self.var_wl, choice_map(WL_CHOICES).keys(), self._on_any_change)

        self.var_f = tk.StringVar(value="25 mm")
        self._add_option_menu(optical_frame, "Working Distance (≈ Focal Length)", self.var_f, choice_map(WD_CHOICES).keys(), self._on_any_change)

        # --- Lens Geometry & Materials ---
        geom_frame = self._create_control_group(parent, "Lens Geometry & Materials")
        self.var_D = tk.StringVar(value="1.5 mm")
        self._add_option_menu(geom_frame, "Lens Aperture (Diameter)", self.var_D, choice_map(DIAM_CHOICES).keys(), self._on_any_change)

        self.var_pitch = tk.StringVar(value="9 µm")
        self._add_option_menu(geom_frame, "Lattice Pitch p (µm)", self.var_pitch, choice_map(PITCH_CHOICES).keys(), self._on_any_change)

        self.var_sub = tk.StringVar(value=list(MATERIALS.keys())[0])
        self._add_option_menu(geom_frame, "Substrate Material", self.var_sub, MATERIALS.keys(), self._on_any_change)

        # --- Meta-Atom Geometry ---
        atom_frame = self._create_control_group(parent, "Meta-Atom Geometry")
        self.var_shape = tk.StringVar(value="Frustum (N-gon)")
        self._add_option_menu(atom_frame, "Shape", self.var_shape, ["Frustum (N-gon)", "Cylinder"], self._on_any_change)

        self.var_sides = tk.IntVar(value=10)
        ttk.Label(atom_frame, text="Sides (N) for Frustum").pack(anchor="w", pady=(5, 2))
        sides_frame = ttk.Frame(atom_frame)
        sides_frame.pack(fill=tk.X, expand=True)
        sides_value_label = ttk.Label(sides_frame, textvariable=self.var_sides, font="-weight bold", width=2)
        sides_value_label.pack(side=tk.RIGHT, padx=(5, 0))
        self.sides_scale = ttk.Scale(sides_frame, from_=8, to=12, variable=self.var_sides, command=self._on_sides_scale_change)
        self.sides_scale.pack(fill=tk.X, expand=True)

        # --- Manufacturing & Performance ---
        mfg_frame = self._create_control_group(parent, "Manufacturing & Performance")
        self.var_method = tk.StringVar(value=METHOD_LIST[1])
        self._add_option_menu(mfg_frame, "Manufacturing Method", self.var_method, METHODS.keys(), self._on_any_change)

        self.var_resin = tk.StringVar(value=list(RESINS.keys())[0])
        self._add_option_menu(mfg_frame, "Resin + Nanoparticle (AM Material)", self.var_resin, RESINS.keys(), self._on_any_change)

        self.compat_lbl = ttk.Label(mfg_frame, text="")
        self.compat_lbl.pack(anchor="w", pady=(6, 0))

        self.var_perf = tk.IntVar(value=80)
        ttk.Label(mfg_frame, text="Expected Focusing Performance (%)").pack(anchor="w", pady=(8, 2))
        perf_frame = ttk.Frame(mfg_frame)
        perf_frame.pack(fill=tk.X, expand=True)
        perf_value_label = ttk.Label(perf_frame, textvariable=self.var_perf, font="-weight bold", width=4)
        perf_value_label.pack(side=tk.RIGHT, padx=(5, 0))
        perf_scale = ttk.Scale(perf_frame, from_=20, to=100, variable=self.var_perf, command=self._on_perf_scale_change)
        perf_scale.pack(fill=tk.X, expand=True)

        # --- Output / Actions ---
        out_frame = self._create_control_group(parent, "Actions")
        self.btn_generate = ttk.Button(out_frame, text="Generate Analysis Plots", command=self.update_plots, bootstyle=SUCCESS)
        self.btn_generate.pack(fill=tk.X, pady=(5, 8))

        btn_grid = ttk.Frame(out_frame)
        btn_grid.pack(fill=tk.X)
        btn_grid.columnconfigure((0, 1), weight=1)
        btn_style = "primary-outline"
        btn_pad = {'padx': 2, 'pady': 2}
        ttk.Button(btn_grid, text="Interactive Layout", command=self._display_vector_layout, bootstyle=btn_style).grid(row=0, column=0, sticky="ew", **btn_pad)
        ttk.Button(btn_grid, text="Show 3D CAD View", command=self._display_3d_view, bootstyle=btn_style).grid(row=0, column=1, sticky="ew", **btn_pad)
        ttk.Button(btn_grid, text="Export GDSII", command=self.export_gdsii, bootstyle=btn_style).grid(row=1, column=0, sticky="ew", **btn_pad)
        ttk.Button(btn_grid, text="Export CSV", command=self.generate_csv_file, bootstyle=btn_style).grid(row=1, column=1, sticky="ew", **btn_pad)

        self.btn_estimate = ttk.Button(out_frame, text="Estimate Writing Time", command=self._estimate_time, bootstyle="secondary-outline")
        self.btn_estimate.pack(fill=tk.X, pady=(8, 0))

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(out_frame, textvariable=self.status, wraplength=400).pack(anchor="w", pady=(8, 0))

    def _mark_plots_stale(self):
        self.btn_generate.config(text="Update Analysis Plots (stale)", bootstyle=(SUCCESS, OUTLINE))

    def _mark_plots_fresh(self):
        self.btn_generate.config(text="Generate Analysis Plots", bootstyle=SUCCESS)
        self.root.update_idletasks()

    def _on_any_change(self, *args, from_init=False):
        self.config["wavelength_um"] = choice_map(WL_CHOICES)[self.var_wl.get()]
        self.config["substrate"] = self.var_sub.get()
        self.config["resin"] = self.var_resin.get()
        self.config["shape"] = "frustum" if self.var_shape.get().lower().startswith("frustum") else "cylinder"
        self.config["sides"] = int(self.var_sides.get())
        self.config["lens_diameter"] = choice_map(DIAM_CHOICES)[self.var_D.get()]
        self.config["pitch_um"] = choice_map(PITCH_CHOICES)[self.var_pitch.get()]
        self.config["working_distance"] = choice_map(WD_CHOICES)[self.var_f.get()]
        self.config["perf_pct"] = float(self.var_perf.get())
        self.config["method"] = self.var_method.get()
        self.sides_scale.config(state=("normal" if self.config["shape"] == "frustum" else "disabled"))
        self.current_layout_data = {}
        self._layout_key = None
        if not from_init:
            self._mark_plots_stale()
        self._update_info_warn()
        self._update_compatibility_label()

    def _update_compatibility_label(self):
        method, resin = self.config["method"], self.config["resin"]
        is_ok = resin in METHODS[method]["compatible_resins"]
        msg = f"Method/Resin: {'OK' if is_ok else 'INCOMPATIBLE'}"
        self.compat_lbl.config(text=msg, bootstyle=(SUCCESS if is_ok else DANGER))

    def _update_info_warn(self):
        sub = self.config["substrate"]
        rng, n = MATERIALS[sub]["lambda_range_um"], MATERIALS[sub]["n"]
        D, f = self.config["lens_diameter"], self.config["working_distance"]
        na = (D / (2.0 * f))
        self.info.set(f"Substrate {sub}: n≈{n:.2f} ({rng[0]}-{rng[1]} μm). Current Lens NA ≈ {na:.3f}.")
        warn_msg = ""
        if na < 0.01:
            warn_msg = "Warning: NA < 0.01 indicates very weak focusing."
        elif na > 0.90:
            warn_msg = "Warning: NA > 0.9 may result in degraded performance."
        self.warn.set(warn_msg)

    # ---------- Core compute ----------
    def _current_cache_key(self): return tuple(self.config.get(k) for k in LAYOUT_KEYS)

    def _compute_layout(self):
        self.status.set("Computing layout...")
        self.root.update_idletasks()
        radius_um = (self.config["lens_diameter"] / 2.0) * 1000.0
        pitch = float(self.config["pitch_um"])
        xs = np.arange(-radius_um, radius_um + 1e-9, pitch)
        ys = np.arange(-radius_um, radius_um + 1e-9, pitch)
        x, y = np.meshgrid(xs, ys, indexing="xy")

        luts = generate_luts_for_method(self.config["substrate"], self.config["resin"],
                                        self.config["wavelength_um"], self.config["pitch_um"],
                                        self.config["method"])
        phase_to_dia, dia_to_phase, dia_to_fab = create_inverse_interpolators(luts["phase_lut"], luts["fab_lut"])

        f_um = self.config["working_distance"] * 1000.0
        wavelength_um = self.config["wavelength_um"]
        r = np.sqrt(x**2 + y**2)
        ideal_phase = (2 * np.pi / wavelength_um) * (np.sqrt(r**2 + f_um**2) - f_um)
        ideal_wrapped = np.mod(ideal_phase, 2 * np.pi)

        diameters = phase_to_dia(ideal_wrapped)
        fab_params = dia_to_fab(diameters)

        active = r <= radius_um
        diameters = np.where(active, diameters, 0.0)
        fab_params = np.where(active, fab_params, 0.0)

        realized_phase = dia_to_phase(np.clip(
            diameters, luts["phase_lut"]["diameters_nm"].min(), luts["phase_lut"]["diameters_nm"].max()))
        realized_phase = np.where(active, np.mod(realized_phase, 2*np.pi), np.nan)

        self.current_layout_data = {
            "x": x, "y": y,
            "active_mask": active,
            "phase_ideal": ideal_wrapped,
            "phase_realized": realized_phase,
            "diameters": diameters, # nm
            "fab_params": fab_params,
            "phase_lut": luts["phase_lut"], "fab_lut": luts["fab_lut"],
            "lens_radius_um": radius_um
        }
        self._layout_key = self._current_cache_key()
        active_d = diameters[active]
        if active_d.size:
            max_d_um = float(active_d.max()) * 1e-3
            if pitch - max_d_um < 0:
                drc_warn = f"DRC Fail: Pillars overlap (pitch {pitch:.2f} < max ø {max_d_um:.2f} µm)."
                self.warn.set((self.warn.get() + " | " if self.warn.get() else "") + drc_warn)
        return True

    def _ensure_layout(self):
        if not self.current_layout_data or self._layout_key != self._current_cache_key():
            return self._compute_layout()
        return True

    def _h_2pi_um(self):
        lam = float(self.config["wavelength_um"])
        n_eff = RESINS[self.config["resin"]]["n_eff"]
        delta_n = max(1e-3, abs(n_eff - 1.0))
        k_shape = 1.15 if self.config.get("shape","frustum")=="frustum" else 1.0
        k_method = METHODS[self.config["method"]].get("h2pi_scale", 1.0)
        k_perf = max(0.6, min(1.2, self.config["perf_pct"]/80.0))
        return (lam / delta_n) * k_shape * k_method * k_perf

    def _pillar_heights_um(self, wrapped_phase):
        h2pi = self._h_2pi_um()
        arr = np.asarray(wrapped_phase, dtype=float)
        return np.where(np.isfinite(arr), (arr / (2.0 * np.pi)) * h2pi, np.nan)
    
    def _on_perf_scale_change(self, value):
        self.var_perf.set(int(round(float(value))))
        self._on_any_change()

    def _on_sides_scale_change(self, value):
        self.var_sides.set(int(round(float(value))))
        self._on_any_change()

    def _on_viewer_scale_change(self, value, var):
        var.set(int(round(float(value))))

    # ---------- Plots ----------
    def update_plots(self):
        if not self._ensure_layout(): return
        self._clear_right_panel() # Clear panel for the plots

        # Adjust figsize to give more room for subplots and colorbars
        self.fig = plt.figure(figsize=(13.5, 9.0)) # Increased figure size
        canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.right_panel_widget = canvas.get_tk_widget()

        d = self.current_layout_data
        
        # Use gridspec to control subplot spacing if needed, but tight_layout should handle this
        (ax1, ax2), (ax3, ax4) = self.fig.subplots(2, 2)
        
        self._draw_img_um(ax1, d["x"], d["y"], d["phase_ideal"], d["active_mask"], "Ideal Phase", cmap="twilight")
        realized_fine = self._upsample_nearest(d["phase_realized"], self.REALIZED_UPSAMPLE)
        mask_fine = np.repeat(np.repeat(d["active_mask"], self.REALIZED_UPSAMPLE, 0), self.REALIZED_UPSAMPLE, 1)
        self._draw_img_um(ax2, d["x"], d["y"], realized_fine, mask_fine, "Realized Phase (plug in data)", cmap="twilight")
        h_um = self._pillar_heights_um(d["phase_realized"])
        self._draw_img_um(ax3, d["x"], d["y"], h_um, d["active_mask"], "Atom Height Map (µm)", cmap="viridis")
        d_um = np.where(d["diameters"] > 0, d["diameters"] * 1e-3, np.nan)
        self._draw_img_um(ax4, d["x"], d["y"], d_um, d["active_mask"], "Atom Base Diameter (µm)", cmap="turbo")

        self.fig.tight_layout(pad=3.0) # Increased padding
        canvas.draw()
        self.status.set("Analysis plots updated.")
        self._mark_plots_fresh()

    def _upsample_nearest(self, arr, factor):
        if factor <= 1 or arr is None: return arr
        return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)

    def _draw_img_um(self, ax, x, y, data, mask, title, cmap="viridis"):
        arr = np.where(mask, data, np.nan)
        finite = np.isfinite(arr)
        vmin, vmax = (np.nanmin(arr), np.nanmax(arr)) if np.any(finite) else (0.0, 1.0)
        if vmin == vmax: vmin, vmax = vmin - 1e-6, vmax + 1e-6
        im = ax.imshow(arr, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (µm)", fontsize=9)
        ax.set_ylabel("y (µm)", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # ---------- NEW: Integrated vector viewer ----------
    def _display_vector_layout(self):
        if not self._ensure_layout(): return
        self._clear_right_panel()
        data = self.current_layout_data
        
        container = ttk.Frame(self.right_frame)
        container.pack(fill="both", expand=True)

        theme_bg = ttk.Style().lookup('TFrame', 'background')

        # --- Top control panel ---
        controls = ttk.Frame(container, padding=(10, 5))
        controls.pack(side=tk.TOP, fill=tk.X)
        controls.columnconfigure((1, 4), weight=1)

        ttk.Label(controls, text="Decimation:").grid(row=0, column=0, padx=(0, 5), sticky='w')
        stride_var = tk.IntVar(value=2)
        ttk.Scale(controls, from_=1, to=20, variable=stride_var, command=lambda v: self._on_viewer_scale_change(v, stride_var)).grid(row=0, column=1, sticky='ew')
        ttk.Label(controls, textvariable=stride_var, width=3).grid(row=0, column=2, padx=(5, 15))

        ttk.Label(controls, text="Min ø (nm):").grid(row=0, column=3, padx=(0, 5), sticky='w')
        mind_var = tk.IntVar(value=20)
        ttk.Scale(controls, from_=10, to=2000, variable=mind_var, command=lambda v: self._on_viewer_scale_change(v, mind_var)).grid(row=0, column=4, sticky='ew')
        ttk.Label(controls, textvariable=mind_var, width=5).grid(row=0, column=5, padx=5)
        count_lbl = ttk.Label(controls, text="", anchor='e')
        count_lbl.grid(row=1, column=0, columnspan=6, sticky='ew', pady=(5,0))

        # --- Matplotlib Figure and Toolbar ---
        fig = plt.figure(figsize=(8.3, 8.3), facecolor=theme_bg)
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.right_panel_widget = container

        ax = fig.add_subplot(111, facecolor=theme_bg)
        ax.set_xlabel("X (μm)"); ax.set_ylabel("Y (μm)")
        ax.set_title("Vector Pillar Layout")

        x, y, dnm, mask = data["x"], data["y"], data["diameters"], data["diameters"] > 0.0
        R = (self.config["lens_diameter"] / 2) * 1000.0
        after_id = None

        def schedule_redraw(*_):
            nonlocal after_id
            if after_id is not None:
                try: self.root.after_cancel(after_id)
                except Exception: pass
            after_id = self.root.after(60, draw_visible)

        def draw_visible():
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            s = max(1, int(stride_var.get())); min_nm = max(1, int(mind_var.get()))
            xs, ys, ds, mk = x[::s, ::s], y[::s, ::s], dnm[::s, ::s], mask[::s, ::s]
            in_view = mk & (ds >= min_nm) & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)

            ax.clear()
            ax.set_facecolor(theme_bg)
            ax.set_xlabel("X (μm)"); ax.set_ylabel("Y (μm)")
            ax.set_title("Vector Pillar Layout")
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.add_patch(Circle((0.0, 0.0), R, fill=False, lw=1.0, ec="white", alpha=0.7))
            
            count = 0
            shape = self.config.get("shape", "frustum")
            if np.any(in_view):
                xs_v, ys_v, ds_v = xs[in_view], ys[in_view], ds[in_view]
                radii_um = 0.5 * ds_v * 1e-3
                if shape == "frustum":
                    N = int(self.config.get("sides", 10))
                    patches = [RegularPolygon((float(cx), float(cy)), N, radius=float(r)) for cx, cy, r in zip(xs_v, ys_v, radii_um)]
                else:
                    patches = [Circle((float(cx), float(cy)), float(r)) for cx, cy, r in zip(xs_v, ys_v, radii_um)]
                coll = PatchCollection(patches, facecolor="#ff5470", edgecolor="none", alpha=0.85)
                ax.add_collection(coll)
                count = len(patches)
            else:
                ax.text(0.5, 0.5, "No pillars in view\n(zoom/pan or adjust controls)", color="w", ha="center", va="center", transform=ax.transAxes, alpha=0.8)
            
            count_lbl.config(text=f"Pillars shown: {count:,} (decimation {s}x, min ø {min_nm} nm)")
            canvas.draw_idle()

        ax.callbacks.connect('xlim_changed', schedule_redraw)
        ax.callbacks.connect('ylim_changed', schedule_redraw)
        canvas.mpl_connect('scroll_event', schedule_redraw)
        canvas.mpl_connect('button_release_event', schedule_redraw)
        ax.set_xlim(-R * 1.05, R * 1.05); ax.set_ylim(-R * 1.05, R * 1.05)
        draw_visible()
        self.status.set("Interactive layout loaded.")
        
    # ---------- NEW: Integrated 3D CAD view ----------
    def _display_3d_view(self):
        if not self._ensure_layout(): return
        self._clear_right_panel()
        self.status.set("Generating 3D view...")
        self.root.update_idletasks()

        data = self.current_layout_data
        x_um = data["x"]
        y_um = data["y"]
        diameters_nm = data["diameters"]
        realized_phase = data["phase_realized"]
        active_mask = data["active_mask"]
        h_um_map = self._pillar_heights_um(realized_phase)

        # Filter active pillars and flatten arrays
        active_x = x_um[active_mask].flatten()
        active_y = y_um[active_mask].flatten()
        active_diameters_nm = diameters_nm[active_mask].flatten()
        active_h_um = h_um_map[active_mask].flatten()

        # Define a reasonable subsampling to keep the plot responsive but representative
        # Only draw a subset for 3D visualization to prevent extreme slowness
        subsample_factor = 2 # Draw every 2nd pillar
        active_x = active_x[::subsample_factor]
        active_y = active_y[::subsample_factor]
        active_diameters_nm = active_diameters_nm[::subsample_factor]
        active_h_um = active_h_um[::subsample_factor]
        
        # Max radius for view
        lens_radius_um = self.config["lens_diameter"] / 2 * 1000

        # Create figure and canvas
        fig = plt.figure(figsize=(12, 8))
        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.right_panel_widget = canvas.get_tk_widget()
        
        # --- Main 3D Axes ---
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#FFFFFF") # White background for contrast with green pillars

        pillar_color = '#0d9488' # A teal/green color
        edge_color = '#084f48' # Darker edge for definition

        num_sides = self.config.get("sides", 10) # N for frustum

        # Function to create frustum polygons for 3D plot
        def create_frustum(cx, cy, h_um, r_base_um, r_top_um, N=10):
            if r_base_um <= 0 or h_um <= 0: return []
            
            # Ensure top radius is not larger than base and not too small
            r_top_um = max(0.01, min(r_top_um, r_base_um * 0.95)) # Top radius slightly smaller than base for frustum
            
            theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
            
            # Base polygon
            x_base = cx + r_base_um * np.cos(theta)
            y_base = cy + r_base_um * np.sin(theta)
            z_base = np.zeros(N)
            base_polygon = list(zip(x_base, y_base, z_base))

            # Top polygon
            x_top = cx + r_top_um * np.cos(theta)
            y_top = cy + r_top_um * np.sin(theta)
            z_top = np.full(N, h_um)
            top_polygon = list(zip(x_top, y_top, z_top))
            
            polygons = [base_polygon, top_polygon] # Base and top faces
            
            # Side faces
            for i in range(N):
                next_i = (i + 1) % N
                side_face = [base_polygon[i], base_polygon[next_i], top_polygon[next_i], top_polygon[i]]
                polygons.append(side_face)
            return polygons

        # Iterate through active pillars and draw frustums
        all_polygons = []
        for i in range(len(active_x)):
            cx, cy = active_x[i], active_y[i]
            d_nm = active_diameters_nm[i]
            h_um = active_h_um[i]

            r_base_um = d_nm * 0.5 * 1e-3 # Base radius in um
            r_top_um = r_base_um * 0.7 # Example: top is 70% of base radius for frustum effect

            if self.config["shape"] == "frustum":
                polys = create_frustum(cx, cy, h_um, r_base_um, r_top_um, N=num_sides)
            else: # Cylinder
                polys = create_frustum(cx, cy, h_um, r_base_um, r_base_um, N=num_sides) # r_top = r_base for cylinder
            all_polygons.extend(polys)
        
        if all_polygons:
            mesh = Poly3DCollection(all_polygons, facecolor=pillar_color, edgecolor=edge_color, linewidth=0.5, alpha=0.9)
            ax.add_collection3d(mesh)

        # --- Appearance settings for main plot ---
        ax.view_init(elev=35, azim=-45)
        ax.axis('off')
        
        # Dynamically set limits based on actual pillar positions and sizes
        x_min, x_max = active_x.min(), active_x.max()
        y_min, y_max = active_y.min(), active_y.max()
        z_min, z_max = 0, active_h_um.max() * 1.2 # Add some buffer to height

        ax.set_xlim(x_min - lens_radius_um * 0.1, x_max + lens_radius_um * 0.1)
        ax.set_ylim(y_min - lens_radius_um * 0.1, y_max + lens_radius_um * 0.1)
        ax.set_zlim(z_min, z_max)
        
        ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min]) # Dynamic aspect ratio
        
        # --- Create Inset Axes ---
        # Position inset axes in top right, adjusted to fit
        inset_ax = fig.add_axes([0.65, 0.65, 0.3, 0.3], projection='3d') # [left, bottom, width, height]
        inset_ax.set_facecolor("#FFFFFF")
        
        # Select a smaller patch of pillars for the inset view (e.g., central region)
        # Using a fixed region relative to the overall lens size for consistency
        inset_center_x = 0 # Center of the lens
        inset_center_y = 0
        inset_region_size_um = lens_radius_um * 0.4 # 40% of lens radius

        inset_mask = (active_x > inset_center_x - inset_region_size_um/2) & \
                     (active_x < inset_center_x + inset_region_size_um/2) & \
                     (active_y > inset_center_y - inset_region_size_um/2) & \
                     (active_y < inset_center_y + inset_region_size_um/2)
        
        inset_x, inset_y = active_x[inset_mask], active_y[inset_mask]
        inset_diameters_nm = active_diameters_nm[inset_mask]
        inset_h_um = active_h_um[inset_mask]

        inset_polygons = []
        for i in range(len(inset_x)):
            cx, cy = inset_x[i], inset_y[i]
            d_nm = inset_diameters_nm[i]
            h_um = inset_h_um[i]

            r_base_um = d_nm * 0.5 * 1e-3
            r_top_um = r_base_um * 0.7

            if self.config["shape"] == "frustum":
                polys = create_frustum(cx, cy, h_um, r_base_um, r_top_um, N=num_sides)
            else: # Cylinder
                polys = create_frustum(cx, cy, h_um, r_base_um, r_base_um, N=num_sides)
            inset_polygons.extend(polys)
            
        if inset_polygons:
            inset_mesh = Poly3DCollection(inset_polygons, facecolor=pillar_color, edgecolor=edge_color, linewidth=0.5, alpha=0.9)
            inset_ax.add_collection3d(inset_mesh)

        # --- Appearance settings for inset plot ---
        inset_ax.view_init(elev=20, azim=-65)
        inset_ax.axis('off')

        # Dynamically set limits for inset
        if len(inset_x) > 0:
            inset_ax.set_xlim(inset_x.min() - inset_region_size_um * 0.1, inset_x.max() + inset_region_size_um * 0.1)
            inset_ax.set_ylim(inset_y.min() - inset_region_size_um * 0.1, inset_y.max() + inset_region_size_um * 0.1)
            inset_ax.set_zlim(z_min, z_max)
            inset_ax.set_box_aspect([inset_x.max()-inset_x.min(), inset_y.max()-inset_y.min(), z_max-z_min])
        else:
            inset_ax.set_xlim(-1,1); inset_ax.set_ylim(-1,1); inset_ax.set_zlim(0,1) # Fallback if no pillars in inset
            
        # --- Draw the red rectangle for the inset region on the main plot ---
        # Convert data coordinates to figure coordinates for the rectangle
        # This is a bit tricky with 3D axes, approximating for now to indicate the region
        
        # Approximate rectangle corners in data coords
        rect_x_min = inset_center_x - inset_region_size_um/2
        rect_x_max = inset_center_x + inset_region_size_um/2
        rect_y_min = inset_center_y - inset_region_size_um/2
        rect_y_max = inset_center_y + inset_region_size_um/2

        # Draw the rectangle on the 2D projection of the 3D axes
        # It's not a perfect box in 3D, but indicative
        ax.plot([rect_x_min, rect_x_max, rect_x_max, rect_x_min, rect_x_min],
                [rect_y_min, rect_y_min, rect_y_max, rect_y_max, rect_y_min],
                zs=[0]*5, color='red', linewidth=2, zorder=10) # Draw on the base plane
        
        canvas.draw()
        self.status.set("Generic 3D CAD view displayed.")
        
    # ---------- Estimate writing time ----------
    def _estimate_time(self):
        if not self._ensure_layout(): return
        d = self.current_layout_data
        mean_wrapped = np.nanmean(d["phase_realized"])
        h_um = (mean_wrapped / (2*np.pi)) * self._h_2pi_um()
        d_nm = d["diameters"]; mask = d_nm > 0
        area_pillars_um2 = (np.pi / 4.0) * np.sum((d_nm[mask] ** 2)) * 1e-6
        hatch_um = 0.5; speed_ums = 5000; layer_thickness_um = 0.5
        layers = max(1, int(np.ceil(max(1.0, h_um) / layer_thickness_um)))
        time_s = layers * (area_pillars_um2 / (speed_ums * hatch_um))
        hrs = time_s / 3600.0
        self.status.set(f"Estimated writing time ≈ {hrs:.2f} h • height≈{h_um:.1f} µm • fill≈{area_pillars_um2/1e6:.3f} mm².")

    # ---------- CSV ----------
    def generate_csv_file(self):
        if not self._ensure_layout(): return
        data = self.current_layout_data
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")], title="Save Metalens Layout")
        if not path: return
        header = f"x_um,y_um,diameter_nm,{data['fab_lut']['param_name'].replace(' ', '_')}"
        x_flat = data["x"].flatten(); y_flat = data["y"].flatten()
        d_flat = data["diameters"].flatten(); f_flat = data["fab_params"].flatten()
        mask = d_flat > 0
        rows = np.stack([x_flat[mask], y_flat[mask], d_flat[mask], f_flat[mask]], axis=1)
        np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.3f")
        self.status.set(f"Saved: {path.split('/')[-1]}")

    # ---------- GDSII export ----------
    def export_gdsii(self):
        if not self._ensure_layout(): return
        data = self.current_layout_data
        try: import gdstk
        except Exception:
            self.status.set("GDS export requires 'gdstk'. Install with: pip install gdstk")
            return
        path = filedialog.asksaveasfilename(defaultextension=".gds", filetypes=[("GDSII files", "*.gds"), ("All files", "*.*")], title="Save GDSII (KLayout)")
        if not path: return
        x = data["x"]; y = data["y"]; dnm = data["diameters"]; mask = dnm > 0.0
        total = int(mask.sum())
        if total == 0:
            self.status.set("Nothing to write: empty layout."); return
        stride = max(1, int(np.ceil(np.sqrt(total / 150_000))))
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        cell = lib.new_cell("METALENS")
        pitch_um = float(abs(x[0,1] - x[0,0])) if x.shape[1] > 1 else 1.0
        h, w = dnm.shape; count = 0; shape = self.config.get("shape", "frustum")
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                if not mask[i, j]: continue
                d_nm = float(dnm[i, j])
                if d_nm < 20.0: continue
                r_um = 0.5 * d_nm * 1e-3; cx = float(x[i, j]); cy = float(y[i, j])
                if shape == "frustum":
                    N = int(self.config.get("sides", 10))
                    poly = gdstk.regular_polygon((cx, cy), r_um, N, layer=1, datatype=0); cell.add(poly)
                else:
                    circ = gdstk.ellipse((cx, cy), r_um, tolerance=0.02, layer=1, datatype=0); cell.add(circ)
                count += 1
        lens_radius_um = (self.config["lens_diameter"] / 2) * 1000.0
        ring_w_um = max(1.0, 0.002 * lens_radius_um)
        outline = gdstk.ellipse((0.0, 0.0), lens_radius_um, inner_radius=lens_radius_um - ring_w_um, tolerance=0.1, layer=100, datatype=0)
        cell.add(outline)
        try: lib.write_gds(path)
        except Exception as e: self.status.set(f"GDS write failed: {e}"); return
        self.status.set(f"GDS saved: {path.split('/')[-1]} | pillars: ~{count:,} | stride={stride}")

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    root.geometry("1920x1080+0+0") # Set size to 1920x1080 and position at top-left
    root.resizable(True, True)
    app = PDK_App(root)
    root.mainloop()
