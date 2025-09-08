import tkinter as tk
from tkinter import ttk, filedialog
import tkinter.font as tkfont
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.collections import PatchCollection

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
        "d_frac": (0.30, 0.80),   # 30–80% of pitch
        "quant_um": 0.10,
        "h2pi_scale": 0.85,
    },
    "fs two-photon AM": {
        "compatible_resins": {"IP-Dip + TiO2", "IP-Dip + SiO2", "ORMOCER + ZrO2"},
        "d_frac": (0.20, 0.90),   # 20–90% of pitch
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
def choice_map(pairs): return {a: b for a, b in pairs}

WL_CHOICES   = [("λ = 9 μm", 9.0), ("λ = 10 μm", 10.0)]
DIAM_CHOICES = [("50 µm", 0.05), ("75 µm", 0.075), ("100 µm", 0.10),  # micro options
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
        self.root.title("Metalens PDK — WIP")
        self.root.configure(bg="#111827")

        # Rendering preference: make realized map look smoother (no new values introduced)
        self.REALIZED_UPSAMPLE = 3  # 1 = original, 2/3/4 = finer visualization

        # -------- Bigger fonts --------
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkTooltipFont"):
            try: tkfont.nametofont(name).configure(size=12)
            except Exception: pass

        style = ttk.Style()
        style.configure("TLabel", background="#1f2937", foreground="white", font=("Inter", 12))
        style.configure("TMenubutton", background="#374151", foreground="white", font=("Inter", 12))
        style.configure("TButton", background="#374151", foreground="white", font=("Inter", 12))
        style.configure("Header.TLabel", background="#1f2937", foreground="#e5e7eb", font=("Inter", 14, "bold"))

        # -------- Defaults (Frustum) --------
        self.config = {
            "wavelength_um": WL_CHOICES[0][1],
            "substrate": list(MATERIALS.keys())[0],
            "resin": list(RESINS.keys())[0],
            "shape": "frustum",          # default geometry
            "sides": 10,
            "lens_diameter": DIAM_CHOICES[3][1],  # 1.5 mm default
            "pitch_um": PITCH_CHOICES[2][1],
            "working_distance": WD_CHOICES[1][1],
            "perf_pct": 80.0,
            "method": METHOD_LIST[1],    # fs two-photon AM
        }
        self.current_layout_data = {}
        self._layout_key = None

        # -------- Frames --------
        left = tk.Frame(root, bg="#1f2937", padx=18, pady=18); left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        right = tk.Frame(root, bg="#111827"); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(left, text="Design Controls", style="Header.TLabel").pack(anchor="w", pady=(0, 8))
        self.info = tk.StringVar(value=""); self.warn = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.info, wraplength=380, foreground="#c7d2fe").pack(anchor="w")
        ttk.Label(left, textvariable=self.warn, wraplength=380, foreground="#fca5a5").pack(anchor="w", pady=(0, 6))

        self.controls_frame = tk.Frame(left, bg="#1f2937"); self.controls_frame.pack(fill=tk.X)
        self._build_all_controls(self.controls_frame)

        out = tk.Frame(left, bg="#1f2937"); out.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(out, text="Output", style="Header.TLabel").pack(anchor="w", pady=(0, 6))

        self.btn_generate = tk.Button(out, text="Generate phase shift profile",
                                      bg="#2563eb", fg="white", font=("Inter", 12, "bold"),
                                      relief="raised", activebackground="#1d4ed8",
                                      command=self.update_plots)
        self.btn_generate.pack(fill=tk.X, pady=(2, 6))

        self.btn_viewer = tk.Button(out, text="Open Interactive Lens Layout (vector)",
                                    bg="#9333ea", fg="white", font=("Inter", 12, "bold"),
                                    relief="raised", activebackground="#7e22ce",
                                    command=self._open_vector_layout_viewer)
        self.btn_viewer.pack(fill=tk.X, pady=(0, 6))

        self.btn_cad3d = tk.Button(out, text="Open 3D CAD view (full lens)",
                                   bg="#d97706", fg="white", font=("Inter", 12, "bold"),
                                   relief="raised", activebackground="#b45309",
                                   command=self._open_3d_cad_plotly_full)
        self.btn_cad3d.pack(fill=tk.X, pady=(0, 6))

        self.btn_export_gds = tk.Button(out, text="Export GDSII (KLayout)",
                                        bg="#ea580c", fg="white", font=("Inter", 12, "bold"),
                                        relief="raised", activebackground="#c2410c",
                                        command=self.export_gdsii)
        self.btn_export_gds.pack(fill=tk.X, pady=(0, 6))

        self.btn_export = tk.Button(out, text="Export CSV",
                                    bg="#059669", fg="white", font=("Inter", 12, "bold"),
                                    relief="raised", activebackground="#047857",
                                    command=self.generate_csv_file)
        self.btn_export.pack(fill=tk.X, pady=(0, 6))

        self.btn_estimate = tk.Button(out, text="Estimate writing time (realistic defaults; no fab info)",
                                      bg="#6b7280", fg="white", font=("Inter", 12, "bold"),
                                      relief="raised", activebackground="#4b5563",
                                      command=self._estimate_time)
        self.btn_estimate.pack(fill=tk.X)

        self.status = tk.StringVar(value="")
        ttk.Label(out, textvariable=self.status, wraplength=380).pack(anchor="w", pady=(8, 0))

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(12.5, 8.0), facecolor="#111827")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._update_info_warn()

    # ---------- Build all controls ----------
    def _build_all_controls(self, parent):
        def add_label(txt): ttk.Label(parent, text=txt).pack(anchor="w", pady=(10, 2))

        add_label("1) Wavelength λ")
        self.var_wl = tk.StringVar(value="λ = 9 μm")
        ttk.OptionMenu(parent, self.var_wl, self.var_wl.get(),
                       *choice_map(WL_CHOICES).keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        add_label("2) Substrate / material")
        self.var_sub = tk.StringVar(value=list(MATERIALS.keys())[0])
        ttk.OptionMenu(parent, self.var_sub, self.var_sub.get(), *MATERIALS.keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        add_label("3) Resin + nanoparticle (AM material)")
        self.var_resin = tk.StringVar(value=list(RESINS.keys())[0])
        ttk.OptionMenu(parent, self.var_resin, self.var_resin.get(), *RESINS.keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        add_label("3b) Select meta-atom geometry (Frustum N=8–12, or Cylinder)")
        self.var_shape = tk.StringVar(value="Frustum (N-gon)")
        ttk.OptionMenu(parent, self.var_shape, self.var_shape.get(),
                       "Frustum (N-gon)", "Cylinder",
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)
        self.var_sides = tk.IntVar(value=10)
        self.sides_scale = tk.Scale(parent, from_=8, to=12, orient="horizontal",
                                    showvalue=True, resolution=1, length=320,
                                    variable=self.var_sides, command=lambda *_: self._on_any_change())
        self.sides_scale.pack(fill=tk.X)

        add_label("4) Lens aperture (diameter)")
        self.var_D = tk.StringVar(value="1.5 mm")
        ttk.OptionMenu(parent, self.var_D, self.var_D.get(), *choice_map(DIAM_CHOICES).keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        add_label("4b) Lattice pitch p (µm)")
        self.var_pitch = tk.StringVar(value="9 µm")
        ttk.OptionMenu(parent, self.var_pitch, self.var_pitch.get(), *choice_map(PITCH_CHOICES).keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        add_label("5) Working distance (≈ focal length)")
        self.var_f = tk.StringVar(value="25 mm")
        ttk.OptionMenu(parent, self.var_f, self.var_f.get(), *choice_map(WD_CHOICES).keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        add_label("6) Expected beam focusing performance (% of diffraction limit)")
        self.var_perf = tk.IntVar(value=80)
        tk.Scale(parent, from_=20, to=100, orient="horizontal", showvalue=True,
                 resolution=1, length=320, variable=self.var_perf,
                 command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        add_label("7) Manufacturing method")
        self.var_method = tk.StringVar(value=list(METHODS.keys())[1])  # fs two-photon
        ttk.OptionMenu(parent, self.var_method, self.var_method.get(), *METHODS.keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        self.compat_lbl = ttk.Label(parent, text="", foreground="#c7d2fe")
        self.compat_lbl.pack(anchor="w", pady=(4, 0))
        self._on_any_change()

    # ---------- Respond to any change ----------
    def _on_any_change(self):
        wl_map = choice_map(WL_CHOICES)
        D_map  = choice_map(DIAM_CHOICES)
        p_map  = choice_map(PITCH_CHOICES)
        f_map  = choice_map(WD_CHOICES)

        self.config["wavelength_um"]  = wl_map[self.var_wl.get()]
        self.config["substrate"]      = self.var_sub.get()
        self.config["resin"]          = self.var_resin.get()
        self.config["shape"]          = "frustum" if self.var_shape.get().lower().startswith("frustum") else "cylinder"
        self.config["sides"]          = int(self.var_sides.get())
        self.sides_scale.configure(state=("normal" if self.config["shape"] == "frustum" else "disabled"))

        self.config["lens_diameter"]  = D_map[self.var_D.get()]
        self.config["pitch_um"]       = p_map[self.var_pitch.get()]
        self.config["working_distance"]= f_map[self.var_f.get()]
        self.config["perf_pct"]       = float(self.var_perf.get())
        self.config["method"]         = self.var_method.get()

        self.current_layout_data = {}
        self._layout_key = None

        self._update_info_warn()
        self._update_compatibility_label()

    def _update_compatibility_label(self):
        method = self.config["method"]; resin = self.config["resin"]
        ok = resin in METHODS[method]["compatible_resins"]
        msg = f"Compatibility: {'OK' if ok else 'NOT COMPATIBLE'}  —  method '{method}' vs AM material '{resin}'"
        self.compat_lbl.config(text=msg, foreground=("#34d399" if ok else "#fca5a5"))

    def _update_info_warn(self):
        sub = self.config["substrate"]
        rng = MATERIALS[sub]["lambda_range_um"]; n = MATERIALS[sub]["n"]
        D = self.config["lens_diameter"]; f = self.config["working_distance"]
        na = (D*1000.0) / (2.0 * f * 1000.0)  # D(mm)->um cancels; keep readable
        extra = ""
        if self.config["method"] == "fs two-photon AM":
            build = 100.0  # µm typical
            curr = D * 1000.0  # mm -> µm
            extra = f" | fs two-photon AM build-size limit ≈ {build} µm; current D = {curr:.0f} µm"
        self.info.set(f"{sub}: n≈{n:.2f}, usable λ ≈ {rng[0]}–{rng[1]} μm.  NA≈{na:.3f}{extra}")
        if na < 0.01:
            self.warn.set("Warning: NA < 0.01 → very weak focusing.")
        elif na > 0.90:
            self.warn.set("Warning: NA > 0.9 → performance likely degraded.")
        else:
            self.warn.set("")

    # ---------- Core compute ----------
    def _current_cache_key(self): return tuple(self.config.get(k) for k in LAYOUT_KEYS)

    def _compute_layout(self):
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
            "diameters": diameters,
            "fab_params": fab_params,
            "phase_lut": luts["phase_lut"], "fab_lut": luts["fab_lut"]
        }
        self._layout_key = self._current_cache_key()

        # Simple DRC info
        active_d = diameters[active]
        if active_d.size:
            max_d_um = float(active_d.max()) * 1e-3
            if pitch - max_d_um < 0:
                self.warn.set((self.warn.get() + " | " if self.warn.get() else "") +
                              f"DRC: pillars overlap (pitch {pitch:.2f} µm < max d {max_d_um:.2f} µm).")
        return True

    def _ensure_layout(self):
        if not self.current_layout_data or self._layout_key != self._current_cache_key():
            return self._compute_layout()
        return True

    # ---------- Height/phase helpers ----------
    def _h_2pi_um(self):
        """Physically-informed 2π height (tunable)."""
        lam   = float(self.config["wavelength_um"])          # µm
        n_eff = RESINS[self.config["resin"]]["n_eff"]
        n_env = 1.0  # air; adjust if using immersion/cladding
        delta_n = max(1e-3, abs(n_eff - n_env))
        k_shape  = 1.15 if self.config.get("shape","frustum")=="frustum" else 1.0
        k_method = METHODS[self.config["method"]].get("h2pi_scale", 1.0)  # <-- FIXED indexing
        k_perf   = max(0.6, min(1.2, self.config["perf_pct"]/80.0))
        return (lam / delta_n) * k_shape * k_method * k_perf

    def _pillar_heights_um(self, wrapped_phase):
        """Map wrapped phase [0..2π] → pillar height using _h_2pi_um()."""
        h2pi = self._h_2pi_um()
        arr = np.asarray(wrapped_phase, dtype=float)
        out = np.where(np.isfinite(arr), (arr / (2.0 * np.pi)) * h2pi, np.nan)
        return out

    # ---------- Plots ----------
    def update_plots(self):
        if not self._ensure_layout(): return
        d = self.current_layout_data

        self.fig.clear()
        (self.ax1, self.ax2), (self.ax3, self.ax4) = self.fig.subplots(2, 2)
        ax1, ax2, ax3, ax4 = self.ax1, self.ax2, self.ax3, self.ax4

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_facecolor("#111827")

        # 1) IDEAL PHASE (wrapped)
        self._draw_img_um(ax1, d["x"], d["y"], d["phase_ideal"], d["active_mask"],
                          "Ideal phase (wrapped)", cmap="twilight")
        # 2) REALIZED PHASE (wrapped, finer viz)
        realized = d["phase_realized"]
        realized_fine = self._upsample_nearest(realized, self.REALIZED_UPSAMPLE)
        self._draw_img_um(ax2, d["x"], d["y"], realized_fine,  # extent still original; imshow handles it
                          np.repeat(np.repeat(d["active_mask"], self.REALIZED_UPSAMPLE, 0),
                                    self.REALIZED_UPSAMPLE, 1),
                          "Realized phase (wrapped, finer viz)", cmap="twilight")

        # 3) ATOM HEIGHT MAP (µm) from realized phase
        h_um = self._pillar_heights_um(d["phase_realized"])
        self._draw_img_um(ax3, d["x"], d["y"], h_um, d["active_mask"],
                          "Atom height map (µm)", cmap="viridis")

        # 4) BASE DIAMETER MAP (µm)
        d_um = np.where(d["diameters"] > 0, d["diameters"] * 1e-3, np.nan)  # nm -> µm
        self._draw_img_um(ax4, d["x"], d["y"], d_um, d["active_mask"],
                          "Atom base diameter map (µm)", cmap="turbo")

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
        self.status.set("Ideal + realized phase maps generated (finer viz enabled for realized).")

    def _upsample_nearest(self, arr, factor):
        if factor <= 1 or arr is None:
            return arr
        return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)

    def _draw_img_um(self, ax, x, y, data, mask, title, cmap="viridis"):
        arr = np.where(mask, data, np.nan)
        finite = np.isfinite(arr)
        if np.any(finite):
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            if vmin == vmax:  # avoid singular color range
                vmin, vmax = vmin - 1e-6, vmax + 1e-6
        else:
            vmin, vmax = 0.0, 1.0

        im = ax.imshow(arr, extent=[x.min(), x.max(), y.min(), y.max()],
                       origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_aspect("equal", adjustable="box")
        cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Small, readable ticks
        cbar.ax.tick_params(labelsize=9)

    # ---------- Interactive vector viewer (view-culling) ----------
    def _open_vector_layout_viewer(self):
        if not self._ensure_layout(): return
        data = self.current_layout_data
        top = tk.Toplevel(self.root); top.title("Interactive Lens Layout — Vector Pillars (μm units; view-culling)")

        controls = tk.Frame(top); controls.pack(side=tk.TOP, fill=tk.X)
        tk.Label(controls, text="Decimation (stride):", font=("Inter", 12)).pack(side=tk.LEFT, padx=(8, 4))
        stride_var = tk.IntVar(value=2)
        tk.Scale(controls, from_=1, to=20, orient="horizontal", variable=stride_var,
                 showvalue=True, length=240, command=lambda *_: schedule_redraw()).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(controls, text="Min diameter (nm):", font=("Inter", 12)).pack(side=tk.LEFT)
        mind_var = tk.IntVar(value=20)
        tk.Scale(controls, from_=10, to=2000, orient="horizontal", variable=mind_var,
                 showvalue=True, length=240, command=lambda *_: schedule_redraw()).pack(side=tk.LEFT, padx=(0, 12))
        count_lbl = tk.Label(controls, text="", font=("Inter", 12)); count_lbl.pack(side=tk.RIGHT, padx=8)

        fig = plt.figure(figsize=(8.3, 8.3), facecolor="#111827")
        canvas = FigureCanvasTkAgg(fig, master=top); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top, pack_toolbar=False); toolbar.update(); toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        ax = fig.add_subplot(111); ax.set_aspect("equal", adjustable="box"); ax.set_facecolor("#111827")
        ax.set_xlabel("X (μm)"); ax.set_ylabel("Y (μm)"); ax.set_title("Vector Pillar Layout (view-culling)")

        x = data["x"]; y = data["y"]; dnm = data["diameters"]; mask = dnm > 0.0
        R = (self.config["lens_diameter"] / 2) * 1000.0
        after_id = None

        def schedule_redraw(*_):
            nonlocal after_id
            if after_id is not None:
                try: top.after_cancel(after_id)
                except Exception: pass
            after_id = top.after(60, draw_visible)

        def draw_visible():
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            s = max(1, int(stride_var.get())); min_nm = max(1, int(mind_var.get()))
            xs = x[::s, ::s]; ys = y[::s, ::s]; ds = dnm[::s, ::s]; mk = mask[::s, ::s]
            in_view = mk & (ds >= min_nm) & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)

            ax.clear()
            ax.set_aspect("equal", adjustable="box"); ax.set_facecolor("#111827")
            ax.set_xlabel("X (μm)"); ax.set_ylabel("Y (μm)"); ax.set_title("Vector Pillar Layout (view-culling)")
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

            outline = Circle((0.0, 0.0), R, fill=False, lw=1.0, ec="white", alpha=0.7)
            ax.add_patch(outline)

            count = 0
            shape = self.config.get("shape", "frustum")
            if np.any(in_view):
                xs_v = xs[in_view]; ys_v = ys[in_view]; ds_v = ds[in_view]
                radii_um = 0.5 * ds_v * 1e-3
                if shape == "frustum":
                    N = int(self.config.get("sides", 10))
                    patches = [RegularPolygon((float(cx), float(cy)), N, radius=float(r), orientation=0.0)
                               for cx, cy, r in zip(xs_v, ys_v, radii_um)]
                else:
                    patches = [Circle((float(cx), float(cy)), float(r))
                               for cx, cy, r in zip(xs_v, ys_v, radii_um)]
                coll = PatchCollection(patches, facecolor="#ff5470", edgecolor="none", alpha=0.85, antialiased=False)
                ax.add_collection(coll); count = len(patches)
            else:
                ax.text(0.5, 0.5, "No pillars in view\n(zoom/pan or lower min d)", color="w",
                        ha="center", va="center", transform=ax.transAxes, alpha=0.8)

            count_lbl.config(text=f"Pillars shown: {count:,}  (stride {s}, min d {min_nm} nm)")
            canvas.draw_idle()

        ax.callbacks.connect('xlim_changed', lambda _ax: schedule_redraw())
        ax.callbacks.connect('ylim_changed', lambda _ax: schedule_redraw())
        canvas.mpl_connect('scroll_event', schedule_redraw)
        canvas.mpl_connect('button_release_event', schedule_redraw)

        f = 0.10
        ax.set_xlim(-R * f, R * f); ax.set_ylim(-R * f, R * f)
        draw_visible()

    # ---------- Estimate writing time ----------
    def _estimate_time(self):
        if not self._ensure_layout(): return
        d = self.current_layout_data
        # average realized phase as a fraction of 2π, then infer typical height
        mean_wrapped = np.nanmean(d["phase_realized"])
        h_um = (mean_wrapped / (2*np.pi)) * self._h_2pi_um()

        d_nm = d["diameters"]; mask = d_nm > 0
        area_pillars_um2 = (np.pi / 4.0) * np.sum((d_nm[mask] ** 2)) * 1e-6

        hatch_um = 0.5; speed_ums = 5000; layer_thickness_um = 0.5
        layers = max(1, int(np.ceil(max(1.0, h_um) / layer_thickness_um)))
        time_s = layers * (area_pillars_um2 / (speed_ums * hatch_um))
        hrs = time_s / 3600.0

        self.status.set(
            f"Estimated writing time ≈ {hrs:.2f} h • height≈{h_um:.1f} µm • fill≈{area_pillars_um2/1e6:.3f} mm² "
            f"(realistic defaults; no fab info)."
        )

    # ---------- CSV ----------
    def generate_csv_file(self):
        if not self._ensure_layout(): return
        data = self.current_layout_data
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                    title="Save Metalens Layout")
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
        try:
            import gdstk
        except Exception:
            self.status.set("GDS export requires 'gdstk'. Install with: pip install gdstk")
            return

        path = filedialog.asksaveasfilename(defaultextension=".gds",
                    filetypes=[("GDSII files", "*.gds"), ("All files", "*.*")],
                    title="Save GDSII (KLayout)")
        if not path: return

        x = data["x"]; y = data["y"]; dnm = data["diameters"]; mask = dnm > 0.0
        total = int(mask.sum())
        if total == 0:
            self.status.set("Nothing to write: empty layout."); return

        target = 150_000
        stride = max(1, int(np.ceil(np.sqrt(total / target))))
        min_d_nm = 20.0

        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        cell = lib.new_cell("METALENS")

        pitch_um = float(abs(x[0,1] - x[0,0])) if x.shape[1] > 1 else 1.0

        h, w = dnm.shape; count = 0; shape = self.config.get("shape", "frustum")
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                if not mask[i, j]: continue
                d_nm = float(dnm[i, j])
                if d_nm < min_d_nm: continue
                r_um = 0.5 * d_nm * 1e-3; cx = float(x[i, j]); cy = float(y[i, j])
                if shape == "frustum":
                    N = int(self.config.get("sides", 10))
                    poly = gdstk.regular_polygon((cx, cy), r_um, N, layer=1, datatype=0); cell.add(poly)
                else:
                    circ = gdstk.ellipse((cx, cy), r_um, tolerance=0.02, layer=1, datatype=0); cell.add(circ)
                count += 1

        lens_radius_um = (self.config["lens_diameter"] / 2) * 1000.0
        ring_w_um = max(1.0, 0.002 * lens_radius_um)
        outline = gdstk.ellipse((0.0, 0.0), lens_radius_um,
                                inner_radius=lens_radius_um - ring_w_um,
                                tolerance=0.1, layer=100, datatype=0)
        cell.add(outline)

        try:
            lib.write_gds(path)
        except Exception as e:
            self.status.set(f"GDS write failed: {e}"); return

        self.status.set(
            f"GDS saved: {path.split('/')[-1]}  |  pillars: ~{count:,}  |  pitch≈{pitch_um:.3f} μm  |  stride={stride}"
        )

    # ---------- 3D CAD (Plotly/WebGL) — FULL LENS, minimal controls ----------
    def _open_3d_cad_plotly_full(self):
        """Full-lens 3D with capped frustums colored by realized phase (rings visible).
        Controls: Stride + Min diameter. Geometry/sides come from main inputs."""
        if not self._ensure_layout():
            return
        try:
            import tempfile
            import plotly.graph_objects as go
            from plotly.offline import plot as plotly_plot
        except Exception:
            self.status.set("3D CAD view needs Plotly. Install: pip install plotly")
            return

        d = self.current_layout_data
        x = d["x"]; y = d["y"]; dnm = d["diameters"]; phase = d["phase_realized"]
        mask = dnm > 0.0
        if not np.any(mask):
            self.status.set("No active pillars to render."); return

        R = (self.config["lens_diameter"] / 2.0) * 1000.0
        H2pi = self._h_2pi_um()
        h_um = (phase / (2.0 * np.pi)) * H2pi

        top = tk.Toplevel(self.root); top.title("3D CAD (Plotly/WebGL) — full lens")

        row = tk.Frame(top); row.pack(side=tk.TOP, fill=tk.X)
        tk.Label(row, text="Stride:", font=("Inter", 12)).pack(side=tk.LEFT, padx=(8, 4))
        stride_var = tk.IntVar(value=2)
        tk.Scale(row, from_=1, to=40, orient="horizontal", variable=stride_var,
                showvalue=True, length=240).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(row, text="Min d (nm):", font=("Inter", 12)).pack(side=tk.LEFT)
        ds_all = dnm[mask].astype(float)
        dmax = float(np.nanmax(ds_all))
        mind_var = tk.DoubleVar(value=20.0)
        tk.Scale(row, from_=10.0, to=max(100.0, 1.2*dmax), resolution=1.0,
                orient="horizontal", variable=mind_var, showvalue=True, length=260).pack(side=tk.LEFT, padx=(0, 12))

        btns = tk.Frame(top); btns.pack(side=tk.TOP, fill=tk.X, pady=(6, 6))
        tk.Button(btns, text="Render", bg="#d97706", fg="white",
                font=("Inter", 12, "bold"),
                command=lambda: render()).pack(side=tk.LEFT, padx=8)
        status_lbl = tk.Label(btns, text="", font=("Inter", 12)); status_lbl.pack(side=tk.LEFT, padx=8)

        # ---------- mesh builder with top/bottom caps + per-vertex colors ----------
        def build_frustums(cx, cy, hh, d_nm, phi):
            Vx = []; Vy = []; Vz = []; I = []; J = []; K = []; VC = []
            vert_ofs = 0
            rb = 0.5 * (d_nm * 1e-3)  # base radius (µm)
            taper = 0.55 if self.config.get("shape", "frustum") == "frustum" else 1.0
            rt = rb * taper
            sides = int(max(3, self.config.get("sides", 10)))
            ang = np.linspace(0, 2*np.pi, sides, endpoint=False)
            ca, sa = np.cos(ang), np.sin(ang)

            # map phase to color with a gentle built-in Plotly colorscale
            def phase_color(u):
                # u in [0, 2π)
                t = (u % (2*np.pi)) / (2*np.pi)
                # shades of twilight-ish: interpolate between key colors
                return f"hsl({int(360*t)}, 60%, 60%)"

            def add_tri(a, b, c): I.append(a); J.append(b); K.append(c)

            for Xc, Yc, H, RB, RT, ph in zip(cx, cy, hh, rb, rt, phi):
                xb = Xc + RB * ca; yb = Yc + RB * sa; zb = np.zeros_like(xb)
                xt = Xc + RT * ca; yt = Yc + RT * sa; zt = np.full_like(xt, H)
                Vx.extend(xb); Vy.extend(yb); Vz.extend(zb)
                Vx.extend(xt); Vy.extend(yt); Vz.extend(zt)
                base = vert_ofs
                topb = base + sides
                c = phase_color(ph)

                # side walls
                for k in range(sides):
                    v0 = base + k
                    v1 = base + ((k + 1) % sides)
                    v2 = topb + k
                    v3 = topb + ((k + 1) % sides)
                    add_tri(v0, v1, v2)
                    add_tri(v1, v3, v2)
                # top cap
                ct = base + 2 * sides
                Vx.append(Xc); Vy.append(Yc); Vz.append(H)
                for k in range(sides):
                    add_tri(ct, topb + k, topb + ((k + 1) % sides))
                # bottom cap
                cb = ct + 1
                Vx.append(Xc); Vy.append(Yc); Vz.append(0.0)
                for k in range(sides):
                    add_tri(cb, base + ((k + 1) % sides), base + k)
                # vertex colors (constant per vertex here)
                VC.extend([c] * (2*sides + 2))
                vert_ofs = cb + 1

            return np.array(Vx), np.array(Vy), np.array(Vz), np.array(I), np.array(J), np.array(K), np.array(VC)

        def select_points():
            s0 = int(max(1, stride_var.get()))
            xs = x[::s0, ::s0]; ys = y[::s0, ::s0]
            ds = dnm[::s0, ::s0]; hs = h_um[::s0, ::s0]; ph = phase[::s0, ::s0]
            rr = np.sqrt(xs**2 + ys**2)
            mk = (ds >= float(mind_var.get())) & (rr <= R)

            MAX_TRIANGLES = 500_000
            sides_here = int(max(3, self.config.get("sides", 10)))
            max_by_tri = max(1000, MAX_TRIANGLES // (4 * sides_here))
            MAX_PILLARS = min(15000, max_by_tri)

            n = int(np.count_nonzero(mk))
            if n > MAX_PILLARS:
                extra = int(np.ceil(np.sqrt(n / MAX_PILLARS)))
                Iidx, Jidx = np.indices(mk.shape)
                mk &= (Iidx % extra == 0) & (Jidx % extra == 0)
                n = int(np.count_nonzero(mk))

            return xs[mk], ys[mk], ds[mk].astype(float), hs[mk].astype(float), ph[mk].astype(float), n

        def render():
            status_lbl.config(text="Rendering…"); top.update_idletasks()
            xs, ys, ds, hs, phs, n = select_points()
            if n == 0:
                status_lbl.config(text="Nothing to render (lower min d or stride)."); return

            Vx, Vy, Vz, I, J, K, VC = build_frustums(xs, ys, hs, ds, phs)

            mesh = go.Mesh3d(
                x=Vx, y=Vy, z=Vz, i=I, j=J, k=K,
                vertexcolor=VC, opacity=1.0, showscale=False, hoverinfo="skip",
                lighting=dict(ambient=0.45, diffuse=0.7, specular=0.9, roughness=0.18, fresnel=0.2),
                lightposition=dict(x=0, y=0, z=2000)
            )

            th = np.linspace(0, 2*np.pi, 512)
            outline = go.Scatter3d(x=R*np.cos(th), y=R*np.sin(th), z=np.zeros_like(th),
                                mode="lines", line=dict(color="rgba(255,255,255,0.6)", width=2),
                                showlegend=False, hoverinfo="skip")

            fig = go.Figure([mesh, outline])
            fig.update_layout(
                scene=dict(
                    xaxis_title="X (μm)", yaxis_title="Y (μm)", zaxis_title="Height (μm)",
                    aspectmode="data",
                    xaxis=dict(backgroundcolor="#111827", showgrid=False, zeroline=False),
                    yaxis=dict(backgroundcolor="#111827", showgrid=False, zeroline=False),
                    zaxis=dict(backgroundcolor="#111827", showgrid=True,  zeroline=False),
                ),
                template="plotly_dark",
                margin=dict(l=0, r=0, t=40, b=0),
                scene_camera=dict(eye=dict(x=0.01, y=0.01, z=2.8))
            )

            import tempfile
            from plotly.offline import plot as plotly_plot
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html"); tmp.close()
            plotly_plot(fig, filename=tmp.name, auto_open=True)
            status_lbl.config(text=f"Opened in browser (pillars: {n:,}).")

        render()

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PDK_App(root)
    root.mainloop()
