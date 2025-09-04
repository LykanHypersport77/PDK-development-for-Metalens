import tkinter as tk
from tkinter import ttk, filedialog
import tkinter.font as tkfont
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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

# ----------------------------- LUT scaled to physical pitch -----------------------------
def generate_generic_luts(substrate, resin, wavelength_um, pitch_um):
    """Process-agnostic LUTs scaled to the physical pitch (µm-scale diameters)."""
    neff = RESINS[resin]["n_eff"]
    d_um = np.linspace(0.2 * pitch_um, 0.9 * pitch_um, 80)  # 20–90% of pitch
    diameters_nm = d_um * 1000.0

    amp = 1.5 + 0.4 * (neff - 1.6)  # mild dependence on index
    phases = np.mod(amp * np.pi * np.sin(np.pi * (d_um - 0.2 * pitch_um) / (0.7 * pitch_um)), 2 * np.pi)

    fab_param = 1000 + 120 * (d_um - 0.2 * pitch_um)  # keep 4th plot alive
    fab_name = "Fabrication Proxy (a.u.)"
    return {
        "phase_lut": {"diameters_nm": diameters_nm, "phases_rad": phases},
        "fab_lut":   {"diameters_nm": diameters_nm, "param_values": fab_param, "param_name": fab_name},
    }

def create_inverse_interpolators(phase_lut, fab_lut):
    ph = np.mod(phase_lut["phases_rad"], 2 * np.pi)
    dia = phase_lut["diameters_nm"]
    order = np.argsort(ph)
    ph_u, idx = np.unique(ph[order], return_index=True)
    dia_u = dia[order][idx]
    ph_aug = np.concatenate([ph_u, ph_u[:1] + 2 * np.pi])
    dia_aug = np.concatenate([dia_u, dia_u[:1]])
    phase_to_dia = interp1d(ph_aug, dia_aug, bounds_error=False, fill_value="extrapolate")
    dia_to_fab = interp1d(
        fab_lut["diameters_nm"], fab_lut["param_values"],
        bounds_error=False, fill_value=(fab_lut["param_values"][0], fab_lut["param_values"][-1])
    )
    return phase_to_dia, dia_to_fab

# ----------------------------- Config keys -----------------------------
LAYOUT_KEYS = [
    "wavelength_um","substrate","resin",
    "shape","sides",
    "lens_diameter","pitch_um","working_distance","perf_pct"
]

# String → numeric maps for the visible controls
WL_CHOICES = [("λ = 9 μm", 9.0), ("λ = 10 μm", 10.0)]
DIAM_CHOICES = [("1.5 mm", 1.5), ("10 mm", 10.0), ("15 mm", 15.0), ("20 mm", 20.0), ("25 mm", 25.0)]
PITCH_CHOICES = [("5 µm", 5.0), ("7 µm", 7.0), ("9 µm", 9.0), ("12 µm", 12.0)]
WD_CHOICES = [("5 mm", 5.0), ("25 mm", 25.0), ("50 mm", 50.0), ("100 mm", 100.0), ("200 mm", 200.0)]

def choice_map(pairs): return {a: b for a, b in pairs}

# ----------------------------- App -----------------------------
class PDK_App:
    def _h_2pi_um(self):
        """Height to produce ~2π phase (heuristic, same model used in estimate)."""
        neff = RESINS[self.config["resin"]]["n_eff"]
        shape = self.config.get("shape", "cylinder")
        h_ref = 10.0 if shape == "frustum" else 5.0  # µm @ n≈1.5
        return h_ref * (1.5 / neff) * max(0.6, min(1.2, self.config["perf_pct"] / 80.0))

    def _pillar_heights_um(self, wrapped_phase):
        """Map wrapped phase [0..2π) -> local pillar height using _h_2pi_um."""
        return (wrapped_phase / (2.0 * np.pi)) * self._h_2pi_um()

    def __init__(self, root):
        self.root = root
        self.root.title("Metalens PDK — Guided Flow")
        self.root.configure(bg="#111827")

        # -------- Bigger GUI text everywhere --------
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkTooltipFont"):
            try:
                tkfont.nametofont(name).configure(size=12)
            except Exception:
                pass

        # ttk styles
        style = ttk.Style()
        style.configure("TLabel", background="#1f2937", foreground="white", font=("Inter", 12))
        style.configure("TMenubutton", background="#374151", foreground="white", font=("Inter", 12))
        style.configure("TButton", background="#374151", foreground="white", font=("Inter", 12))
        style.configure("Header.TLabel", background="#1f2937", foreground="#e5e7eb", font=("Inter", 14, "bold"))

        self.config = {
            "wavelength_um": WL_CHOICES[0][1],
            "substrate": list(MATERIALS.keys())[0],
            "resin": list(RESINS.keys())[0],
            "shape": "cylinder",  # 'cylinder' or 'frustum'
            "sides": 10,
            "lens_diameter": DIAM_CHOICES[1][1],  # 10 mm
            "pitch_um": PITCH_CHOICES[2][1],      # 9 µm
            "working_distance": WD_CHOICES[1][1], # 25 mm
            "perf_pct": 80.0,
        }
        self.current_layout_data = {}
        self._layout_key = None  # cache-key for layout

        # -------- Layout frames --------
        left = tk.Frame(root, bg="#1f2937", padx=18, pady=18)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        right = tk.Frame(root, bg="#111827")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(left, text="Design Controls", style="Header.TLabel").pack(anchor="w", pady=(0, 8))
        self.info = tk.StringVar(value="")
        self.warn = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.info, wraplength=380, foreground="#c7d2fe").pack(anchor="w")
        ttk.Label(left, textvariable=self.warn, wraplength=380, foreground="#fca5a5").pack(anchor="w", pady=(0, 6))

        self.controls_frame = tk.Frame(left, bg="#1f2937")
        self.controls_frame.pack(fill=tk.X)

        self._build_all_controls(self.controls_frame)

        # Output area (persistent buttons)
        out = tk.Frame(left, bg="#1f2937")
        out.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(out, text="Output", style="Header.TLabel").pack(anchor="w", pady=(0, 6))

        self.btn_generate = tk.Button(out, text="Generate Images", bg="#2563eb", fg="white",
                                      font=("Inter", 12, "bold"), relief="raised",
                                      activebackground="#1d4ed8", command=self.update_plots)
        self.btn_generate.pack(fill=tk.X, pady=(2, 6))

        self.btn_viewer = tk.Button(out, text="Open Interactive Lens Layout (vector)",
                                    bg="#9333ea", fg="white", font=("Inter", 12, "bold"),
                                    relief="raised", activebackground="#7e22ce",
                                    command=self._open_vector_layout_viewer)
        self.btn_viewer.pack(fill=tk.X, pady=(0, 6))

        self.btn_cad3d = tk.Button(
            out, text="Open 3D CAD view",
            bg="#d97706", fg="white", font=("Inter", 12, "bold"),
            relief="raised", activebackground="#b45309",
            command=self._open_3d_cad_plotly
        )
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

        # -------- Plots (smaller graphs) --------
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(7.5, 6.5), facecolor="#111827")  # smaller than before
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial NA info
        self._update_info_warn()

    # ---------- Build all controls at once ----------
    def _build_all_controls(self, parent):
        # Helper to add labeled section
        def add_label(txt):
            ttk.Label(parent, text=txt).pack(anchor="w", pady=(10, 2))

        # 1) Wavelength
        add_label("1) Wavelength λ")
        self.var_wl = tk.StringVar(value=WL_CHOICES[0][0])
        ttk.OptionMenu(parent, self.var_wl, self.var_wl.get(), *choice_map(WL_CHOICES).keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        # 2) Substrate
        add_label("2) Substrate / material")
        self.var_sub = tk.StringVar(value=list(MATERIALS.keys())[0])
        ttk.OptionMenu(parent, self.var_sub, self.var_sub.get(), *MATERIALS.keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        # 3) Resin
        add_label("3) Resin + nanoparticle")
        self.var_resin = tk.StringVar(value=list(RESINS.keys())[0])
        ttk.OptionMenu(parent, self.var_resin, self.var_resin.get(), *RESINS.keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        # 3b) Geometry
        add_label("3b) Meta-atom geometry")
        self.var_shape = tk.StringVar(value="Cylinder")
        ttk.OptionMenu(parent, self.var_shape, self.var_shape.get(),
                       "Cylinder", "Frustum (N-gon)",
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)
        self.var_sides = tk.IntVar(value=10)
        self.sides_scale = tk.Scale(parent, from_=4, to=12, orient="horizontal",
                                    showvalue=True, resolution=1, length=320,
                                    variable=self.var_sides, command=lambda *_: self._on_any_change())
        self.sides_scale.pack(fill=tk.X)
        self.sides_scale.configure(state="disabled")  # enabled when frustum selected

        # 4) Aperture
        add_label("4) Lens aperture (diameter)")
        self.var_D = tk.StringVar(value=DIAM_CHOICES[1][0])
        ttk.OptionMenu(parent, self.var_D, self.var_D.get(), *choice_map(DIAM_CHOICES).keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        # 4b) Lattice pitch
        add_label("4b) Lattice pitch p (µm)")
        self.var_pitch = tk.StringVar(value=PITCH_CHOICES[2][0])
        ttk.OptionMenu(parent, self.var_pitch, self.var_pitch.get(), *choice_map(PITCH_CHOICES).keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        # 5) Working distance
        add_label("5) Working distance (≈ focal length)")
        self.var_f = tk.StringVar(value=WD_CHOICES[1][0])
        ttk.OptionMenu(parent, self.var_f, self.var_f.get(), *choice_map(WD_CHOICES).keys(),
                       command=lambda *_: self._on_any_change()).pack(fill=tk.X)

        # 6) Performance slider
        add_label("6) Expected beam focusing performance (% of diffraction limit)")
        self.var_perf = tk.IntVar(value=80)
        tk.Scale(parent, from_=20, to=100, orient="horizontal", showvalue=True,
                 resolution=1, length=320, variable=self.var_perf,
                 command=lambda *_: self._on_any_change()).pack(fill=tk.X)

    # ---------- Respond to any control change ----------
    def _on_any_change(self):
        wl_map = choice_map(WL_CHOICES)
        D_map = choice_map(DIAM_CHOICES)
        p_map = choice_map(PITCH_CHOICES)
        f_map = choice_map(WD_CHOICES)

        self.config["wavelength_um"] = wl_map[self.var_wl.get()]
        self.config["substrate"] = self.var_sub.get()
        self.config["resin"] = self.var_resin.get()

        shape_sel = self.var_shape.get().lower()
        self.config["shape"] = "cylinder" if shape_sel.startswith("cyl") else "frustum"
        if self.config["shape"] == "frustum":
            self.sides_scale.configure(state="normal")
            self.config["sides"] = int(self.var_sides.get())
        else:
            self.sides_scale.configure(state="disabled")
            self.config["sides"] = int(self.var_sides.get())

        self.config["lens_diameter"] = D_map[self.var_D.get()]
        self.config["pitch_um"] = p_map[self.var_pitch.get()]
        self.config["working_distance"] = f_map[self.var_f.get()]
        self.config["perf_pct"] = float(self.var_perf.get())

        # invalidate cached layout
        self.current_layout_data = {}
        self._layout_key = None

        self._update_info_warn()

    def _update_info_warn(self):
        # Info string about materials and NA
        sub = self.config["substrate"]
        rng = MATERIALS[sub]["lambda_range_um"]; n = MATERIALS[sub]["n"]
        D = self.config["lens_diameter"]; f = self.config["working_distance"]
        na = D / (2.0 * f)
        self.info.set(f"{sub}: n≈{n:.2f}, usable λ range ≈ {rng[0]}–{rng[1]} μm.  NA≈{na:.3f}")
        if na < 0.01:
            self.warn.set("Warning: NA < 0.01 → very weak focusing.")
        elif na > 0.90:
            self.warn.set("Warning: NA > 0.9 → performance likely degraded.")
        else:
            self.warn.set("")

    # ---------- Layout computation & cache ----------
    def _layout_requirements_met(self):
        return set(LAYOUT_KEYS) <= self.config.keys()

    def _current_cache_key(self):
        return tuple(self.config.get(k) for k in LAYOUT_KEYS)

    def _compute_layout(self):
        if not self._layout_requirements_met():
            self.warn.set("Complete all fields first.")
            return False

        radius_um = (self.config["lens_diameter"] / 2.0) * 1000.0
        pitch = float(self.config["pitch_um"])
        xs = np.arange(-radius_um, radius_um + 1e-9, pitch)
        ys = np.arange(-radius_um, radius_um + 1e-9, pitch)
        x, y = np.meshgrid(xs, ys, indexing="xy")

        luts = generate_generic_luts(self.config["substrate"], self.config["resin"],
                                     self.config["wavelength_um"], self.config["pitch_um"])
        phase_to_dia, dia_to_fab = create_inverse_interpolators(luts["phase_lut"], luts["fab_lut"])

        f_um = self.config["working_distance"] * 1000.0
        wavelength_um = self.config["wavelength_um"]
        r = np.sqrt(x**2 + y**2)
        phase = (2 * np.pi / wavelength_um) * (np.sqrt(r**2 + f_um**2) - f_um)
        wrapped = np.mod(phase, 2 * np.pi)

        diameters = phase_to_dia(wrapped)
        fab_params = dia_to_fab(diameters)

        active = r <= radius_um
        diameters = np.where(active, diameters, 0.0)
        fab_params = np.where(active, fab_params, 0.0)

        self.current_layout_data = {
            "x": x, "y": y, "phase": wrapped,
            "diameters": diameters, "fab_params": fab_params,
            "phase_lut": luts["phase_lut"], "fab_lut": luts["fab_lut"]
        }
        self._layout_key = self._current_cache_key()

        # DRC: pitch vs max diameter
        active_d = diameters[diameters > 0]
        if active_d.size:
            max_d_um = float(active_d.max()) * 1e-3
            if pitch - max_d_um < 0:
                self.warn.set((self.warn.get() + " | " if self.warn.get() else "") +
                              f"DRC: pillars overlap (pitch {pitch:.2f} µm < max d {max_d_um:.2f} µm).")
        return True

    def _ensure_layout(self):
        if not self._layout_requirements_met():
            self.warn.set("Complete all fields first.")
            return False
        if not self.current_layout_data or self._layout_key != self._current_cache_key():
            return self._compute_layout()
        return True

    # ---------- Plots (smaller figure) ----------
    def update_plots(self):
        if not self._ensure_layout():
            return
        d = self.current_layout_data
        self.fig.clear()
        ((ax1, ax2), (ax3, ax4)) = self.fig.subplots(2, 2)
        self._draw_lut(ax1, d["phase_lut"])
        self._draw_img(ax2, d["x"], d["y"], d["phase"], "2. Ideal Phase Profile", "hsv")
        self._draw_img(ax3, d["x"], d["y"], d["diameters"], "3. Nanopillar Layout (nm)", "viridis")
        self._draw_img(ax4, d["x"], d["y"], d["fab_params"], f"4. {d['fab_lut']['param_name']}", "plasma")
        self.fig.tight_layout(pad=2.6)
        self.canvas.draw()
        self.status.set("Design images generated.")

    def _draw_lut(self, ax, phase_lut):
        ax.plot(phase_lut["diameters_nm"], phase_lut["phases_rad"], "o-", color="#3b82f6")
        ax.set_title("1. Meta-Atom LUT")
        ax.set_xlabel("Pillar Diameter (nm)")
        ax.set_ylabel("Phase Shift (rad)")
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.grid(True, linestyle="--", alpha=0.5)

    def _draw_img(self, ax, x, y, data, title, cmap):
        valid = data[data > 0]
        vmin, vmax = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)
        im = ax.imshow(
            data,
            extent=[x.min()/1000.0, x.max()/1000.0, y.min()/1000.0, y.max()/1000.0],
            cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
        )
        ax.set_title(title)
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal", adjustable="box")
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---------- Interactive vector viewer (view-culling, starts zoomed-in) ----------
    def _open_vector_layout_viewer(self):
        if not self._ensure_layout():
            return
        data = self.current_layout_data

        top = tk.Toplevel(self.root)
        top.title("Interactive Lens Layout — Vector Pillars (μm units; view-culling)")

        # Controls
        controls = tk.Frame(top)
        controls.pack(side=tk.TOP, fill=tk.X)
        tk.Label(controls, text="Decimation (stride):", font=("Inter", 12)).pack(side=tk.LEFT, padx=(8, 4))
        stride_var = tk.IntVar(value=1)
        tk.Scale(controls, from_=1, to=20, orient="horizontal", variable=stride_var,
                 showvalue=True, length=240, command=lambda *_: schedule_redraw()).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(controls, text="Min diameter (nm):", font=("Inter", 12)).pack(side=tk.LEFT)
        mind_var = tk.IntVar(value=20)
        tk.Scale(controls, from_=10, to=250, orient="horizontal", variable=mind_var,
                 showvalue=True, length=240, command=lambda *_: schedule_redraw()).pack(side=tk.LEFT, padx=(0, 12))

        count_lbl = tk.Label(controls, text="", font=("Inter", 12))
        count_lbl.pack(side=tk.RIGHT, padx=8)

        # Figure + toolbar
        fig = plt.figure(figsize=(8.2, 8.2), facecolor="#111827")
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top, pack_toolbar=False)
        toolbar.update(); toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#111827")
        ax.set_xlabel("X (μm)"); ax.set_ylabel("Y (μm)")
        ax.set_title("Vector Pillar Layout (view-culling)")

        x = data["x"]; y = data["y"]; dnm = data["diameters"]
        mask = dnm > 0.0
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
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor("#111827")
            ax.set_xlabel("X (μm)"); ax.set_ylabel("Y (μm)")
            ax.set_title("Vector Pillar Layout (view-culling)")
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

            outline = Circle((0.0, 0.0), R, fill=False, lw=1.0, ec="white", alpha=0.7)
            ax.add_patch(outline)

            count = 0
            shape = self.config.get("shape", "cylinder")
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

        # connect to pan/zoom/scroll
        ax.callbacks.connect('xlim_changed', lambda _ax: schedule_redraw())
        ax.callbacks.connect('ylim_changed', lambda _ax: schedule_redraw())
        canvas.mpl_connect('scroll_event', schedule_redraw)
        canvas.mpl_connect('button_release_event', schedule_redraw)

        # start zoomed in (~10% radius)
        f = 0.10
        ax.set_xlim(-R * f, R * f); ax.set_ylim(-R * f, R * f)
        draw_visible()

    # ---------- 3D Visualization ----------

    def _open_3d_cad_plotly(self):
        """Open a CAD-like 3D view with tapered frustums rendered with Plotly/WebGL."""
        if not self._ensure_layout():
            return

        try:
            import numpy as _np
            import tempfile, os
            import plotly.graph_objects as go
            from plotly.offline import plot as plotly_plot
        except Exception:
            self.status.set("3D CAD view needs Plotly. Install: pip install plotly")
            return

        d = self.current_layout_data
        x = d["x"]; y = d["y"]; dnm = d["diameters"]; phase = d["phase"]
        mask = dnm > 0.0
        if not _np.any(mask):
            self.status.set("No active pillars to render."); return

        R = (self.config["lens_diameter"] / 2.0) * 1000.0         # lens radius [µm]
        pitch_um = float(abs(x[0,1] - x[0,0])) if x.shape[1] > 1 else 1.0

        # Height model (same as estimate)
        H2pi = self._h_2pi_um()
        h_um = (phase / (2.0 * _np.pi)) * H2pi

        # --- UI window with controls ---
        top = tk.Toplevel(self.root); top.title("3D CAD (Plotly/WebGL) — tapered frustums")

        # Controls row
        row = tk.Frame(top); row.pack(side=tk.TOP, fill=tk.X)
        tk.Label(row, text="Stride:", font=("Inter", 12)).pack(side=tk.LEFT, padx=(8, 4))
        stride_var = tk.IntVar(value=6)
        tk.Scale(row, from_=1, to=40, orient="horizontal", variable=stride_var,
                showvalue=True, length=160).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(row, text="Min d (nm):", font=("Inter", 12)).pack(side=tk.LEFT)
        # auto-range from data
        ds_all = dnm[mask].astype(float)
        dmin, dmax = float(_np.nanmin(ds_all)), float(_np.nanmax(ds_all))
        default_min = float(_np.nanpercentile(ds_all, 10.0))
        mind_var = tk.DoubleVar(value=max(10.0, default_min))
        tk.Scale(row, from_=max(10.0, 0.5*dmin), to=max(100.0, 1.2*dmax),
                resolution=max(1.0, (dmax-dmin)/200), orient="horizontal",
                variable=mind_var, showvalue=True, length=200).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(row, text="Sides:", font=("Inter", 12)).pack(side=tk.LEFT)
        sides_var = tk.IntVar(value=int(self.config.get("sides", 10)))
        tk.Scale(row, from_=4, to=16, orient="horizontal", variable=sides_var,
                showvalue=True, length=120).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(row, text="Taper d₂/d₁:", font=("Inter", 12)).pack(side=tk.LEFT)
        taper_var = tk.DoubleVar(value=0.55)  # ~ look of your example
        tk.Scale(row, from_=0.2, to=1.0, resolution=0.05, orient="horizontal",
                variable=taper_var, showvalue=True, length=160).pack(side=tk.LEFT, padx=(0, 10))

        # ROI controls
        roi_row = tk.Frame(top); roi_row.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        mode_var = tk.StringVar(value="ring")
        ttk.Radiobutton(roi_row, text="Ring ROI (fast)", variable=mode_var, value="ring").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Radiobutton(roi_row, text="Full lens (slower)", variable=mode_var, value="full").pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(roi_row, text="Ring radius (×R):", font=("Inter", 12)).pack(side=tk.LEFT)
        ring_r_var = tk.DoubleVar(value=0.35)
        tk.Scale(roi_row, from_=0.05, to=0.95, resolution=0.02, orient="horizontal",
                variable=ring_r_var, showvalue=True, length=180).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(roi_row, text="Ring width (×R):", font=("Inter", 12)).pack(side=tk.LEFT)
        ring_w_var = tk.DoubleVar(value=0.30)
        tk.Scale(roi_row, from_=0.05, to=0.80, resolution=0.02, orient="horizontal",
                variable=ring_w_var, showvalue=True, length=180).pack(side=tk.LEFT, padx=(0, 6))

        # Action buttons
        btns = tk.Frame(top); btns.pack(side=tk.TOP, fill=tk.X, pady=(6, 6))
        tk.Button(btns, text="Render", bg="#d97706", fg="white",
                font=("Inter", 12, "bold"),
                command=lambda: render()).pack(side=tk.LEFT, padx=8)
        status_lbl = tk.Label(btns, text="", font=("Inter", 12)); status_lbl.pack(side=tk.LEFT, padx=8)

        # --------- mesh builder ----------
        def build_frustums(cx, cy, hh, d_nm, sides, taper):
            """Return concatenated Mesh3d vertex and triangle arrays for many pillars."""
            Vx = []; Vy = []; Vz = []; I = []; J = []; K = []
            vert_ofs = 0
            # base/top radii in µm
            rb = 0.5 * (d_nm * 1e-3)            # nm -> µm
            rt = rb * float(taper)              # top smaller = tapered
            s = int(max(3, sides))
            # angles for polygon
            ang = _np.linspace(0, 2*_np.pi, s, endpoint=False)
            ca, sa = _np.cos(ang), _np.sin(ang)

            for Xc, Yc, H, RB, RT in zip(cx, cy, hh, rb, rt):
                # vertices: bottom ring then top ring
                xb = Xc + RB * ca; yb = Yc + RB * sa; zb = _np.zeros_like(xb)
                xt = Xc + RT * ca; yt = Yc + RT * sa; zt = _np.full_like(xt, H)
                Vx.extend(xb); Vy.extend(yb); Vz.extend(zb)
                Vx.extend(xt); Vy.extend(yt); Vz.extend(zt)
                # side faces (two triangles per side)
                for k in range(s):
                    v0 = vert_ofs + k
                    v1 = vert_ofs + ((k+1) % s)
                    v2 = vert_ofs + s + k
                    v3 = vert_ofs + s + ((k+1) % s)
                    # (v0, v1, v2) and (v1, v3, v2)
                    I += [v0, v1]; J += [v1, v3]; K += [v2, v2]
                vert_ofs += 2 * s

            return _np.array(Vx), _np.array(Vy), _np.array(Vz), _np.array(I), _np.array(J), _np.array(K)

        # --------- data selection ----------
        def select_points():
            """
            Deterministic grid decimation that preserves ring structure.
            We start with the user stride s0, then (if too many pillars)
            increase stride uniformly in X and Y instead of random sampling.
            """
            s0 = int(max(1, stride_var.get()))
            xs = x[::s0, ::s0]; ys = y[::s0, ::s0]; ds = dnm[::s0, ::s0]; hs = h_um[::s0, ::s0]

            # initial mask: in-lens + diameter threshold
            rr = _np.sqrt(xs**2 + ys**2)
            mk = (ds >= float(mind_var.get())) & (rr <= R)

            # optional ring-ROI
            if mode_var.get() == "ring":
                rc = float(ring_r_var.get()) * R  # center radius
                w  = float(ring_w_var.get()) * R  # width
                mk &= (rr >= (rc - 0.5*w)) & (rr <= (rc + 0.5*w))

            # budget: either pillar count or triangles (2*sides per pillar)
            MAX_TRIANGLES = 400_000
            sides_here = int(max(3, sides_var.get()))
            max_by_tri = max(1000, MAX_TRIANGLES // (2 * sides_here))
            MAX_PILLARS = min(12000, max_by_tri)

            n = int(_np.count_nonzero(mk))
            if n > MAX_PILLARS:
                # increase stride uniformly so spatial pattern stays intact
                extra = int(_np.ceil(_np.sqrt(n / MAX_PILLARS)))
                I, J = _np.indices(mk.shape)
                mk &= (I % extra == 0) & (J % extra == 0)
                n = int(_np.count_nonzero(mk))

            xs, ys, ds, hs = xs[mk], ys[mk], ds[mk], hs[mk]
            return xs, ys, ds.astype(float), hs.astype(float), n

        # --------- render ----------
        def render():
            status_lbl.config(text="Rendering…"); top.update_idletasks()
            xs, ys, ds, hs, n = select_points()
            if n == 0:
                status_lbl.config(text="Nothing to render (relax filters or widen ring)."); return

            Vx, Vy, Vz, I, J, K = build_frustums(
                xs, ys, hs, ds,
                sides=int(sides_var.get()),
                taper=float(taper_var.get())
            )

            mesh = go.Mesh3d(
                x=Vx, y=Vy, z=Vz, i=I, j=J, k=K,
                color="goldenrod", opacity=1.0,
                lighting=dict(ambient=0.4, diffuse=0.7, specular=0.9, roughness=0.15, fresnel=0.2),
                lightposition=dict(x=0, y=0, z=2000)
            )

            # Thin substrate disk for context (wireframe ring)
            th = _np.linspace(0, 2*_np.pi, 400)
            sub = go.Scatter3d(x=R*_np.cos(th), y=R*_np.sin(th), z=_np.zeros_like(th),
                            mode="lines", line=dict(color="rgba(255,255,255,0.6)", width=2), showlegend=False)

            fig = go.Figure([mesh, sub])
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
                title=f"Tapered frustums — pillars: {xs.size:,}  (stride {int(max(1,stride_var.get()))})"
            )

            # Start camera aimed like your example
            fig.update_layout(scene_camera=dict(eye=dict(x=1.6, y=1.2, z=0.9)))

            # Write to a temp HTML and open
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            tmp.close()
            plotly_plot(fig, filename=tmp.name, auto_open=True)
            status_lbl.config(text=f"Opened in browser (pillars: {xs.size:,}).")

        # initial render
        render()
    # ---------- Estimate writing time ----------
    def _estimate_time(self):
        if not self._ensure_layout(): return
        d = self.current_layout_data
        neff = RESINS[self.config["resin"]]["n_eff"]
        shape = self.config.get("shape", "cylinder")

        # Height model (heuristic): frustum ~10 µm @ n≈1.5; cylinder ~half; scale ~1/n and with perf%
        h_ref = 10.0 if shape == "frustum" else 5.0
        h_um = h_ref * (1.5 / neff) * max(0.6, min(1.2, self.config["perf_pct"] / 80.0))

        d_nm = d["diameters"]; mask = d_nm > 0
        area_pillars_um2 = (np.pi / 4.0) * np.sum((d_nm[mask] ** 2)) * 1e-6

        hatch_um = 0.5; speed_ums = 5000; layer_thickness_um = 0.5
        layers = max(1, int(np.ceil(h_um / layer_thickness_um)))
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

        h, w = dnm.shape; count = 0; shape = self.config.get("shape", "cylinder")
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

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PDK_App(root)
    root.mainloop()
