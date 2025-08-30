import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

# ----------------------------- Mock Materials / Resins -----------------------------
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

# ----------------------------- Generic (process-free) LUTs -----------------------------

def generate_generic_luts(substrate, resin, wavelength_um):
    """Process-agnostic LUTs; slight scaling with n_eff to stay realistic-ish."""
    neff = RESINS[resin]["n_eff"]
    diameters_nm = np.linspace(110, 250, 80)
    amp = 1.5 + 0.4 * (neff - 1.6)  # around ~1.5..1.8
    phases = np.mod(amp * np.pi * np.sin(np.pi * (diameters_nm - 110) / 280), 2 * np.pi)
    fab_param = 1000 + 4 * (diameters_nm - 110)  # generic fabrication proxy
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

def calculate_phase_profile(config, grid_resolution):
    radius_um = (config["lens_diameter"] / 2.0) * 1000.0
    f_um = config["working_distance"] * 1000.0
    wavelength_um = config["wavelength_um"]
    xy_lin = np.linspace(-radius_um, radius_um, grid_resolution)
    x, y = np.meshgrid(xy_lin, xy_lin)
    r = np.sqrt(x**2 + y**2)
    phase = (2 * np.pi / wavelength_um) * (np.sqrt(r**2 + f_um**2) - f_um)
    return x, y, np.mod(phase, 2 * np.pi)

# ----------------------------- App -----------------------------
class PDK_App:
    def __init__(self, root):
        self.root = root
        self.root.title("Metalens PDK — Guided Flow (Process-Free)")
        self.root.configure(bg="#111827")

        self.PX_PER_MM = 75
        self.MIN_RES = 151
        self.MAX_RES = 1201

        self.config = {}
        self.current_layout_data = {}

        # Left / right panes
        self.left = tk.Frame(root, bg="#1f2937", padx=14, pady=14)
        self.left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.right = tk.Frame(root, bg="#111827")
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(self.left, text="GUI (step-by-step)", font=("Inter", 16, "bold")).pack(anchor="w", pady=(0, 10))

        self.info = tk.StringVar(value="")
        ttk.Label(self.left, textvariable=self.info, foreground="#c7d2fe", wraplength=360).pack(anchor="w", pady=(0, 8))
        self.warn = tk.StringVar(value="")
        ttk.Label(self.left, textvariable=self.warn, foreground="#fca5a5", wraplength=360).pack(anchor="w")

        self.steps_container = tk.Frame(self.left, bg="#1f2937")
        self.steps_container.pack(fill=tk.X)

        # Output area (persistent buttons)
        self.output_container = tk.Frame(self.left, bg="#1f2937")
        self.output_container.pack(fill=tk.X, pady=(12, 0))
        tk.Label(self.output_container, text="Output", bg="#1f2937", fg="#ffffff",
                 font=("Inter", 12, "bold")).pack(anchor="w", pady=(0, 6))

        self.btn_generate = tk.Button(
            self.output_container, text="Generate Images",
            bg="#2563eb", fg="white", activebackground="#1d4ed8", activeforeground="white",
            font=("Inter", 11, "bold"), relief="raised", command=self.update_plots, state=tk.DISABLED
        ); self.btn_generate.pack(fill=tk.X, pady=(2, 6))

        # NEW: vector (pillar-circle) viewer like KLayout
        self.btn_viewer = tk.Button(
            self.output_container, text="Open Interactive Lens Layout (vector)",
            bg="#9333ea", fg="white", activebackground="#7e22ce", activeforeground="white",
            font=("Inter", 11, "bold"), relief="raised", command=self._open_vector_layout_viewer, state=tk.DISABLED
        ); self.btn_viewer.pack(fill=tk.X, pady=(0, 6))

        self.btn_export_gds = tk.Button(
            self.output_container, text="Export GDSII (KLayout)",
            bg="#ea580c", fg="white", activebackground="#c2410c", activeforeground="white",
            font=("Inter", 11, "bold"), relief="raised", command=self.export_gdsii, state=tk.DISABLED
        ); self.btn_export_gds.pack(fill=tk.X, pady=(0, 6))

        self.btn_export = tk.Button(
            self.output_container, text="Export CSV",
            bg="#059669", fg="white", activebackground="#047857", activeforeground="white",
            font=("Inter", 11, "bold"), relief="raised", command=self.generate_csv_file, state=tk.DISABLED
        ); self.btn_export.pack(fill=tk.X, pady=(0, 6))

        self.btn_estimate = tk.Button(
            self.output_container,
            text="Estimate writing time (realistic defaults; no fab info)",
            bg="#6b7280", fg="white", activebackground="#4b5563", activeforeground="white",
            font=("Inter", 11, "bold"), relief="raised", command=self._estimate_time, state=tk.DISABLED
        ); self.btn_estimate.pack(fill=tk.X)

        self.status = tk.StringVar(value="")
        tk.Label(self.output_container, textvariable=self.status, bg="#1f2937", fg="#e5e7eb",
                 wraplength=360).pack(anchor="w", pady=(8, 0))

        # Plot canvas
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(10, 10), facecolor="#111827")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ttk style
        style = ttk.Style()
        style.configure("TLabel", background="#1f2937", foreground="white", font=("Inter", 10))
        style.configure("TMenubutton", background="#374151", foreground="white", font=("Inter", 10))

        self.step_frames = []
        self._build_step_1()
        self._update_output_buttons_state()

    # ---------- utilities ----------
    def _update_output_buttons_state(self):
        req = {"wavelength_um","substrate","resin","lens_diameter","working_distance","perf_pct"}
        ready = req <= self.config.keys()
        state = tk.NORMAL if ready else tk.DISABLED
        self.btn_generate.config(state=state)
        self.btn_export.config(state=state)
        self.btn_estimate.config(state=state)
        self.btn_viewer.config(state=state)
        self.btn_export_gds.config(state=state)

    def _clear_from_step(self, idx):
        for i in range(idx, len(self.step_frames)):
            f = self.step_frames[i]
            if f.winfo_exists():
                f.destroy()
        self.step_frames = self.step_frames[:idx]
        keys_by_step = [
            ["wavelength_um"],
            ["substrate"],
            ["resin"],
            ["lens_diameter"],
            ["working_distance"],
            ["perf_pct"],
        ]
        for klist in keys_by_step[idx:]:
            for k in klist:
                self.config.pop(k, None)
        self._update_output_buttons_state()

    def _add_step_frame(self, label_text):
        frame = tk.Frame(self.steps_container, bg="#1f2937")
        frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(frame, text=label_text).pack(anchor="w")
        self.step_frames.append(frame)
        return frame

    # ---------- steps ----------
    def _build_step_1(self):
        f = self._add_step_frame("1) Define optical specifications — Wavelength λ")
        var = tk.StringVar()
        choices = [("λ = 9 μm", 9.0), ("λ = 10 μm", 10.0)]
        var.set(choices[0][0])
        optmap = {d: v for d, v in choices}
        ttk.OptionMenu(f, var, var.get(), *optmap.keys(),
                       command=lambda _: self._on_step_1(var, optmap)).pack(fill=tk.X)

    def _on_step_1(self, var, optmap):
        self._clear_from_step(1)
        self.config["wavelength_um"] = optmap[var.get()]
        self.info.set(f"λ = {self.config['wavelength_um']} μm selected.")
        self._build_step_2()
        self._update_output_buttons_state()

    def _build_step_2(self):
        f = self._add_step_frame("2) Select substrate / material")
        var = tk.StringVar(value="Si")
        ttk.OptionMenu(f, var, var.get(), *MATERIALS.keys(),
                       command=lambda _: self._on_step_2(var)).pack(fill=tk.X)

    def _on_step_2(self, var):
        self._clear_from_step(2)
        sub = var.get()
        self.config["substrate"] = sub
        rng = MATERIALS[sub]["lambda_range_um"]; n = MATERIALS[sub]["n"]
        self.info.set(f"{sub}: n≈{n:.2f}, usable λ range ≈ {rng[0]}–{rng[1]} μm.")
        self._build_step_3()
        self._update_output_buttons_state()

    def _build_step_3(self):
        f = self._add_step_frame("3) Select resin + nanoparticle")
        var = tk.StringVar(value=list(RESINS.keys())[0])
        ttk.OptionMenu(f, var, var.get(), *RESINS.keys(),
                       command=lambda _: self._on_step_3(var)).pack(fill=tk.X)

    def _on_step_3(self, var):
        self._clear_from_step(3)
        self.config["resin"] = var.get()
        ne = RESINS[self.config["resin"]]["n_eff"]
        self.info.set(f"Composite index n_eff≈{ne:.2f}.")
        self._build_step_4()
        self._update_output_buttons_state()

    def _build_step_4(self):
        f = self._add_step_frame("4) Lens aperture (diameter)")
        var = tk.StringVar(value="1.5 mm")
        options = ["1.5 mm", "10 mm", "15 mm", "20 mm", "25 mm"]
        ttk.OptionMenu(f, var, var.get(), *options,
                       command=lambda _: self._on_step_4(var)).pack(fill=tk.X)

    def _on_step_4(self, var):
        self._clear_from_step(4)
        self.config["lens_diameter"] = float(var.get().split()[0])
        self._build_step_5()
        self._update_output_buttons_state()

    def _build_step_5(self):
        f = self._add_step_frame("5) Working distance (≈ focal length)")
        var = tk.StringVar(value="25 mm")
        options = ["5 mm", "25 mm", "50 mm", "100 mm", "200 mm"]
        ttk.OptionMenu(f, var, var.get(), *options,
                       command=lambda _: self._on_step_5(var)).pack(fill=tk.X)

    def _on_step_5(self, var):
        self._clear_from_step(5)
        self.config["working_distance"] = float(var.get().split()[0])
        D = self.config["lens_diameter"]; f = self.config["working_distance"]
        na = D / (2.0 * f)
        msg = f"Computed NA ≈ {na:.3f}. "
        if na < 0.01: self.warn.set(msg + "Warning: NA < 0.01 → very weak focusing.")
        elif na > 0.90: self.warn.set(msg + "Warning: NA > 0.9 → performance likely degraded.")
        else: self.warn.set(msg + "OK.")
        self._build_step_6()
        self._update_output_buttons_state()

    def _build_step_6(self):
        f = self._add_step_frame("6) Expected beam focusing performance (% of diffraction limit)")
        s = tk.Scale(f, from_=20, to=100, orient="horizontal", showvalue=True,
                     resolution=1, length=300, command=lambda _=None: None)
        s.set(80)
        s.pack(fill=tk.X)
        s.bind("<ButtonRelease-1>", lambda e, sv=s: self._on_step_6(sv.get()))

    def _on_step_6(self, value):
        self._clear_from_step(6)
        self.config["perf_pct"] = float(value)
        self.info.set(f"Target: {self.config['perf_pct']:.0f}% of diffraction limit.")
        self._update_output_buttons_state()

    # ---------- plotting (static 2x2 inside main window) ----------
    def update_plots(self):
        req = {"wavelength_um","substrate","resin","lens_diameter","working_distance","perf_pct"}
        if not req <= self.config.keys():
            self.warn.set("Complete steps 1–6 first.")
            return

        self.fig.clear()
        grid_res = int(np.clip(self.config["lens_diameter"] * self.PX_PER_MM, self.MIN_RES, self.MAX_RES))
        if grid_res % 2 == 0: grid_res += 1

        luts = generate_generic_luts(self.config["substrate"], self.config["resin"], self.config["wavelength_um"])
        phase_to_dia, dia_to_fab = create_inverse_interpolators(luts["phase_lut"], luts["fab_lut"])

        x, y, phase = calculate_phase_profile(self.config, grid_res)
        diameters = phase_to_dia(phase)
        fab_params = dia_to_fab(diameters)

        r = np.sqrt(x**2 + y**2)
        lens_radius_um = (self.config["lens_diameter"] / 2) * 1000.0
        active = r <= lens_radius_um
        diameters = np.where(active, diameters, 0.0)
        fab_params = np.where(active, fab_params, 0.0)

        self.current_layout_data = {"x": x, "y": y, "diameters": diameters,
                                    "fab_params": fab_params, "fab_lut": luts["fab_lut"]}

        ((ax1, ax2), (ax3, ax4)) = self.fig.subplots(2, 2)
        self._draw_lut(ax1, luts["phase_lut"])
        self._draw_img(ax2, x, y, phase, "2. Ideal Phase Profile", "hsv")
        self._draw_img(ax3, x, y, diameters, "3. Nanopillar Layout (nm)", "viridis")
        self._draw_img(ax4, x, y, fab_params, f"4. {luts['fab_lut']['param_name']}", "plasma")
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
        self.status.set("Design generated.")
        self._update_output_buttons_state()

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
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_aspect("equal", adjustable="box")
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---------- Interactive vector viewer (KLayout-like circles) ----------
    def _open_vector_layout_viewer(self):
        data = self.current_layout_data
        if not data:
            self.status.set("Generate images first.")
            return

        import numpy as np
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection

        top = tk.Toplevel(self.root)
        top.title("Interactive Lens Layout — Vector Pillars (μm units; view-culling)")

        # --- Controls ---
        controls = tk.Frame(top)
        controls.pack(side=tk.TOP, fill=tk.X)
        tk.Label(controls, text="Decimation (stride):").pack(side=tk.LEFT, padx=(8, 4))
        stride_var = tk.IntVar(value=1)   # dense by default so it's not empty
        tk.Scale(controls, from_=1, to=20, orient="horizontal", variable=stride_var,
                showvalue=True, length=220).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(controls, text="Min diameter (nm):").pack(side=tk.LEFT)
        mind_var = tk.IntVar(value=20)    # permissive default
        tk.Scale(controls, from_=10, to=250, orient="horizontal", variable=mind_var,
                showvalue=True, length=220).pack(side=tk.LEFT, padx=(0, 12))

        count_lbl = tk.Label(controls, text="")
        count_lbl.pack(side=tk.RIGHT, padx=8)

        # --- Figure + toolbar ---
        fig = Figure(figsize=(8.8, 8.8), facecolor="#111827")
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#111827")
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
        ax.set_title("Vector Pillar Layout (view-culling)")

        # Pre-pull arrays
        x = data["x"]; y = data["y"]; dnm = data["diameters"]
        mask = dnm > 0.0
        R = (self.config["lens_diameter"] / 2) * 1000.0  # lens radius in μm

        # Debounced redraw
        after_id = None
        def schedule_redraw(*_):
            nonlocal after_id
            if after_id is not None:
                try:
                    top.after_cancel(after_id)
                except Exception:
                    pass
            after_id = top.after(60, draw_visible)

        def draw_visible():
            # Current viewport
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            s = max(1, int(stride_var.get()))
            min_nm = max(1, int(mind_var.get()))

            # Decimate first, then cull to view & min diameter
            xs = x[::s, ::s]; ys = y[::s, ::s]; ds = dnm[::s, ::s]; mk = mask[::s, ::s]
            in_view = mk & (ds >= min_nm) & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)

            ax.clear()
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor("#111827")
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Y (μm)")
            ax.set_title("Vector Pillar Layout (view-culling)")
            # Keep the same viewport the user is looking at
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

            # Lens outline (cheap)
            outline = Circle((0.0, 0.0), R, fill=False, lw=1.0, ec="white", alpha=0.7)
            ax.add_patch(outline)

            count = 0
            if np.any(in_view):
                xs_v = xs[in_view]; ys_v = ys[in_view]; ds_v = ds[in_view]
                radii_um = 0.5 * ds_v * 1e-3  # nm -> μm
                circles = [Circle((float(cx), float(cy)), float(r)) for cx, cy, r in zip(xs_v, ys_v, radii_um)]
                coll = PatchCollection(circles, facecolor="#ff5470", edgecolor="none",
                                    alpha=0.85, antialiased=False)
                ax.add_collection(coll)
                count = len(circles)
            else:
                ax.text(0.5, 0.5, "No pillars in view\n(zoom/pan or lower min d)",
                        color="w", ha="center", va="center", transform=ax.transAxes, alpha=0.8)

            count_lbl.config(text=f"Pillars shown: {count:,}  (stride {s}, min d {min_nm} nm)")
            canvas.draw_idle()

        # React to any limit change (pan/zoom) and mouse/scroll actions
        ax.callbacks.connect('xlim_changed', lambda _ax: schedule_redraw())
        ax.callbacks.connect('ylim_changed', lambda _ax: schedule_redraw())
        canvas.mpl_connect('scroll_event', schedule_redraw)
        canvas.mpl_connect('button_release_event', schedule_redraw)

        # Initial zoom: start tight so draw cost is tiny (~10% of radius)
        f = 0.10
        ax.set_xlim(-R * f, R * f)
        ax.set_ylim(-R * f, R * f)

        # First paint
        draw_visible()

        # Live updates from sliders
        stride_var.trace_add("write", schedule_redraw)
        mind_var.trace_add("write", schedule_redraw)

    # ---------- Estimate writing time ----------
    def _estimate_time(self):
        req = {"lens_diameter","perf_pct"}
        if not req <= self.config.keys():
            self.status.set("Set lens diameter and Step 6 first.")
            return
        D = self.config["lens_diameter"]          # mm
        area_mm2 = np.pi * (D/2)**2
        area_um2 = area_mm2 * 1e6                 # 1 mm^2 = 1e6 μm^2
        # Realistic defaults (no fab info yet)
        hatch_um = 0.5
        speed_ums = 5000
        layer_thickness_um = 0.5
        pillar_height_um = 6.0
        layers = max(1, int(np.ceil(pillar_height_um / layer_thickness_um)))
        fill = min(1.0, max(0.2, self.config["perf_pct"]/100.0))
        time_s = fill * layers * (area_um2 / (speed_ums * hatch_um))
        hrs = time_s / 3600.0
        self.status.set(f"Estimated writing time ≈ {hrs:.2f} h (uses realistic defaults; no fab info).")

    # ---------- CSV ----------
    def generate_csv_file(self):
        data = self.current_layout_data
        if not data:
            self.status.set("No data to save. Generate images first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Metalens Layout"
        )
        if not path:
            return
        header = f"x_um,y_um,diameter_nm,{data['fab_lut']['param_name'].replace(' ', '_')}"
        x_flat = data["x"].flatten()
        y_flat = data["y"].flatten()
        d_flat = data["diameters"].flatten()
        f_flat = data["fab_params"].flatten()
        mask = d_flat > 0
        rows = np.stack([x_flat[mask], y_flat[mask], d_flat[mask], f_flat[mask]], axis=1)
        np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.3f")
        self.status.set(f"Saved: {path.split('/')[-1]}")

    # ---------- GDSII export for KLayout ----------
    def export_gdsii(self):
        data = self.current_layout_data
        if not data:
            self.status.set("No layout yet. Generate images first.")
            return
        try:
            import gdstk  # modern gdspy successor
        except Exception:
            self.status.set("GDS export requires 'gdstk'. Install with: pip install gdstk")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".gds",
            filetypes=[("GDSII files", "*.gds"), ("All files", "*.*")],
            title="Save GDSII (KLayout)"
        )
        if not path:
            return

        x = data["x"]; y = data["y"]; dnm = data["diameters"]
        mask = dnm > 0.0
        total = int(mask.sum())
        if total == 0:
            self.status.set("Nothing to write: empty layout.")
            return

        # Auto-decimate to keep shape count manageable
        target = 150_000
        stride = max(1, int(np.ceil(np.sqrt(total / target))))
        min_d_nm = 20.0  # ignore tiny features
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        cell = lib.new_cell("METALENS")

        if x.shape[1] > 1:
            pitch_um = float(abs(x[0,1] - x[0,0]))
        else:
            pitch_um = 1.0

        h, w = dnm.shape
        count = 0
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                if not mask[i, j]:
                    continue
                d_nm = float(dnm[i, j])
                if d_nm < min_d_nm:
                    continue
                r_um = 0.5 * d_nm * 1e-3  # nm -> μm
                cx = float(x[i, j]); cy = float(y[i, j])
                circ = gdstk.ellipse((cx, cy), r_um, tolerance=0.02, layer=1, datatype=0)
                cell.add(circ)
                count += 1

        lens_radius_um = (self.config["lens_diameter"] / 2) * 1000.0
        outline = gdstk.ellipse((0.0, 0.0), lens_radius_um, tolerance=0.1, layer=10, datatype=0)
        cell.add(outline)

        try:
            lib.write_gds(path)
        except Exception as e:
            self.status.set(f"GDS write failed: {e}")
            return

        self.status.set(
            f"GDS saved: {path.split('/')[-1]}  |  pillars: ~{count:,}  |  pitch≈{pitch_um:.3f} μm  |  stride={stride}"
        )

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PDK_App(root)
    root.mainloop()
