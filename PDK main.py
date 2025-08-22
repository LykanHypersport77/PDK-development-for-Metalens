import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- 1. PDK CORE: FABRICATION PROCESS SIMULATION ---

def generate_luts_for_process(process_id):
    """
    Generates mock lookup tables (LUTs) based on the selected fabrication process.
    Each process has different capabilities for feature size and phase coverage.
    """
    diameters_nm, phase_shift_rad, fab_parameter = None, None, None
    fab_param_name = "E-Beam Dose (a.u.)"

    if process_id == 'Process A: High-Res E-Beam':
        diameters_nm = np.linspace(50, 200, 100)
        phase_shift_rad = np.mod(2.1 * np.pi * (1 - np.cos(np.pi * (diameters_nm - 50) / 150)), 2 * np.pi)
        fab_parameter = 100 * (1 + 0.15 * np.exp(-(diameters_nm - 50) / 40))
    elif process_id == 'Process B: Standard E-Beam':
        diameters_nm = np.linspace(80, 220, 80)
        phase_shift_rad = np.mod(1.85 * np.pi * (1 - np.cos(np.pi * (diameters_nm - 80) / 140)), 2 * np.pi)
        fab_parameter = 120 * (1 + 0.10 * np.exp(-(diameters_nm - 80) / 60))
    elif process_id == 'Process C: DUV Photolithography':
        diameters_nm = np.linspace(110, 250, 60)
        phase_shift_rad = np.mod(1.6 * np.pi * np.sin(np.pi * (diameters_nm - 110) / 280), 2 * np.pi)
        fab_parameter = 1500 + 5 * (diameters_nm - 110)
        fab_param_name = "Etch Time (ms)"

    return {
        'phase_lut': {'diameters_nm': diameters_nm, 'phases_rad': phase_shift_rad},
        'fab_lut': {'diameters_nm': diameters_nm, 'param_values': fab_parameter, 'param_name': fab_param_name},
    }

# --- 2. PDK CORE: INVERSE DESIGN & LAYOUT GENERATION ---

def create_inverse_interpolators(phase_lut, fab_lut):
    """Creates interpolator functions to find diameter from phase and fab param from diameter."""
    # Sort by phase to handle non-monotonic behavior for interpolation
    sort_indices = np.argsort(phase_lut['phases_rad'])
    phase_to_dia = interp1d(
        phase_lut['phases_rad'][sort_indices],
        phase_lut['diameters_nm'][sort_indices],
        bounds_error=False,
        fill_value=(phase_lut['diameters_nm'][sort_indices][0], phase_lut['diameters_nm'][sort_indices][-1])
    )
    dia_to_fab = interp1d(
        fab_lut['diameters_nm'],
        fab_lut['param_values'],
        bounds_error=False,
        fill_value=(fab_lut['param_values'][0], fab_lut['param_values'][-1])
    )
    return phase_to_dia, dia_to_fab

def calculate_phase_profile(config, grid_resolution):
    """Calculates the ideal hyperbolic phase profile for a focusing lens."""
    radius_um = (config['lens_diameter'] / 2.0) * 1000
    f_um = config['focal_length'] * 1000
    wavelength_um = config['wavelength']
    
    xy_lin = np.linspace(-radius_um, radius_um, grid_resolution)
    x_coords, y_coords = np.meshgrid(xy_lin, xy_lin)
    r = np.sqrt(x_coords**2 + y_coords**2)
    
    phase = (2 * np.pi / wavelength_um) * (np.sqrt(r**2 + f_um**2) - f_um)
    wrapped_phase = np.mod(phase, 2 * np.pi)
    
    return x_coords, y_coords, wrapped_phase

# --- 3. GUI APPLICATION CLASS ---

class PDK_App:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Metalens PDK")
        self.root.configure(bg='#111827') # bg-gray-900

        self.GRID_RESOLUTION = 150 # Lower for better GUI performance
        self.current_layout_data = {}

        # --- Create GUI Layout ---
        controls_frame = tk.Frame(root, bg='#1f2937', padx=15, pady=15) # bg-gray-800
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        plots_frame = tk.Frame(root, bg='#111827')
        plots_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Controls ---
        ttk.Label(controls_frame, text="Design Controls", font=("Inter", 16, "bold"), background='#1f2937', foreground='white').pack(pady=10)
        
        self.controls = {}
        options = {
            "Wavelength (λ)": {"values": [("1550 nm (Telecom)", 1.55), ("1064 nm (YAG)", 1.064), ("905 nm (LiDAR)", 0.905)], "default": 1.55},
            "Focal Length (f)": {"values": [("5 mm", 5.0), ("10 mm", 10.0), ("20 mm", 20.0)], "default": 10.0},
            "Lens Diameter": {"values": [("1 mm", 1.0), ("2 mm", 2.0), ("3 mm", 3.0)], "default": 2.0},
            "Pinhole Aperture": {"values": [("No Aperture", 9999.0), ("0.5 mm", 0.5), ("1.0 mm", 1.0)], "default": 9999.0},
            "Fabrication Process": {"values": [("Process A: High-Res E-Beam",), ("Process B: Standard E-Beam",), ("Process C: DUV Photolithography",)], "default": "Process A: High-Res E-Beam"},
        }

        style = ttk.Style()
        style.configure('TLabel', background='#1f2937', foreground='lightgray', font=('Inter', 10))
        style.configure('TMenubutton', background='#374151', foreground='white', font=('Inter', 10), borderwidth=1)

        for label, opts in options.items():
            ttk.Label(controls_frame, text=label).pack(anchor='w', pady=(10, 2))
            var = tk.StringVar(value=opts['default'])
            
            display_values = [v[0] for v in opts['values']]
            
            if len(opts['values'][0]) == 1:
                option_map = {v[0]: v[0] for v in opts['values']}
                var.set(display_values[0])
            else:
                option_map = {v[0]: v[1] for v in opts['values']}
                default_display = next(k for k, v in option_map.items() if v == opts['default'])
                var.set(default_display)

            self.controls[label] = {'var': var, 'map': option_map}
            
            dropdown = ttk.OptionMenu(controls_frame, var, var.get(), *display_values, command=lambda _: self.update_plots())
            dropdown.pack(fill=tk.X)

        # --- Action Button ---
        generate_btn = ttk.Button(controls_frame, text="Generate Layout CSV", command=self.generate_csv_file)
        generate_btn.pack(pady=20, fill=tk.X)
        self.status_label = ttk.Label(controls_frame, text="")
        self.status_label.pack()

        # --- Plots ---
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(10, 10), facecolor='#111827')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plots()

    def update_plots(self):
        """Main function to re-calculate and re-draw everything."""
        # --- FIX: Clear the entire figure to prevent stacking artifacts ---
        self.fig.clear()
        
        # 1. Get config from UI
        config = {
            'wavelength': self.controls['Wavelength (λ)']['map'][self.controls['Wavelength (λ)']['var'].get()],
            'focal_length': self.controls['Focal Length (f)']['map'][self.controls['Focal Length (f)']['var'].get()],
            'lens_diameter': self.controls['Lens Diameter']['map'][self.controls['Lens Diameter']['var'].get()],
            'pinhole_diameter': self.controls['Pinhole Aperture']['map'][self.controls['Pinhole Aperture']['var'].get()],
            'fab_process': self.controls['Fabrication Process']['var'].get()
        }

        # 2. Generate LUTs and interpolators
        luts = generate_luts_for_process(config['fab_process'])
        phase_to_dia, dia_to_fab = create_inverse_interpolators(luts['phase_lut'], luts['fab_lut'])

        # 3. Calculate phase and layout
        x, y, phase = calculate_phase_profile(config, self.GRID_RESOLUTION)
        diameters = phase_to_dia(phase)
        fab_params = dia_to_fab(diameters)

        # Apply pinhole
        r = np.sqrt(x**2 + y**2)
        pinhole_radius_um = (config['pinhole_diameter'] / 2.0) * 1000
        mask = r > pinhole_radius_um
        diameters[mask] = 0
        fab_params[mask] = 0
        
        self.current_layout_data = {'x': x, 'y': y, 'diameters': diameters, 'fab_params': fab_params, 'fab_lut': luts['fab_lut']}

        # 4. Draw plots on a fresh set of axes
        ((ax1, ax2), (ax3, ax4)) = self.fig.subplots(2, 2)
        self.draw_lut_plot(ax1, luts['phase_lut'])
        self.draw_image_plot(ax2, x, y, phase, "2. Ideal Phase Profile", 'hsv')
        self.draw_image_plot(ax3, x, y, diameters, "3. Generated Nanopillar Layout", 'viridis')
        self.draw_image_plot(ax4, x, y, fab_params, f"4. {luts['fab_lut']['param_name']}", 'plasma')
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def draw_lut_plot(self, ax, phase_lut):
        ax.plot(phase_lut['diameters_nm'], phase_lut['phases_rad'], 'o-', color='#3b82f6')
        ax.set_title("1. Meta-Atom LUT")
        ax.set_xlabel("Pillar Diameter (nm)")
        ax.set_ylabel("Phase Shift (rad)")
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.grid(True, linestyle='--', alpha=0.5)

    def draw_image_plot(self, ax, x, y, data, title, cmap):
        valid_data = data[data > 0]
        vmin = valid_data.min() if len(valid_data) > 0 else 0
        vmax = valid_data.max() if len(valid_data) > 0 else 1
        
        im = ax.imshow(data, extent=[x.min()/1000, x.max()/1000, y.min()/1000, y.max()/1000],
                       cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def generate_csv_file(self):
        """Saves the current layout data to a CSV file."""
        data = self.current_layout_data
        if not data:
            self.status_label.config(text="No data to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Metalens Layout"
        )
        if not filepath:
            return

        self.status_label.config(text="Generating CSV...")
        self.root.update_idletasks()

        header = f"x_um,y_um,diameter_nm,{data['fab_lut']['param_name'].replace(' ', '_')}"
        
        x_flat = data['x'].flatten()
        y_flat = data['y'].flatten()
        d_flat = data['diameters'].flatten()
        f_flat = data['fab_params'].flatten()

        mask = d_flat > 0
        rows = np.stack([x_flat[mask], y_flat[mask], d_flat[mask], f_flat[mask]], axis=1)
        
        np.savetxt(filepath, rows, delimiter=',', header=header, comments='', fmt='%.3f')

        self.status_label.config(text=f"Saved to {filepath.split('/')[-1]}")
        self.root.after(4000, lambda: self.status_label.config(text=""))

if __name__ == '__main__':
    root = tk.Tk()
    app = PDK_App(root)
    root.mainloop()
