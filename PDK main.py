import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import gdspy # The library for creating GDSII files

# --- Phase 1: Foundational Physics & Process Simulation ---

def generate_mock_luts():
    """
    Generates mock lookup tables (LUTs) for both phase and a fabrication parameter.
    In a real PDK, this comes from extensive, validated simulations.

    Returns:
        tuple: A tuple containing dictionaries for the phase and dose LUTs.
    """
    # LUT 1: Pillar Diameter to Phase Shift
    pillar_diameters_nm = np.linspace(60, 200, 100)
    phase_shift_rad = np.mod(
        (2 * np.pi * (1 - np.cos(np.pi * (pillar_diameters_nm - 60) / 140))), 
        2 * np.pi
    )
    phase_lut = {'diameters_nm': pillar_diameters_nm, 'phases_rad': phase_shift_rad}

    # LUT 2: Pillar Diameter to E-Beam Dose (Manufacturing Parameter)
    # This simulates a correction for proximity effects where smaller, denser
    # features might need a slightly higher dose to resolve properly.
    # The dose is a relative value here.
    base_dose = 100
    dose_correction = 1 + 0.15 * np.exp(-(pillar_diameters_nm - 60) / 50)
    ebeam_dose = base_dose * dose_correction
    dose_lut = {'diameters_nm': pillar_diameters_nm, 'dose': ebeam_dose}
    
    return phase_lut, dose_lut

def create_inverse_interpolators(phase_lut, dose_lut):
    """
    Creates interpolator functions for the "inverse design" step.

    Args:
        phase_lut (dict): Dictionary containing diameter and phase data.
        dose_lut (dict): Dictionary containing diameter and dose data.

    Returns:
        tuple: A tuple of two functions: (phase_to_diameter, diameter_to_dose)
    """
    # Interpolator 1: Find diameter for a given phase
    sort_indices = np.argsort(phase_lut['phases_rad'])
    sorted_phases = phase_lut['phases_rad'][sort_indices]
    sorted_diameters = phase_lut['diameters_nm'][sort_indices]
    
    phase_to_diameter_func = interp1d(
        sorted_phases, 
        sorted_diameters, 
        bounds_error=False, 
        fill_value=(sorted_diameters[0], sorted_diameters[-1])
    )
    
    # Interpolator 2: Find the required e-beam dose for a given diameter
    diameter_to_dose_func = interp1d(
        dose_lut['diameters_nm'],
        dose_lut['dose'],
        bounds_error=False,
        fill_value=(dose_lut['dose'][0], dose_lut['dose'][-1])
    )

    return phase_to_diameter_func, diameter_to_dose_func

# --- Phase 2: Optical Function Definition ---

def calculate_phase_profile(lens_type, **kwargs):
    """
    Calculates the ideal phase profile for a specified optical component.
    This modular function can be expanded with more lens types.

    Args:
        lens_type (str): The type of lens ('focusing', 'axicon', etc.).
        **kwargs: Keyword arguments specific to the lens type.

    Returns:
        tuple: (x_coords_mm, y_coords_mm, phase_profile_rad)
    """
    diameter_mm = kwargs['diameter_mm']
    grid_resolution = kwargs['grid_resolution']
    wavelength_um = kwargs['wavelength_um']
    
    radius_mm = diameter_mm / 2.0
    xy_lin = np.linspace(-radius_mm * 1000, radius_mm * 1000, grid_resolution)
    x_coords, y_coords = np.meshgrid(xy_lin, xy_lin)
    r = np.sqrt(x_coords**2 + y_coords**2)

    if lens_type == 'focusing':
        f = kwargs['focal_length_mm'] * 1000
        phase_profile = (2 * np.pi / wavelength_um) * (np.sqrt(r**2 + f**2) - f)
    # --- ADD NEW LENS TYPES HERE ---
    # elif lens_type == 'axicon':
    #     beta = kwargs['beta_degrees']
    #     phase_profile = -(2 * np.pi / wavelength_um) * np.tan(np.deg2rad(beta)) * r
    else:
        raise ValueError(f"Lens type '{lens_type}' is not supported.")

    wrapped_phase_profile = np.mod(phase_profile, 2 * np.pi)
    return x_coords / 1000, y_coords / 1000, wrapped_phase_profile

# --- Phase 3: Layout Generation & Manufacturing Interface ---

def generate_gdsii_file(filename, x_coords_mm, y_coords_mm, pillar_diameters_nm, pillar_doses):
    """
    Creates a GDSII file from the generated layout data.
    This is the manufacturing interface.

    Args:
        filename (str): The name of the output GDSII file.
        x_coords_mm (ndarray): 2D array of x-coordinates.
        y_coords_mm (ndarray): 2D array of y-coordinates.
        pillar_diameters_nm (ndarray): 2D array of pillar diameters.
        pillar_doses (ndarray): 2D array of e-beam doses.
    """
    print(f"Step 4: Generating GDSII file '{filename}'...")
    
    # GDSII library setup
    gdsii_lib = gdspy.GdsLibrary()
    main_cell = gdsii_lib.new_cell('METALENS')
    
    # GDSII layer and datatype for e-beam dose
    EBEAM_LAYER = 1
    # GDSII allows storing integer data with polygons (e.g., for dose)
    EBEAM_DATATYPE_DOSE = 1 

    # Iterate through every meta-atom in our design grid
    for i in range(pillar_diameters_nm.shape[0]):
        for j in range(pillar_diameters_nm.shape[1]):
            center_x_um = x_coords_mm[i, j] * 1000
            center_y_um = y_coords_mm[i, j] * 1000
            radius_um = (pillar_diameters_nm[i, j] / 2.0) / 1000
            dose_value = int(pillar_doses[i, j])

            # Create a circle polygon for the pillar
            pillar = gdspy.Round(
                (center_x_um, center_y_um), 
                radius_um, 
                layer=EBEAM_LAYER,
                # Here we attach the dose information directly to the polygon
                datatype=EBEAM_DATATYPE_DOSE 
            )
            # The actual dose value is stored in a separate text-based file
            # for many e-beam systems, but this shows how it can be linked.
            
            main_cell.add(pillar)

    # Save the GDSII file
    gdsii_lib.write_gds(filename)
    print("Step 4: GDSII generation complete.")


if __name__ == '__main__':
    # --- High-Level Designer Inputs ---
    LENS_DIAMETER_MM = 1.0
    WAVELENGTH_UM = 1.55
    GRID_RESOLUTION = 250 

    # P-Cell parameters for a focusing lens
    pcell_params = {
        'lens_type': 'focusing',
        'focal_length_mm': 5.0,
        'diameter_mm': LENS_DIAMETER_MM,
        'grid_resolution': GRID_RESOLUTION,
        'wavelength_um': WAVELENGTH_UM
    }
    
    print("--- Metalens PDK Prototype v2 ---")
    start_time = time.time()

    # 1. Load physics and process data
    print("Step 1: Generating mock LUTs...")
    phase_lut, dose_lut = generate_mock_luts()
    phase_to_dia, dia_to_dose = create_inverse_interpolators(phase_lut, dose_lut)
    print("Step 1: Complete.")

    # 2. Calculate the ideal optical function
    print(f"Step 2: Calculating phase profile for a '{pcell_params['lens_type']}' lens...")
    x_coords, y_coords, ideal_phase = calculate_phase_profile(**pcell_params)
    print("Step 2: Complete.")

    # 3. Map optical function to physical layout
    print("Step 3: Mapping phase to physical diameters and doses...")
    final_diameters = phase_to_dia(ideal_phase)
    final_doses = dia_to_dose(final_diameters)
    print("Step 3: Complete.")
    
    # 4. Generate the manufacturing file (GDSII)
    generate_gdsii_file(
        'metalens_layout.gds', 
        x_coords, 
        y_coords, 
        final_diameters, 
        final_doses
    )
    
    end_time = time.time()
    print(f"\nFull PDK flow for {GRID_RESOLUTION**2:,} meta-atoms took {end_time - start_time:.4f} seconds.")

    # You can now open 'metalens_layout.gds' in a GDSII viewer like KLayout.

