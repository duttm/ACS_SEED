import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simps
import os

# Config Section
# Definitions
DEBUG = True

WATER_PLOT_RESOLUTION = 5000  # Number of points for water q-value range
TRYPSIN_PLOT_RESOLUTION = 4000  # Number of points for trypsin q-value range
WATER_FILE = "waterOO_npt_edited.xvg"  # Input file for water RDF data
TRYPSIN_FILE = "trypsin_100ns.xvg"  # Input file for trypsin RDF data
OUTPUT_PLOT = "water_trypsin_sq_detailed_adjusted.png"  # Output plot file
OUTPUT_DATA = "sq_values_adjusted.txt"  # Output data file

# Data Loading and Validation Section
# Load and process water data
print(f"Loading water data from {WATER_FILE}...")
if not os.path.exists(WATER_FILE):
    print(f"Error: {WATER_FILE} not found!")
    exit(1)
data_water = pd.read_csv(WATER_FILE, sep='\s+', comment='#', header=None)
data2_water = data_water.to_numpy()
print(f"Water data shape: {data2_water.shape}")
print("First 10 rows of water:\n", data2_water[:10])
print("Last 10 rows of water:\n", data2_water[-10:])
print("Middle 10 rows (200-209):\n", data2_water[200:210])  # Check for g(r) peaks
print("Full r range (nm):", data2_water[:, 0].min(), "to", data2_water[:, 0].max())

# Prepare water data arrays
r_nm_water = data2_water[:, 0]  # Radial distance in nanometers
gr_water = data2_water[:, 1] - 1  # Radial distribution function (g(r), shifted by -1)
r_A_water = r_nm_water * 10  # Convert radial distance to Angstroms
# Find index where g(r) becomes positive
start_idx_water = np.where(gr_water > 0)[0][0]
print(f"Initial g(r) values near start: {gr_water[start_idx_water:start_idx_water + 5]}")
r_water = r_A_water[start_idx_water:]  # Filtered radial distances
gr_water = gr_water[start_idx_water:]  # Filtered g(r) values
print(f"Filtered r_water range: {r_water[0]} to {r_water[-1]} Å")
print(f"Filtered g(r) range: {gr_water.min()} to {gr_water.max()}")

# Load and process trypsin data
print(f"\nLoading trypsin data from {TRYPSIN_FILE}...")
if not os.path.exists(TRYPSIN_FILE):
    print(f"Error: {TRYPSIN_FILE} not found!")
    exit(1)
data_trypsin = pd.read_csv(TRYPSIN_FILE, sep='\s+', comment='#', header=None)
data2_trypsin = data_trypsin.to_numpy()
print(f"Trypsin data shape: {data2_trypsin.shape}")
print("First 10 rows of trypsin:\n", data2_trypsin[:10])
print("Last 10 rows of trypsin:\n", data2_trypsin[-10:])
print("Middle 10 rows (3000-3009):\n", data2_trypsin[3000:3010])  # Check for g(r) peaks
print("Full r range (nm):", data2_trypsin[:, 0].min(), "to", data2_trypsin[:, 0].max())

# Prepare trypsin data arrays
r_nm_trypsin = data2_trypsin[:, 0]  # Radial distance in nanometers
gr_trypsin = data2_trypsin[:, 1]  # Radial distribution function (g(r))
r_A_trypsin = r_nm_trypsin * 10  # Convert radial distance to Angstroms
# Find index where g(r) exceeds 0.1 (or 0 if none)
start_idx_trypsin = np.where(gr_trypsin > 0.1)[0][0] if np.any(gr_trypsin > 0.1) else np.where(gr_trypsin > 0)[0][0]
print(f"Initial g(r) values near start: {gr_trypsin[start_idx_trypsin:start_idx_trypsin + 5]}")
r_trypsin = r_A_trypsin[start_idx_trypsin:]  # Filtered radial distances
gr_trypsin = gr_trypsin[start_idx_trypsin:]  # Filtered g(r) values
print(f"Filtered r_trypsin range: {r_trypsin[0]} to {r_trypsin[-1]} Å")
print(f"Filtered g(r) range: {gr_trypsin.min()} to {gr_trypsin.max()}")

# Physical Parameters Section
# Define constants for physical calculations
rho_water = 0.0334  # Number density of water (Å⁻³)
rho_trypsin = 0.0061  # Number density of trypsin (Å⁻³)
dr_water = r_water[1] - r_water[0]  # Step size for water data
dr_trypsin = r_trypsin[1] - r_trypsin[0]  # Step size for trypsin data
q_values_water = np.linspace(0.5, 15, WATER_PLOT_RESOLUTION)  # q-range for water (Å⁻¹)
q_values_trypsin = np.linspace(0.1, 15, TRYPSIN_PLOT_RESOLUTION)  # q-range for trypsin (Å⁻¹)
print(f"\nWater q values range: {q_values_water[0]} to {q_values_water[-1]} Å⁻¹ with {len(q_values_water)} points")
print(
    f"Trypsin q values range: {q_values_trypsin[0]} to {q_values_trypsin[-1]} Å⁻¹ with {len(q_values_trypsin)} points")


# Computation Functions Section
# Define function to compute structure factor S(q)
def compute_Sq(q, r, gr, rho, dr, label):
    """Calculate the structure factor S(q) using numerical integration.

    Args:
        q (float): Scattering wavevector (Å⁻¹)
        r (np.array): Radial distances (Å)
        gr (np.array): Radial distribution function
        rho (float): Number density (Å⁻³)
        dr (float): Step size (Å)
        label (str): Identifier for debugging

    Returns:
        float: Structure factor S(q)
    """
    if not np.all(r > 0) or not np.all(np.isfinite(gr)):
        print(f"Warning: Invalid r or gr values detected for {label}!")
        return np.nan
    integrand = r ** 2 * (gr) * np.sin(q * r) / (q * r + 1e-10)  # Avoid division by zero
    if DEBUG:
        print(f"{label} - q={q:.2f}: integrand min={integrand.min():.4e}, max={integrand.max():.4e}")
    integral = simps(integrand, x=r)  # Integrate using Simpson's rule
    if DEBUG:
        print(f"{label} - q={q:.2f}: integral={integral:.4e}")
    sq = 1 + (4 * np.pi * rho * integral)  # Compute S(q)
    if DEBUG:
        print(f"{label} - q={q:.2f}: S(q)={sq:.4f}")
    return sq


# Compute S(q) for water
print("\nStarting S(q) computation for water...")
S_q_water = np.array([compute_Sq(q, r_water, gr_water, rho_water, dr_water, "Water") for q in q_values_water])
print("Water S(q) computed:", S_q_water[:5], "...", S_q_water[-5:])

# Compute S(q) for trypsin
print("\nStarting S(q) computation for trypsin...")
if len(r_trypsin) > 2000:  # Subsample if data is large
    step = len(r_trypsin) // 500  # Reduce data points
    r_trypsin_sub = r_trypsin[::step]
    gr_trypsin_sub = gr_trypsin[::step]
    dr_trypsin_sub = r_trypsin_sub[1] - r_trypsin_sub[0]
    print(f"Subsampling trypsin: step={step}, new length={len(r_trypsin_sub)}")
    S_q_trypsin = np.array(
        [compute_Sq(q, r_trypsin_sub, gr_trypsin_sub, rho_trypsin, dr_trypsin_sub, "Trypsin") for q in
         q_values_trypsin])
else:
    S_q_trypsin = np.array(
        [compute_Sq(q, r_trypsin, gr_trypsin, rho_trypsin, dr_trypsin, "Trypsin") for q in q_values_trypsin])
print("Trypsin S(q) computed:", S_q_trypsin[:5], "...", S_q_trypsin[-5:])

# Data Output Section
# Save computed S(q) values to a file
with open(OUTPUT_DATA, 'w') as f:
    f.write("q_values_water,Water_Sq,q_values_trypsin,Trypsin_Sq\n")
    for qw, sw, qt, st in zip(q_values_water, S_q_water, q_values_trypsin, S_q_trypsin):
        f.write(f"{qw},{sw},{qt},{st}\n")
print(f"S(q) values saved to {OUTPUT_DATA}")


# Smoothing Function Section
# Define function for moving average smoothing
def moving_average(data, window_size):
    """Compute moving average with a given window size.

    Args:
        data (np.array): Input data to smooth
        window_size (int): Size of the moving window

    Returns:
        np.array: Smoothed data
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


# Smoothing Configuration
# Adjust MOVING_AVERAGE_WINDOW to change the smoothing level for Trypsin S(q) plots.
# A smaller value (e.g., 5) keeps more detail, while a larger value (e.g., 15) smooths more.
# Experiment with values like 10, 20, or 30 to reduce noise at low q (0.1–1.5 Å⁻¹).
MOVING_AVERAGE_WINDOW = 49  # Large window for significant smoothing

# Apply moving average to trypsin S(q)
S_q_trypsin_smooth = moving_average(S_q_trypsin, MOVING_AVERAGE_WINDOW)
# Adjust q_values_trypsin to match the shortened S_q_trypsin_smooth length
q_values_trypsin_smooth = q_values_trypsin[:len(S_q_trypsin_smooth)]

# Plotting Section
# Create figure for visualization
plt.figure(figsize=(15, 10))

# Water S(q) - Main plot
plt.subplot(2, 2, 1)
plt.plot(q_values_water, S_q_water, label='Water S(q)', color='green', linewidth=2)
plt.xlabel('q (Å⁻¹)')
plt.ylabel('S(q)')
plt.title('Water Structure Factor - Full Range (1.5-10 Å⁻¹)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0.5, 15)

# Trypsin S(q) - Main plot with experimental data
plt.subplot(2, 2, 3)
plt.plot(q_values_trypsin[100:-50], S_q_trypsin_smooth[100:-50],
         label=f'Trypsin S(q) Smoothed (Window={MOVING_AVERAGE_WINDOW})', color='red', linewidth=2)
data = np.loadtxt('pdfgetx2_Trypsin', delimiter=None, dtype=float, unpack=True)
q = data[0]
sq = data[1]
plt.plot(q, sq, label='S(q) Experimental', color='blue', linewidth=2)  # Corrected typo in label
plt.xlabel('q (Å⁻¹)')
plt.ylabel('S(q)')
plt.title('Trypsin Structure Factor - Full Range (0.1-15 Å⁻¹)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0.5, 15)

# Finalize and save plot
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"Plot saved to {OUTPUT_PLOT}")
plt.show()