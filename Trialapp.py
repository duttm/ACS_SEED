import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

#  Load RDF Data


def load_rdf(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith(('#', '@')):
                parts = line.strip().split()
                if len(parts) >= 2:
                    data.append([float(parts[0]), float(parts[1])])
    return np.array(data)

#  Compute S(q) from RDF


def compute_structure_factor(r, g_r, q_values, rho):
    S_q = []
    delta_r = r[1] - r[0]

    for q in q_values:
        qr = q * r
        sinc_term = np.ones_like(qr)
        nonzero = qr != 0
        sinc_term[nonzero] = np.sin(qr[nonzero]) / qr[nonzero]
        integrand = r**2 * g_r * sinc_term
        integral = simpson(integrand, r)
        S = 1 + 4 * np.pi * rho * integral
        S_q.append(S)
    return np.array(S_q)

#  Moving Average Function


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


#  Parameters
rho = 0.0061  # number density for tryspin
rho_water = 0.0334  # number density for water
q_values = np.linspace(0.5, 15, 10000)  # Water q-range

#  Load and compute Tryspin S(q)
rdf_data = load_rdf("/Users/juliethalaseh/Python SEED/trypsin_100ns (1).xvg")
r = rdf_data[:, 0] * 10  # Convert from nm → Å
g_r = rdf_data[:, 1]
trypsin_sq = compute_structure_factor(r, g_r, q_values, rho)
window = 199
sq_trypsin_smooth = moving_average(trypsin_sq, window)

#  Load and compute Water S(q)
rdf_data = load_rdf(
    "/Users/juliethalaseh/Python SEED/waterOO_npt_edited (1).xvg")
r = rdf_data[:, 0] * 10  # Convert from nm → Å
g_r = rdf_data[:, 1] - 1
water_sq = compute_structure_factor(r, g_r, q_values, rho_water)

# Load Trypsin S(q) Experimental Data
data = np.loadtxt('pdfgetx2_Trypsin', unpack=True)
q_trypsin_raw = data[0]
sq_trypsin_raw = data[1]

# Plot All 4 Graphs in Subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

#  Trypsin Full Range (Top Left)
axs[0, 0].plot(q_values[100:], trypsin_sq[100:], color='green',
               linewidth=2, label='Tryspin S(q)')
axs[0, 0].set_title("Tryspin Structure Factor - Full Range (0.5-15 Å⁻¹)")
axs[0, 0].set_xlabel("q (Å⁻¹)")
axs[0, 0].set_ylabel("S(q)")
axs[0, 0].legend()
axs[0, 0].grid(True)
#  Water Full Range (Top Right)
axs[0, 1].plot(q_values, water_sq, color='purple',
               linewidth=2, label='Water S(q)')
axs[0, 1].set_title("Water Structure Factor - Full Range (0.5-15 Å⁻¹)")
axs[0, 1].set_xlabel("q (Å⁻¹)")
axs[0, 1].set_ylabel("S(q)")
axs[0, 1].legend()
axs[0, 1].grid(True)


# Trypsin Full Range (Bottom Left)
axs[1, 0].plot(q_values, sq_trypsin_smooth, color='red', linewidth=2,
               label=f'Trypsin S(q) Smoothed (Window={window})')
axs[1, 0].plot(q_trypsin_raw, sq_trypsin_raw, color='blue', linewidth=2,
               label=f'Trypsin S(q) Experimental (Window={window})')
axs[1, 0].set_xlim(0.1, 15)
axs[1, 0].set_title(
    "Overlay of Experimental vs Computational Data(0.1-15 Å⁻¹)")
axs[1, 0].set_xlabel("q (Å⁻¹)")
axs[1, 0].set_ylabel("S(q)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Trypsin Zoomed (Bottom Right)
axs[1, 1].plot(q_values, sq_trypsin_smooth, color='red', linewidth=2,
               label=f'Trypsin S(q) Smoothed (Window={window})')
axs[1, 1].set_xlim(0.1, 1)
axs[1, 1].set_ylim(-5, 20)
axs[1, 1].set_title("Trypsin Structure Factor - Zoomed (0.1-1 Å⁻¹)")
axs[1, 1].set_xlabel("q (Å⁻¹)")
axs[1, 1].set_ylabel("S(q)")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
