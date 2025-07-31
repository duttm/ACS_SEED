import os
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import pip
import numpy as np
# Compute structure factor #


def load_rdf(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith(('#', '@')):
                parts = line.strip().split()
                if len(parts) >= 2:
                    data.append([float(parts[0]), float(parts[1])])
    return np.array(data)


# Parameters #
rho = 0.0334  # number density for water (molecules/Å³)
q_values = np.linspace(0.06, 15, 2000)  # Wavenumbers q (1/Å)

# Load and prepare data#
rdf_data = load_rdf(
    filename="/Users/juliethalaseh/Python SEED/trypsin_100ns (1).xvg")
r = rdf_data[:, 0] * 10  # Convert r from nm → Å
g_r = rdf_data[:, 1]

# Function to compute structure factor#


def compute_structure_factor(r, g_r, q_values, rho):
    S_q = []
    delta_r = r[1] - r[0]
    gr_minus_1 = g_r - 1

    for q in q_values:
        qr = q * r
        sinc_term = np.ones_like(qr)

        nonzero = qr != 0
        sinc_term[nonzero] = np.sin(qr[nonzero]) / qr[nonzero]
        integrand = r**2 * gr_minus_1 * sinc_term
        integral = simpson(integrand, r)
        S = 1 + 4 * np.pi * rho * integral
        S_q.append(S)
    return S_q


StructureFactor1 = compute_structure_factor(r, g_r, q_values, rho)


# Plot the result#
plt.figure(figsize=(10, 5))

# Bottom: Structure Factor
plt.subplot(2, 1, 2)
plt.plot(q_values, StructureFactor1, 'r-', label='S(q)')
plt.xlabel("q (Å⁻¹)")
plt.ylabel("S(q)")
plt.title("Structure Factor using Fourier Transform Water")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Convert structure factor and q to NumPy arrays for indexing #
StructureFactor1 = np.array(StructureFactor1)
q_values = np.array(q_values)
IndexedStructureFactor1 = StructureFactor1[15:]
IndexedQ1 = q_values[15:]
# Plot the sliced section
plt.figure(figsize=(8, 5))
plt.plot(IndexedQ1, IndexedStructureFactor1, color='blue',
         label='S(q) between q=0.8 and q=15')
plt.xlabel("q (Å⁻¹)")
plt.ylabel("S(q)")
plt.title("Zoomed-in Structure Factor (0.8 ≤ q ≤ 15)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# Smooth with 150-point moving average#


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# Trim initial spike - 75
cut_index = 75
TrimmedStructureFactor = StructureFactor1[cut_index:]
TrimmedQ = q_values[cut_index:]

# Apply moving average
window = 50
SmoothedS = moving_average(TrimmedStructureFactor, window)
SmoothedQ = TrimmedQ[window - 1:]

# Plot clean smooth line
plt.figure(figsize=(10, 5))
plt.plot(SmoothedQ, SmoothedS, color='darkgreen',
         linewidth=2, label=f'Smoothed S(q), window={window}')
plt.xlabel("q (Å⁻¹)")
plt.ylabel("S(q)")
plt.title("Final Smoothed Structure Factor 07/31")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
