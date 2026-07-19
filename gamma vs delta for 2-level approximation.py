# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:21:30 2026

@author: Amaan Ansari
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. Hilbert-space Dimensions & Operators
# ==========================================
Nc = 6          # Cavity cutoff
Nq = 2          # Two-level qubit

a = qt.tensor(qt.destroy(Nc), qt.qeye(Nq))
nc = a.dag() * a

sm = qt.tensor(qt.qeye(Nc), qt.sigmam())
sp = sm.dag()

# Initial state: Cavity |0>, Qubit |e>
psi0 = qt.tensor(qt.basis(Nc, 0), qt.basis(Nq, 0))

# Projector onto qubit excited state
Pe = qt.tensor(qt.qeye(Nc), qt.basis(Nq, 0) * qt.basis(Nq, 0).dag())

# ==========================================
# 2. Physical Parameters
# ==========================================
wq = 6.0       # In GHz 
g = 0.07       # In GHz
kappa = 1.0    # GHz

# Define the requested detuning ranges and merge them cleanly
Delta_coarse = np.arange(-4.0, 4.1, 0.1)
Delta_fine = np.arange(-0.1, 0.11, 0.01)
Delta_list = np.unique(np.concatenate((Delta_coarse, Delta_fine)))
Delta_list = np.sort(Delta_list)

# Time window for simulation
# Note: Near Delta = 0, decay is very fast. A long tlist is needed for large Delta,
# but we can fit the initial decay part to ensure curve_fit succeeds everywhere.
tlist = np.linspace(0, 10000, 2000)   # In nanoseconds 

# ==========================================
# 3. Fitting Function
# ==========================================
def exponential_decay(t, A, gamma):
    """Simple exponential decay function for fitting."""
    return A * np.exp(-gamma * t)

# Arrays to store extracted decay rates
gamma_list = []

print(f"Starting simulation for {len(Delta_list)} detuning points...")

# ==========================================
# 4. Simulation and Fitting Loop
# ==========================================
for i, Delta in enumerate(Delta_list):
    wc = wq - Delta

    # Hamiltonian
    Hc = wc * nc
    Hq = (wq / 2) * qt.tensor(qt.qeye(Nc), qt.sigmaz())
    Hint = g * (a.dag() * sm + a * sp)
    H = Hc + Hq + Hint

    # Cavity decay
    c_ops = [np.sqrt(kappa) * a]

    # Solve Master Equation
    result = qt.mesolve(H, psi0, tlist, c_ops)
    qubit_population = np.array([qt.expect(Pe, state) for state in result.states])

    # Fit the population curve to extract gamma
    # We provide a reasonable initial guess: p0=[amplitude=1.0, gamma=1e-4]
    try:
        # For resonant/near-resonant cases, oscillations occur. 
        # curve_fit will track the average exponential envelope.
        popt, _ = curve_fit(exponential_decay, tlist, qubit_population, p0=[1.0, 1e-4], bounds=(0, [2.0, 1.0]))
        gamma_extracted = popt[1]
    except Exception as e:
        print(f"Fitting failed for Delta = {Delta}: {e}")
        gamma_extracted = np.nan

    gamma_list.append(gamma_extracted)

    # Progress indicator
    if (i + 1) % 10 == 0 or i == len(Delta_list) - 1:
        print(f"Progress: {i + 1}/{len(Delta_list)} points calculated.")

gamma_list = np.array(gamma_list)

# ==========================================
# 5. Plotting the Results
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(Delta_list, gamma_list, 'o-', color='darkblue', markersize=4, label='Extracted $\gamma$')

# Analytical Purcell formula for comparison: gamma_P = kappa * g^2 / (Delta^2 + (kappa/2)^2)
# Note: Very close to Delta=0, the weak-coupling Purcell approximation breaks down 
# as the system enters the strong coupling/Rabi oscillation regime.
gamma_analytical = kappa * (g**2) / (Delta_list**2 + (kappa/2)**2)
plt.plot(Delta_list, gamma_analytical, '--', color='red', alpha=0.7, label='Analytical Purcell Formula')

plt.xlabel(f"Detuning, $\Delta = \omega_q - \omega_c$ (GHz) ------------>",fontsize = 22)
plt.ylabel(f"Decay rate, $\gamma$ $(10^{{-9}}s)$ ---------->",fontsize = 22)
plt.title(r"Transmon qubit decay rate $\gamma$ vs Detuning $\Delta$ for two level approximation",fontsize = 22)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize = 22)
plt.show()