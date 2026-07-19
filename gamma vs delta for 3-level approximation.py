# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 00:52:26 2026

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
n = 3          # Initial state of the qubit
Nq = 4          # Three-level system (0, 1, 2)
# Cavity operators
a = qt.tensor(qt.destroy(Nc), qt.qeye(Nq))
nc = a.dag() * a

# Three-level system (qutrit) operators
# destroy(3) creates a 3x3 lowering operator matrix
b = qt.tensor(qt.qeye(Nc), qt.destroy(Nq))
nq = b.dag() * b  # Number operator for qutrit

# Initial state: Cavity |0>, Qubit |n> (the second excited state)
psi0 = qt.tensor(qt.basis(Nc, 0), qt.basis(Nq, n))

# Projector onto the qutrit nth excited state |n> 
Pe = qt.tensor(qt.qeye(Nc), qt.basis(Nq, n) * qt.basis(Nq, n).dag())

# ==========================================
# 2. Physical Parameters
# ==========================================
wq = 6.0        # In GHz
alpha = -0.7    # In GHz
g = 0.05        # In GHz
kappa = 1.0     # GHz 

# Define detuning ranges (Delta = wq - wc)
Delta_coarse = np.arange(-4.0, 4.1, 0.1)    # In GHz
Delta_fine = np.arange(-0.1, 0.11, 0.01)    # In GHz
Delta_list = np.unique(np.concatenate((Delta_coarse, Delta_fine)))
Delta_list = np.sort(Delta_list)

# Time window for simulation
tlist = np.linspace(0, 10000, 2000)    # In nano seconds

# ==========================================
# 3. Fitting Function
# ==========================================
def exponential_decay(t, A, gamma):
    return A * np.exp(-gamma * t)

gamma_list = []

print(f"Starting 3-level simulation for {len(Delta_list)} detuning points...")

# ==========================================
# 4. Simulation and Fitting Loop
# ==========================================
for i, Delta in enumerate(Delta_list):
    wc = wq - Delta

    # Cavity Hamiltonian
    Hc = wc * nc
    
    # Three-level Qutrit Hamiltonian with Anharmonicity
    # H_q = wq * |1><1| + (2*wq + alpha) * |2><2|
    # This can be written elegantly using the number operator:
    Hq = wq * nq + (alpha / 2.0) * nq * (nq - 1)
    
    # Interaction Hamiltonian (Jaynes-Cummings extension for 3 levels)
    # b handles both |1><0| and sqrt(2)*|2><1| transitions automatically
    Hint = g * (a.dag() * b + a * b.dag())
    
    H = Hc + Hq + Hint

    # Cavity decay
    c_ops = [np.sqrt(kappa) * a]

    # Solve Master Equation
    result = qt.mesolve(H, psi0, tlist, c_ops)
    qutrit_population = np.array([qt.expect(Pe, state) for state in result.states])

    # Fit the population curve to extract gamma
    try:
        popt, _ = curve_fit(exponential_decay, tlist, qutrit_population, p0=[1.0, 1e-4], bounds=(0, [2.0, 1.0]))
        gamma_extracted = popt[1]
    except Exception as e:
        print(f"Fitting failed for Delta = {Delta}: {e}")
        gamma_extracted = np.nan

    gamma_list.append(gamma_extracted)

    if (i + 1) % 10 == 0 or i == len(Delta_list) - 1:
        print(f"Progress: {i + 1}/{len(Delta_list)} points calculated.")

gamma_list = np.array(gamma_list)

# ==========================================
# 5. Plotting the Results
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(Delta_list, gamma_list, 'o-', color='darkgreen', markersize=4, label=r'Extracted $\gamma$ (3-level)')

# Multi-level analytical Purcell expression for comparison:
# The analytical decay rate for state |n> features an asymmetric peak shifted by alpha
gamma_analytical = kappa * (g**2)*n / ((Delta_list + (n-1)*alpha)**2 + (kappa/2)**2)
plt.plot(Delta_list, gamma_analytical, '--', color='red', alpha=0.7, label=f'Analytical {3}-Level Purcell')

plt.xlabel(r"Detuning, $\Delta = \omega_q - \omega_c$ (GHz) ----------->",fontsize = 22)
plt.ylabel(r"Decay Rate, $\gamma$ $(10^{{-9}}s)$ ----------->",fontsize = 22)
plt.title(r"Transmon qubit decay rate $\gamma$ vs Detuning $\Delta$ for two 3 approximation",fontsize = 22)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize = 22)
plt.show()