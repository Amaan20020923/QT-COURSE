# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 02:48:08 2026

@author: Amaan Ansari
"""


import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# Hilbert Space Dimensions
n = 1
N_c = 6   # Cavity cutoff dimension
N_q = n + 1   # Transmon dimension (0, 1, 2 states to account for anharmonicity)

# Cavity operators 
c = qt.tensor(qt.destroy(N_c), qt.qeye(N_q))
nc = c.dag() * c

# Transmon operators
a = qt.tensor(qt.qeye(N_c), qt.destroy(N_q))
na = a.dag() * a

# Time window
tlist = np.linspace(0, 100000, 3000)

# Projector for the transmon excited state |n> probability
P1_q = qt.tensor(qt.qeye(N_c), qt.basis(N_q, n) * qt.basis(N_q, n).dag())

# Initial State: Cavity in vacuum |0>, Transmon in excited state |n>
psi0 = qt.tensor(qt.basis(N_c, 0), qt.basis(N_q, n))

# Parameters & Simulation Loop
D = [1.0, 2.0, 3.0, 4.0, 5.0]  # Detunings (\Delta) in GHz
wq = 6.0                       # Transmon frequency in GHz
g = 0.05                       # Coupling strength in GHz
alpha = -0.2                   # Transmon anharmonicity in GHz
kappa = 1.0                    # Cavity line width in GHz

plt.figure(figsize=(10, 6))

for d_val in D:
    wc = wq - d_val            # Cavity frequency
    chi = (g**2) / d_val       # Dispersive shift rate
    
    # Hamiltonian
    H_cavity = wc * nc
    H_transmon = wq * na + (alpha / 2.0) * a.dag() * a.dag() * a * a
    H_disp = chi * nc * na
    H = H_cavity + H_transmon + H_disp
    
    gamma_P = kappa * (g / d_val)**2
    c_ops = [np.sqrt(gamma_P) * a]
    
    # Solve the Master Equation
    result = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[P1_q])
    
    # Plotting the excited state probability profile
    plt.plot(tlist, result.expect[0], label=r'$\Delta$ = {}'.format(d_val))
    
# Plot Formatting
plt.xlabel(r'Time ($10^{{-6}}$s) ----------->', fontsize=22)
plt.ylabel(r'Probability ----------->', fontsize=22)
plt.ylim(-0.05, 1.05)
plt.title("Purcell protection to transmon due to cavity.",fontsize=22)
plt.grid(True)
plt.legend(fontsize = 22)
plt.show()
