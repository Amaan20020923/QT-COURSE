# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 02:15:17 2026

@author: Amaan Ansari
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# Hilbert-space dimensions
Nc = 6          # Cavity cutoff
Nq = 2          # Two-level qubit

# Operators
a = qt.tensor(qt.destroy(Nc), qt.qeye(Nq))
nc = a.dag() * a

sm = qt.tensor(qt.qeye(Nc), qt.sigmam())
sp = sm.dag()
sz = qt.tensor(qt.qeye(Nc), qt.sigmaz())

# Parameters
wq = 6.0     # In GHz
g = 0.05     # In GHz
kappa = 0.1  # In GHz 

Delta_list = [1.0, 2.0, 3.0, 4.0, 5.0]

# Initial state
# Cavity : |0>
# Qubit  : |e> (= basis(2,0) in your QuTiP version)
psi0 = qt.tensor(qt.basis(Nc, 0),qt.basis(Nq, 0))

# Projector onto qubit excited state
Pe = qt.tensor(qt.qeye(Nc),qt.basis(Nq, 0) * qt.basis(Nq, 0).dag())

# Time window
tlist = np.linspace(0, 100000, 5000) # In nano seconds
plt.figure(figsize=(10,6))
for Delta in Delta_list:
    wc = wq - Delta

    # Hamiltonian
    Hc = wc * nc
    Hq = (wq / 2) * qt.tensor(qt.qeye(Nc), qt.sigmaz())
    Hint = g * (a.dag()*sm + a*sp)

    H = Hc + Hq + Hint

    # Cavity decay
    c_ops = [np.sqrt(kappa) * a]

    # Master equation
    result = qt.mesolve(H, psi0, tlist, c_ops)

    # Qubit excited-state probability
    qubit_population = [qt.expect(Pe, state) for state in result.states]

    plt.plot(tlist,qubit_population,linewidth=2,label=r'$\Delta={}$ GHz'.format(Delta))

plt.xlabel("Time ($10^{{-9}}$s)----------->",fontsize = 22)
plt.ylabel("Transmon qubit excited-state probability ----------->",fontsize = 22)
plt.title("Purcell decay of the transmon qubit",fontsize = 22)
plt.grid(True)
plt.legend(fontsize = 22)
plt.ylim(-0.05,1.05)
plt.show()