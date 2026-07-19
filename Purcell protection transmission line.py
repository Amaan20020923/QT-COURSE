# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:45:29 2026

@author: Amaan Ansari
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# Hilbert-space dimensions
Nc = 6
n = 2
# Operators
a = qt.tensor(qt.destroy(Nc), qt.qeye(n))
nc = a.dag()*a

sm = qt.tensor(qt.qeye(Nc), qt.sigmam())
sp = sm.dag()
sz = qt.tensor(qt.qeye(Nc), qt.sigmaz())

# Parameters
wq = 6.0     # In GHz
g = 0.05     # In GHz
kappa = 0.1  # In GHz

Delta_list = [1.0 ,2.0 ,3.0 ,4.0 ,5.0] # In GHz

# Initial state :- cavity vacuum + qubit excited
psi0 = qt.tensor(qt.basis(Nc,0),qt.basis(n,0))

# Time window
tlist = np.linspace(0,400,3000) # In nano seconds 

plt.figure(figsize=(10,6))

for Delta in Delta_list:

    wc = wq - Delta

    Hc = wc*nc                                                      # Cavity Hamiltonian
    Hq = (wq/2)*qt.tensor(qt.qeye(Nc), qt.sigmaz())                 # Qubit Hamiltonian
    Hint = g*(a.dag()*sm + a*sp)                                    # Interaction Hamiltonian

    H = Hc + Hq + Hint                                              # Complete Hamiltonian

    # cavity leakage into transmission line
    c_ops = [np.sqrt(kappa)*a]                                      # Collapse operator

    result = qt.mesolve(H,psi0,tlist,c_ops)                         # Solve master equation

    cavity_pop = [qt.expect(nc, state) for state in result.states]  # Cavity population

    plt.plot(tlist,cavity_pop,linewidth=2,label=r'$\Delta={}$'.format(Delta))

plt.xlabel(f'Time ($10^{{-9}}$s) ------------>')
plt.ylabel(r'Cavity population, $\langle a^\dagger a\rangle$ ------------>')
plt.title('Cavity population with time.')
plt.grid(True)
#plt.xlim(-0.5,16)
plt.legend()
plt.show()