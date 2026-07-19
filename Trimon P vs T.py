# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:55:27 2026

@author: Amaan Ansari
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

x = int(input('Enter the initial state of transmon A :- '))
y = int(input('Enter the initial state of transmon B :- '))
z = int(input('Enter the initial state of transmon C :- '))



# Dimension of the Hilbert space
N = 4

# Annihilation operators
aA = tensor(destroy(N),qeye(N),qeye(N))
aB = tensor(qeye(N),destroy(N),qeye(N))
aC = tensor(qeye(N),qeye(N),destroy(N))

# Number operators
nA = aA.dag()*aA
nB = aB.dag()*aB
nC = aC.dag()*aC

# Frequencies (GHz × 2π)
wA = 4.9278*2*np.pi
wB = 4.5146*2*np.pi
wC = 5.6864*2*np.pi

# Self-Kerr (MHz × 2π)
JA = 107.9*np.pi
JB = 113.1*np.pi
JC = 138.0*np.pi

# Cross-Kerr (MHz × 2π)
JAB = JBA = 192.4*np.pi
JBC = JCB = 211.4*np.pi
JAC = JCA = 242.0*np.pi

# Frequency shifts (MHz × 2π)
betaA = JA + JAB + JAC
betaB = JB + JBC + JBA
betaC = JC + JCA + JCB


# Hamiltonian
H = ((wA-betaA)*nA - JA*nA**2 + (wB-betaB)*nB - JB*nB**2 + (wC-betaC)*nC
    - JC*nC**2 - 2*JAB*nA*nB - 2*JAC*nA*nC - 2*JBC*nB*nC)

# Relaxation rates in (us)^-1
gammaA = 1.0/43.7
gammaB = 1.0/43.5
gammaC = 1.0/26.3

# Collapse operators
L = [np.sqrt(gammaA)*aA,np.sqrt(gammaB)*aB,np.sqrt(gammaC)*aC]

# Initial state
psi0 = tensor(basis(N, x),basis(N, y),basis(N, z))

# Time (microseconds)
tlist = np.linspace(0,290,500)

# Solve master equation
result = mesolve(H,psi0,tlist,L,e_ops=[])

p1_A = []
p1_B = []
p1_C = []
for state in result.states:
        # ptrace(0) isolates Mode A, ptrace(1) isolates Mode B, ptrace(2) isolates Mode C
        rho_A = state.ptrace(0)
        rho_B = state.ptrace(1)
        rho_C = state.ptrace(2)
        
        # Pull the excited state population (index 1 of the diagonal)
        p1_A.append(rho_A.diag()[x])
        p1_B.append(rho_B.diag()[y])
        p1_C.append(rho_C.diag()[z])


fig, ax = plt.subplots() 
# Anchored text
param_text2 = (
    "Literature Parameters :-\n\n"
    r"$\omega_A = 30.9623$ GHz" + "\n"
    r"$\omega_B = 28.3660$ GHz" + "\n"
    r"$\omega_C = 35.7287$ GHz" + "\n\n"
    r"$J_A = 338.9778$ MHz" + "\n"
    r"$J_B = 355.3141$ MHz" + "\n"
    r"$J_C = 433.5398$ MHz" + "\n\n"
    r"$J_{AB} = J_{BA} = 604.4425$ MHz" + "\n"
    r"$J_{BC} = J_{CB} = 664.1327$ MHz" + "\n"
    r"$J_{CA} = J_{AC} = 760.2654$ MHz" + "\n\n"
    r"$T_A = 43.7$ $\mu s$" + r" $\Rightarrow$ " + "$\gamma_A$ = " + "0.0229" + " $(\mu s)^{-1}$" + "\n"
    r"$T_B = 43.5$ $\mu s$" + r" $\Rightarrow$ " + "$\gamma_A$ = " + "0.0230" + " $(\mu s)^{-1}$" + "\n"
    r"$T_C = 26.3$ $\mu s$" + r" $\Rightarrow$ " + "$\gamma_A$ = " + "0.0380" + " $(\mu s)^{-1}$" 
)

at2 = AnchoredText(param_text2,prop=dict(size=15, color="indigo"),frameon=True,loc='center right')
at2.patch.set_boxstyle("round,pad=0.3")
at2.patch.set_alpha(0.8)
ax.add_artist(at2) 

# Numerical data
plt.grid(True)
plt.plot(tlist,p1_A,label='Transmon A',color='red',lw=2)
plt.plot(tlist,p1_B,label='Transmon B',color='gold',lw=4,linestyle='dotted')
plt.plot(tlist,p1_C,label='Transmon C',color='black',lw=3,linestyle='dotted')
plt.xlabel(f'Time ($10^{{-6}}$s) -------------->',fontsize=22)
plt.ylabel('Probability --------------->',fontsize=22) 
plt.title('Probability vs Time for a Trimon qubit.',fontsize=22)
plt.legend(fontsize=16)
plt.show()

