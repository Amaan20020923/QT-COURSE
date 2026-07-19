# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:03:46 2026

@author: Amaan Ansari
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, create, mesolve, basis

def solve_two_level_system():
    # Simulation Time
    tlist = np.linspace(0, 100, 300) # In nano seconds
    n = int(input('Enter the initial state :- ')) 
    if n == 0 : 
        p0 = np.ones(len(tlist)) 
        plt.figure(figsize=(9, 5))
        plt.plot(tlist,p0,color='black',lw=2) 
        plt.title('Time Evolution of State Probabilities at T = 0 K.')
        plt.xlabel('Time --------------->')
        plt.ylabel('Probability --------------->')
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.show()
        
    if n >= 1 : 
        N = n + 1             # Hilbert space dimension
        omega = 1.0           # Transition frequency
        
        a = destroy(N)        # Annihilation operator
        adag = create(N)      # Creation operator
        H = omega * adag * a  # Hamiltonian 
        
        # Initial State
        psi0 = basis(N, n)

        # Collapse Operators (Environment Interaction)
        gamma = 0.1                   # Damping coefficient in GHz
        c_ops = np.sqrt(gamma)*a      # Collapse operator 

        # Solve the Master Equation
        result = mesolve(H, psi0, tlist, c_ops, e_ops=[])
        
        # Extracting Probabilities and Expectation Values
        p0 = [state.diag()[0] for state in result.states] # Probability of being in |0>
        p1 = [state.diag()[n] for state in result.states] # Probability of being in |n>
       
        # Plotting the Probabilities
        plt.figure(figsize=(10, 6))
        plt.plot(tlist, p1, label=f'Prob of Excited State $P_{{{n}}}(t)$', color='darkorange', lw=2.5)
        plt.plot(tlist, p0, label=r'Prob of Ground State $P_0(t)$', color='teal', lw=2)
      
        plt.grid(True)
        plt.title('Time evolution of state probabilities of a harmonic oscillator at T = 0 K.',fontsize = 22)
        plt.xlabel(f'Time ($10^{{-9}}$s)--------------->',fontsize=22)
        plt.ylabel('Probability --------------->',fontsize = 22)
        plt.ylim(-0.05, 1.05)
        plt.legend(fontsize = 22)
        plt.show()
        
if __name__ == "__main__":
    solve_two_level_system()