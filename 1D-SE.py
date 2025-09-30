# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 00:01:43 2024

@author: Mark Kaplanskiy
"""

# Full script is described in details here: https://www.researchgate.net/publication/357203546_Solving_3D_Time_Independent_Schrodinger_Equation_Using_Numerical_Method

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

Path = 'path to txt file with q1 and potential values'
data = np.loadtxt(Path)
q1, ener = data[:, 0], data[:, 1]

c = np.polyfit(q1, ener, 8)
potential_as_polynom = np.poly1d(c)
q1_out = np.linspace(-0.5, 0.5, 200)
potential_out = potential_as_polynom(q1_out)

N = len(potential_out)
L = max(q1_out) - min(q1_out)
y = (q1_out - min(q1_out))/L
dy = abs(y[0] - y[1])
m = 1836
bohr_to_Ang = 0.529

d = 1/dy**2 + m*((L/bohr_to_Ang)**2)*potential_out             #data[:,1]
e = -1/(2*dy**2) * np.ones(len(d)-1)
w, v = eigh_tridiagonal(d, e)

# Plot potential - x
plt.figure(figsize=(8,5))
plt.plot(q1_out, potential_out, 'r', lw=5)
plt.scatter(q1, ener, c='k', s=35)
plt.title('Potential', fontsize=30)
plt.ylabel('$E$', fontsize=25)
plt.xlabel('$q1$', fontsize=25)
plt.grid()

# Plot eigenfunctions - x
plt.figure(figsize=(10,5))
plt.plot(q1_out, v.T[0]**2, 'r', lw=5)
plt.ylabel('|$\psi$|^2', fontsize=15)
plt.xlabel('q1, Å', fontsize=15)
plt.grid()

# Ploting potential and probability density function
# Plot Line1 (Left Y Axis)
fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
ax1.plot(q1_out, potential_out*627.5, color='k', lw=10)

# Plot Line2 (Right Y Axis)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(q1_out, v.T[0]**2, color='orange', lw=10)
ax2.set_ylim([0, max(v.T[0]**2)*2.5])

# Decorations
# ax1 (left Y axis)
ax1.set_xlabel('q1, Å', fontsize=20)
ax1.tick_params(axis='x', rotation=0, labelsize=12)
ax1.set_ylabel('Energy, kcal/mol', color='k', fontsize=20)
ax1.tick_params(axis='y', rotation=0, labelcolor='black' )
ax1.grid(alpha=.4)

# ax2 (right Y axis)
ax2.set_ylabel("|$\psi$|^2", color='orange', fontsize=20)
ax2.tick_params(axis='y', labelcolor='orange')
fig.tight_layout()
plt.grid()
plt.show()
