# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:12:05 2025

@author: Mark Kaplanskiy
"""

# Full script is described in details here: https://www.researchgate.net/publication/357203546_Solving_3D_Time_Independent_Schrodinger_Equation_Using_Numerical_Method

import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.font_manager as fnt
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse

Path = 'path to txt file with r1, r2 and potential values'
data = np.loadtxt(Path)
r1, r2, E = data[:, 0], data[:, 1], data[:, 2]

N = 25
X = r1.reshape(N,N)
Y = r2.reshape(N,N)
ener = E.reshape(N,N) - np.min(E)

bohr_to_Ang = 0.529
dx = (abs(X[0][0] - X[1][1]))/bohr_to_Ang

m = 1836
V = m*dx**2*ener

diag = np.ones([N])
diags = np.array([diag, -2*diag, diag])
D = sparse.spdiags(diags, np.array([-1,0,1]), N, N)
T = -1/2 * sparse.kronsum(D,D)
U = sparse.diags(V.reshape(N**2), (0))
H = T+U
eigenvalues, eigenvectors = eigsh(H, k=2, which='SM')

def get_e(n):
    return eigenvectors.T[n].reshape((N,N))  

#Plotting graph
func_to_plot = ener

func_to_plot = func_to_plot/np.max(func_to_plot)
colors_absolute = [
    (0, (1, 64, 75)), 
    (0.005, (1, 97, 114)),
    (0.02, (10, 147, 150)),# Dark blue
    (0.05, (148, 210, 189)), # Light blue
    (0.15, (238, 155, 0)),  # Greenish
    (0.2, (202, 103, 2)), # Yellowish
    (0.5, (187, 62, 3)),    # Reddish
    (1, (148, 47, 2))    # Reddish
]

# Convert absolute RGB values to relative scale (0-1)
colors_relative = [(pos, tuple(val / 255.0 for val in rgb)) for pos, rgb in colors_absolute]

# Create a custom colormap
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors_relative)

font = fnt.FontProperties(family='Segoe UI')
font_style = {'fontname': 'Segoe UI', 'fontsize': 20}

plt.figure(figsize=(6,6))
plt.pcolormesh(X, Y, ener, cmap=cmap)
plt.xlabel('r\u2081, Å', fontsize=24, fontproperties=font)
plt.ylabel('r\u2082, Å', fontsize=24, fontproperties=font)
plt.xlim(0.8, 2.05)
plt.ylim(0.8, 2.0)
plt.rcParams['axes.linewidth'] = 2

ax = plt.gca()
ax.set_xticklabels([str(i.round(2)) for i in np.arange(0.8, 2.2, 0.2)], fontdict=font_style)
ax.set_yticklabels([str(i.round(2)) for i in np.arange(0.8, 2.2, 0.2)], fontdict=font_style)
plt.show()
