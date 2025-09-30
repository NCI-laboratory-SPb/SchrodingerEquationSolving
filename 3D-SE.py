# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:20:49 2025

@author: Mark Kaplanskiy
"""

# Full script is described in details here: https://www.researchgate.net/publication/357203546_Solving_3D_Time_Independent_Schrodinger_Equation_Using_Numerical_Method

import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy import sparse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fnt

Path = 'path to txt file with r1, r2, alpha and potential values'
data = np.loadtxt(Path)
r1, r2, alpha, E = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

##Create meshgrid for x y z
N1 = 19
N2 = 23
m = 1836
X, Y, Z, ener = r1.resize(N1, N2, N2), r2.resize(N1, N2, N2), alpha.resize(N1, N2, N2), E.resize(N1, N2, N2)
dx = abs(X[0][0][0] - X[0][0][1])/0.529

V = m * dx**2 * (ener - np.min(ener))

##create matrix
diag = np.ones([N1])
diags = np.array([diag, -2*diag, diag])
D1 = sparse.spdiags(diags, np.array([-1, 0,1]), N1, N1)

diag = np.ones([N2])
diags = np.array([diag, -2*diag, diag])
D2 = sparse.spdiags(diags, np.array([-1, 0,1]), N2, N2)

##define energy
D11 = sparse.kronsum(D2,D2)
D22 = sparse.kronsum(D1,D1)

T = -1/2 * D22
U = sparse.diags(V.reshape(N1 * N2 * N2),(0))
H = T+U

##Solve for eigenvector and eigenvalue
eigenvalues , eigenvectors = eigsh(H, k=2, which='SM')

def get_e(n):
    return eigenvectors.T[n].reshape((N1,N2,N2))

##plot V
func_to_plot = ener # or get_e(0)**2
func_to_plot = func_to_plot - np.min(func_to_plot)
func_to_plot = func_to_plot/np.max(func_to_plot)

colors_absolute = [
    (0, (1, 64, 75)), 
    (0.005, (1, 97, 114)),
    (0.02, (10, 147, 150)),# Dark blue
    (0.05, (148, 210, 189)), # Light blue
    (0.15, (238, 155, 0)),  # Greenish
    (0.25, (202, 103, 2)), # Yellowish
    (0.4, (187, 62, 3)),    # Reddish
    (1, (148, 47, 2))    # Reddish
]

# Convert absolute RGB values to relative scale (0-1)
colors_relative = [(pos, tuple(val / 255.0 for val in rgb)) for pos, rgb in colors_absolute]

# Create a custom colormap
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors_relative)
font = fnt.FontProperties(family='Segoe UI')
font_style = {'fontname': 'Segoe UI', 'fontsize': 12}

fig = plt.figure(0,dpi=100,figsize=(4.5,4.5))
ax = fig.add_subplot(111, projection='3d')
plot0 = ax.scatter3D(X, Y, Z, c=func_to_plot, cmap=cmap, s=5, alpha=0.3, antialiased=True)
plt.rcParams['axes.linewidth'] = 2

ax.set_xticks([i.round(2) for i in np.arange(0.8, 2.2, 0.2)])
ax.set_yticks([i.round(2) for i in np.arange(0.8, 2.2, 0.2)])
ax.set_zticks([i.round(2) for i in np.arange(100, 200, 20)])

ax.set_xticklabels([str(i.round(2)) for i in np.arange(0.8, 2.2, 0.2)], fontdict=font_style)
ax.set_yticklabels([str(i.round(2)) for i in np.arange(0.8, 2.2, 0.2)], fontdict=font_style)
ax.set_zticklabels([str(i.round(2)) for i in np.arange(100, 200, 20)], fontdict=font_style)
plt.show()
