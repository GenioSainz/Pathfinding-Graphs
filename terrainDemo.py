# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:25:24 2023

@author: Genio
"""
import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from utils import roundGrid




noise = PerlinNoise(octaves=4, seed=1)
frec  = 0.00125*3

cellSize = 5
minXY = 0
N     = 200
maxXY = N*cellSize
x     = np.linspace(minXY,maxXY,N)
y     = np.linspace(minXY,maxXY,N)
X,Y   = np.meshgrid(x,y)
Z     = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        z = noise([i*frec,j*frec])
        Z[i,j] = np.interp(z,[-1,1],[200,600])
        
minLevel = roundGrid(np.amin(Z),cellSize)-cellSize
maxLevel = roundGrid(np.amax(Z),cellSize)+cellSize
contour_levels = np.arange( minLevel, maxLevel, cellSize)

plt.close('all')
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
fig.subplots_adjust(top=1,bottom=0)
ax.view_init(elev=45, azim=45)

mesh    = ax.plot_surface(X,Y,Z, cmap=plt.cm.terrain,rcount=100,ccount=100,alpha=0.75)
contour = ax.contour(X, Y, Z,levels=contour_levels,colors='k', linewidths=1,linestyles='solid')

fig.colorbar(mesh, shrink=0.5)
ax.grid();ax.set_ylabel('X');ax.set_xlabel('Y');ax.set_zlabel('Z')
ax.set( box_aspect=(1, 1, 0.2) )
plt.show()

