# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:25:24 2023

@author: Genio
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from   perlin_noise import PerlinNoise
from   utils import roundGrid,gridGraphDIS,gridGraph_DIS_SLOPE,node2coords,coords2node

# create procedural terrain
############################## 200 3
noise = PerlinNoise(octaves=4, seed=1)
frec  = 0.00125*3

cellSize     = 5
contourLabel = cellSize
minXY = 0
N     = 220
maxXY = N*cellSize
x     = np.arange(N)*cellSize
y     = np.arange(N)*cellSize
X,Y   = np.meshgrid(x,y)
Z     = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        z = noise([i*frec,j*frec])
        Z[i,j] = np.interp(z,[-1,1],[100,500])

# compute gradient
####################
gY, gX = np.gradient(Z,cellSize)

# create graph 1
#################
G1 = gridGraphDIS(X,Y,Z)

# dijkstra 1
#############
node_i         = coords2node(400,200,x,y,cellSize)
node_f         = coords2node(800,1000,x,y,cellSize)
nodesSP1       = nx.dijkstra_path(G1, node_i, node_f, weight='weight')
xSP1,ySP1,zSP1 = node2coords(nodesSP1,X,Y,Z)

minLevel = roundGrid(np.amin(Z),contourLabel)-contourLabel
maxLevel = roundGrid(np.amax(Z),contourLabel)+contourLabel
contour_levels = np.arange( minLevel, maxLevel,contourLabel)

# create graph 2
#################
G2 = gridGraph_DIS_SLOPE(X,Y,Z,gX,gY)

# dijkstra 2
#############
node_i         = coords2node(400,200,x,y,cellSize)
node_f         = coords2node(800,1000,x,y,cellSize)
nodesSP2      = nx.dijkstra_path(G2, node_i, node_f, weight='weight')
xSP2,ySP2,zSP2 = node2coords(nodesSP2,X,Y,Z)

minLevel = roundGrid(np.amin(Z),contourLabel)-contourLabel
maxLevel = roundGrid(np.amax(Z),contourLabel)+contourLabel
contour_levels = np.arange( minLevel, maxLevel,contourLabel)


plt.close('all')

fig = plt.figure(figsize=plt.figaspect(0.5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.grid();ax1.set_xlabel('X');ax1.set_ylabel('Y');ax1.set_zlabel('Z')
ax1.set( box_aspect=(1, 1, 0.2) )

mesh    = ax1.plot_surface(X,Y,Z, cmap=plt.cm.terrain,rcount=100,ccount=100,alpha=0.75)
contour = ax1.contour(X, Y, Z,levels=contour_levels,colors='k', linewidths=1,linestyles='solid')
ax1.plot(xSP1,ySP1,zSP1,'r')
ax1.plot(xSP2,ySP2,zSP2,'b')


ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor((0.25,0.25,0.25))
ax2.grid();ax2.set_xlabel('X');ax2.set_ylabel('Y');

ax2.contour(X, Y, Z, levels=contour_levels, cmap=plt.cm.terrain, linewidths=1)
ax2.plot(xSP1,ySP1,'r')
ax2.plot(xSP2,ySP2,'b')
ax2.scatter(xSP1[0] , ySP1[0] , s=50, c='b', edgecolors='black')
ax2.scatter(xSP1[-1], ySP1[-1], s=50, c='b', edgecolors='black')

plt.tight_layout()
plt.show()

