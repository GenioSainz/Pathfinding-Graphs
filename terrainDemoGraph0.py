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
import time


starTime = time.time()

# create procedural terrain
############################## 200 3
noise = PerlinNoise(octaves=4, seed=1)
frec  = 0.00125*3

cellSize     = 5
contourLabel = cellSize
minXY = 0
N     = 201
maxXY = N*cellSize
x     = np.arange(N)*cellSize
y     = np.arange(N)*cellSize
X,Y   = np.meshgrid(x,y)
#Y     = np.flipud(Y)
Z     = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        z = noise([i*frec,j*frec])
        Z[i,j] = np.interp(z,[-1,1],[100,500])
     
print('Terrain: ',np.round(time.time() - starTime,3),' s')

# contour_levels 
#################
minLevel = roundGrid(np.amin(Z),contourLabel)-contourLabel
maxLevel = roundGrid(np.amax(Z),contourLabel)+contourLabel
contour_levels = np.arange( minLevel, maxLevel,contourLabel)

# compute gradient
####################
gY, gX = np.gradient(Z,cellSize)

# create graph 1
#################
G1 = gridGraphDIS(X,Y,Z)

# start - end nodes
####################
pA     = [70,480]
#pB     = [770,770]
pB     = [950,250]
node_i = coords2node(pA[0],pA[1],x,y,cellSize)
node_f = coords2node(pB[0],pB[1],x,y,cellSize)

# dijkstra 1
#############
nodesSP1       = nx.dijkstra_path(G1, node_i, node_f, weight='weight')
xSP1,ySP1,zSP1 = node2coords(nodesSP1,X,Y,Z)

print('Graph1: ',np.round(time.time() - starTime,3),' s')

# create graph 2
#################
G2 = gridGraph_DIS_SLOPE(X,Y,Z,gX,gY)

# dijkstra 2
#############
nodesSP2       = nx.dijkstra_path(G2, node_i, node_f, weight='weight')
xSP2,ySP2,zSP2 = node2coords(nodesSP2,X,Y,Z)

print('Graph2: ',np.round(time.time() - starTime,3),' s')

# init figure
##################
plt.close('all')
fig = plt.figure()

# subplot (1,2,1)
##################
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.grid();ax1.set_xlabel('X');ax1.set_ylabel('Y');ax1.set_zlabel('Z')
ax1.set( box_aspect=(1, 1, 0.2) )

meshH   = ax1.plot_surface(X,Y,Z, cmap=plt.cm.terrain,rcount=100,ccount=100,alpha=0.75)
contour = ax1.contour(X, Y, Z,levels=contour_levels,colors='k', linewidths=1,linestyles='solid')
ax1.plot(xSP1,ySP1,zSP1,'m')
ax1.plot(xSP2,ySP2,zSP2,'g')
#ax1.view_init(elev=90, azim=-90)
fig.colorbar(meshH, ax=ax1,shrink=0.5)

# subplot (1,2,2)
##################
ax2 = fig.add_subplot(1, 2, 2)
ax2.grid();ax2.set_xlabel('X');ax2.set_ylabel('Y')
ax2.set_aspect(1)

# custom color map
####################
color_dimension  = np.sqrt( gX**2 + gY**2 )
# minn, maxx = color_dimension.min(), color_dimension.max()
# norm =  matplotlib.colors.Normalize(minn, maxx)
# m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
# m.set_array([])
# fcolors = m.to_rgba(color_dimension)
# mesh    = ax1.plot_surface(X,Y,Z, facecolors=fcolors, vmin=minn, vmax=maxx,rcount=200,ccount=200,alpha=0.75)

ax2.grid(False)
meshC = ax2.pcolormesh(X, Y, color_dimension, cmap='jet')
ax2.contour(X, Y, Z, colors='k', levels=contour_levels, linewidths=0.5)
ax2.plot(xSP1,ySP1,'w',linewidth=3);ax2.plot(xSP1,ySP1,'r',linewidth=1,label="Path AB min(Distance)")
ax2.plot(xSP2,ySP2,'w',linewidth=3);ax2.plot(xSP2,ySP2,'b',linewidth=1,label="Path AB min(Slope)")
ax2.scatter(xSP1[0] , ySP1[0] , s=50, c='b', edgecolors='black')
ax2.scatter(xSP1[-1], ySP1[-1], s=50, c='b', edgecolors='black')

ax2.text(pA[0],pA[1],'A',bbox={'facecolor': 'w', 'alpha': 0.9, 'pad': 3})
ax2.text(pB[0],pB[1],'B',bbox={'facecolor': 'w', 'alpha': 0.9, 'pad': 3})


ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=1, fancybox=True).get_frame().set_edgecolor('k')
fig.colorbar(meshC, ax=ax2,shrink=0.5)

print('Figures: ',np.round(time.time() - starTime,3),' s')

plt.tight_layout()
plt.show()


figure = plt.gcf()  
figure.set_size_inches(16, 9)
plt.savefig('imgs/img0', bbox_inches='tight',dpi=300)
