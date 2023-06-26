# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:25:24 2023

@author: Genio
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from   utils import roundGrid,gridGraphDIS,gridGraph_DIS_SLOPE,node2coords,coords2node


cellSize     = 5
contourLabel = cellSize
minXY = 0
N     = 4
maxXY = N*cellSize
x     = np.arange(N)*cellSize
y     = np.arange(N)*cellSize
X,Y   = np.meshgrid(x,y)
#Y     = np.flipud(Y)
Z     = np.zeros((N,N))
Z[2]=5
#Z[1]=5

# compute gradient
####################
gY, gX = np.gradient(Z,cellSize)

# create graph 1
#################
G1 = gridGraphDIS(X,Y,Z)


# start - end nodes
####################
pA     = [0,10]
pB     = [10,10]
node_i = coords2node(pA[0],pA[1],x,y,cellSize)
node_f = coords2node(pB[0],pB[1],x,y,cellSize)

# dijkstra 1
#############
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
nodesSP2       = nx.dijkstra_path(G2, node_i, node_f, weight='weight')
xSP2,ySP2,zSP2 = node2coords(nodesSP2,X,Y,Z)

minLevel = roundGrid(np.amin(Z),contourLabel)-contourLabel
maxLevel = roundGrid(np.amax(Z),contourLabel)+contourLabel
contour_levels = np.arange( minLevel, maxLevel,contourLabel)

# init figure
##################
plt.close('all')


fig1 = plt.figure()

# subplot (1,2,1)
##################
ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
ax1.grid();ax1.set_xlabel('X');ax1.set_ylabel('Y');ax1.set_zlabel('Z')
ax1.set( box_aspect=(1, 1, 0.2) )

# custom color map
####################
color_dimension  = np.sqrt( gX**2 + gY**2 )

meshH   = ax1.plot_surface(X,Y,Z, cmap=plt.cm.terrain,rcount=100,ccount=100,alpha=0.75)
#contour = ax1.contour(X, Y, Z,levels=contour_levels,colors='k', linewidths=1,linestyles='solid')
ax1.plot(xSP1,ySP1,zSP1,'m')
ax1.plot(xSP2,ySP2,zSP2,'g')
ax1.view_init(elev=90, azim=-90)
fig1.colorbar(meshH, ax=ax1,shrink=0.5)

# subplot (1,2,2)
##################
ax2 = fig1.add_subplot(1, 2, 2)
ax2.grid();ax2.set_xlabel('X');ax2.set_ylabel('Y')
ax2.set_aspect(1)

meshC = ax2.pcolormesh(X, Y, color_dimension, cmap='jet')
#ax2.contour(X, Y, Z, colors='k', levels=contour_levels, linewidths=0.5)
ax2.plot(xSP1,ySP1,'w',linewidth=3);ax2.plot(xSP1,ySP1,'r',linewidth=1)
ax2.plot(xSP2,ySP2,'w',linewidth=3);ax2.plot(xSP2,ySP2,'b',linewidth=1)
ax2.scatter(xSP1[0] , ySP1[0] , s=50, c='b', edgecolors='black')
ax2.scatter(xSP1[-1], ySP1[-1], s=50, c='b', edgecolors='black')
fig1.colorbar(meshC, ax=ax2,shrink=0.5)

plt.tight_layout()
plt.show()


W1 = nx.get_edge_attributes(G1,'weight')
P1 = nx.get_node_attributes(G1,'pos')

W2 = nx.get_edge_attributes(G2,'weight')
P2 = nx.get_node_attributes(G2,'pos')

for w in W1: W1[w]=np.round(W1[w],2)
for w in W2: W2[w]=np.round(W2[w],2)

fig2 = plt.figure()
ax3 = fig2.add_subplot(1, 2, 1)
ax3.set_box_aspect(1)
nx.draw_networkx(G1, P1, node_color='r', node_size=200)
nx.draw_networkx_edge_labels(G1, pos=P1, edge_labels=W1, font_size=8, label_pos=0.3)

ax4 = fig2.add_subplot(1, 2, 2)
ax4.set_box_aspect(1)
nx.draw_networkx(G2, P2,  node_color='g',node_size=200)
nx.draw_networkx_edge_labels(G2, pos=P2, edge_labels=W2, font_size=8, label_pos=0.3)




