# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:24:26 2023

@author: Genio
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from utils import gridGraph

nX = 10
nY = 10
cellSize = 5

# coordinates
################
#Z = np.arange(nY*nX).reshape(nY, nX)
Z = np.random.randint(1,1000, size=(nY,nX))
x = np.arange(nX)*cellSize
y = np.arange(nY)*cellSize

# create graph
#################
G = gridGraph(Z,x,y,cellSize)
W = nx.get_edge_attributes(G,'weight')
P = nx.get_node_attributes(G,'pos')

# dijkstra
#############
nodesSP = nx.dijkstra_path(G, 0, nX*nY-1, weight='weight')
edgesSP = list(zip(nodesSP,nodesSP[1:]))

nodes_color = ['red' if node in nodesSP else 'blue'  for node in G.nodes]
edges_color = ['red' if edge in edgesSP else 'black' for edge in G.edges]

# plot axis
###############
plt.close('all')
fig, ax1 = plt.subplots()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title("Graph nº Nodes: {}".format(nX*nY))

# plot axis
###############
nx.draw_networkx(G, P, node_color=nodes_color, edge_color=edges_color, font_size=8, font_color='w', node_size=150)

nx.draw_networkx_edges(G, P)

nx.draw_networkx_edge_labels(G, pos=P, edge_labels=W, font_size=8)

plt.show()
