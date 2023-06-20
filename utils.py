# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:15:33 2023

@author: Genio
"""


import numpy as np
import networkx as nx

def gridGraph(Z,x,y,cellSize):
    
    
    G=nx.Graph()
    
    nY,nX = Z.shape
    
    X,Y = np.meshgrid(x,y) 
    Y = np.flipud(Y)
    
    zShape = Z.shape
    
    ## rows cols  - 1
    #####################
    for i in range(nY-1):
        for j in range(nX-1):
            
            # a -- b
            # | \/
            # c    d
            # G.add_weighted_edges_from([(n1,n2,weight)])
            
            a = np.ravel_multi_index((i,j), zShape)
            b = np.ravel_multi_index((i,j+1), zShape)
            c = np.ravel_multi_index((i+1,j), zShape)
            d = np.ravel_multi_index((i+1,j+1), zShape)
            
            
            G.add_node(a,pos=(X[i,j],Y[i,j]))
            G.add_node(b,pos=(X[i,j+1],Y[i,j+1]))
            G.add_node(c,pos=(X[i+1,j],Y[i+1,j]))
            G.add_node(d,pos=(X[i+1,j+1],Y[i+1,j+1]))
            
            weight_ab = Z[i,j] + Z[i,j+1]
            G.add_weighted_edges_from([(a,b,weight_ab)])
            
            weight_ac = Z[i,j] + Z[i+1,j]
            G.add_weighted_edges_from([(a,c,weight_ac)])
            
            weight_ad = Z[i,j] + Z[i+1,j+1]
            G.add_weighted_edges_from([(a,d,weight_ad)])
            
            weight_cd = Z[i+1,j] + Z[i,j+1]
            G.add_weighted_edges_from([(c,b,weight_cd)])
            
            
    ## last row
    ###############
    i = nY-1
    for j in range(nX-1):
    
        # c -- d
        
        c = np.ravel_multi_index((i,j), zShape)
        d = np.ravel_multi_index((i,j+1), zShape)
    
        weight_cd = Z[i,j] + Z[i,j+1]
        G.add_weighted_edges_from([(c,d,weight_cd)])
        
        
    ## last column
    ##################
    j = nX-1
    for i in range(nY-1):
    
        # b
        # |
        # d
        
        b = np.ravel_multi_index((i,j), zShape)
        d = np.ravel_multi_index((i+1,j), zShape)
    
        weight_bd = Z[i,j] + Z[i+1,j]
        G.add_weighted_edges_from([(b,d,weight_bd)])
                      
    return G



def linearIndx():
    
    a = np.random.randint(1,9, size=(3,4))
    
    # sub2ind
    sub1  = np.array([[0,1,2],[0,2,1]])
    ind1  = np.ravel_multi_index(sub1, a.shape)
    
    # ind2sub
    ind2 = np.array([2,5])
    sub2 = np.unravel_index(ind2, a.shape)
    
    return (ind1,sub2)


def roundGrid(x,cellSize):
    
    return np.round(x/cellSize)*cellSize
    
    

# import numpy as np

# a = np.arange(15).reshape(3, 5)
# b = np.arange(1,20,2)
# c = np.zeros(a.shape)
# d = np.linspace(1,8,5)
# e = np.random.rand(3,2)
# f = np.random.randint(1,9, size=(3,3))

# x = np.arange(8)
# y = np.arange(10)
# X,Y = np.meshgrid(x,y)

# m1 = np.ones((2,3))*5
# m2 = np.ones((2,3))*7

# mH = np.hstack((m1,m2))
# mV = np.vstack((m1,m2))

