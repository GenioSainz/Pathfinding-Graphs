# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:15:33 2023

@author: Genio
"""

import numpy as np
import networkx as nx

def gridGraphINDX(Z,x,y):
    
    G=nx.Graph()
    
    nY,nX = Z.shape
    
    X,Y = np.meshgrid(x,y) 
    Y   = np.flipud(Y)
    
    zShape = Z.shape
    
    ## rows cols  - 1
    #####################
    for i in range(nY-1):
        for j in range(nX-1):
            
            # a -- b
            # | \/ |
            # c -- d
            # G.add_weighted_edges_from([(n1,n2,weight)])

            indxa = (i  ,j  )
            indxb = (i  ,j+1)
            indxc = (i+1,j  )
            indxd = (i+1,j+1)
            
            a = np.ravel_multi_index(indxa,zShape)
            b = np.ravel_multi_index(indxb,zShape)
            c = np.ravel_multi_index(indxc,zShape)
            d = np.ravel_multi_index(indxd,zShape)
            
            G.add_node(a,pos=(X[indxa],Y[indxa]))
            G.add_node(b,pos=(X[indxb],Y[indxb]))
            G.add_node(c,pos=(X[indxc],Y[indxc]))
            G.add_node(d,pos=(X[indxd],Y[indxd]))
            
            weight_ab = Z[indxa] + Z[indxb]
            G.add_weighted_edges_from([(a,b,weight_ab)])
            
            weight_ac = Z[indxa] + Z[indxc]
            G.add_weighted_edges_from([(a,c,weight_ac)])
            
            weight_ad = Z[indxa] + Z[indxd]
            G.add_weighted_edges_from([(a,d,weight_ad)])
            
            weight_cd = Z[indxc] + Z[indxd]
            G.add_weighted_edges_from([(c,d,weight_cd)])

            weight_bd = Z[indxb] + Z[indxd]
            G.add_weighted_edges_from([(b,d,weight_bd)])

            weight_cb = Z[indxc] + Z[indxb]
            G.add_weighted_edges_from([(c,b,weight_cb)])
                      
    return G


def gridGraphDIS(X,Y,Z):
    
    zShape = Z.shape
    nY,nX  = Z.shape
    G      = nx.Graph()
    
    ## rows cols  - 1
    #####################
    for i in range(nY-1):
        for j in range(nX-1):
            
            # a -- b
            # | \/ |
            # c -- d
            # G.add_weighted_edges_from([(n1,n2,weight)])
            
            indxa = (i  ,j  )
            indxb = (i  ,j+1)
            indxc = (i+1,j  )
            indxd = (i+1,j+1)
            
            a = np.ravel_multi_index(indxa,zShape)
            b = np.ravel_multi_index(indxb,zShape)
            c = np.ravel_multi_index(indxc,zShape)
            d = np.ravel_multi_index(indxd,zShape)

            G.add_node(a,pos=(X[indxa],Y[indxa]))
            G.add_node(b,pos=(X[indxb],Y[indxb]))
            G.add_node(c,pos=(X[indxc],Y[indxc]))
            G.add_node(d,pos=(X[indxd],Y[indxd]))
            
            weight_ab = distanceAB(indxa,indxb,X,Y,Z)
            G.add_weighted_edges_from([(a,b,weight_ab)])
            
            weight_ac = distanceAB(indxa,indxc,X,Y,Z)
            G.add_weighted_edges_from([(a,c,weight_ac)])
            
            weight_ad = distanceAB(indxa,indxd,X,Y,Z)
            G.add_weighted_edges_from([(a,d,weight_ad)])
            
            # weight_cd = distanceAB(indxc,indxd,X,Y,Z)
            # G.add_weighted_edges_from([(c,b,weight_cd)])

            weight_cb = distanceAB(indxc,indxb,X,Y,Z)
            G.add_weighted_edges_from([(c,b,weight_cb)])
            
                      
    return G


def gridGraph_DIS_SLOPE(X,Y,Z,gX,gY):
    
    zShape = Z.shape
    nY,nX  = Z.shape    
    G      = nx.Graph()

    ## rows cols  - 1
    #####################
    for i in range(nY-1):
        for j in range(nX-1):
            
            # a -- b
            # | \/ |
            # c -- d
            # G.add_weighted_edges_from([(n1,n2,weight)])
            
            indxa = (i  ,j  )
            indxb = (i  ,j+1)
            indxc = (i+1,j  )
            indxd = (i+1,j+1)
            
            vx  = np.array([1,0])
            vy  = np.array([0,1])
            vad = np.array([1,-1])/np.sqrt(2)
            vcb = np.array([1,1]) /np.sqrt(2) 
        
            
            a = np.ravel_multi_index(indxa,zShape)
            b = np.ravel_multi_index(indxb,zShape)
            c = np.ravel_multi_index(indxc,zShape)
            d = np.ravel_multi_index(indxd,zShape)
            
            G.add_node(a,pos=(X[indxa],Y[indxa]))
            G.add_node(b,pos=(X[indxb],Y[indxb]))
            G.add_node(c,pos=(X[indxc],Y[indxc]))
            G.add_node(d,pos=(X[indxd],Y[indxd]))
            
            weight_ab = 0*distanceAB(indxa,indxb,X,Y,Z) + directionalSlope(indxa,gX,gY,vx)
            G.add_weighted_edges_from([(a,b,weight_ab)])
            
            weight_ac = 0*distanceAB(indxa,indxc,X,Y,Z) + directionalSlope(indxa,gX,gY,vy)
            G.add_weighted_edges_from([(a,c,weight_ac)])
            
            weight_ad = 0*distanceAB(indxa,indxd,X,Y,Z) + directionalSlope(indxa,gX,gY,vad)
            G.add_weighted_edges_from([(a,d,weight_ad)])
            
            # weight_cd = 0*distanceAB(indxc,indxd,X,Y,Z) + directionalSlope(indxc,gX,gY,vx)
            # G.add_weighted_edges_from([(c,b,weight_cd)])

            weight_cb = 0*distanceAB(indxc,indxb,X,Y,Z) + directionalSlope(indxc,gX,gY,vcb)
            G.add_weighted_edges_from([(c,b,weight_cb)])
            
                      
    return G


def roundGrid(x,cellSize):
    
    return np.round(x/cellSize)*cellSize


def distanceAB(a,b,X,Y,Z):

    # a = (ia,ja)
    # b = (ib,jb)
    
    return np.sqrt( (X[a]-X[b])**2 + (Y[a]-Y[b])**2 + (Z[a]-Z[b])**2 )


def directionalSlope(indx,gX,gY,vec):
    
    gradient_vec = np.array( [gX[indx],gY[indx]] )
    
    return  np.abs( np.dot(vec,gradient_vec) )
    

def node2coords(nodesSP,X,Y,Z):
    
    # ind2sub       index
    ind = np.array(nodesSP)
    sub = np.unravel_index(ind, X.shape)

    # subs to coordinates
    xi = X[sub]
    yi = Y[sub]
    zi = Z[sub]

    return xi,yi,zi

def coords2node(xi,yi,x,y,cellSize):
    
    x_input_round = roundGrid(xi,cellSize)
    y_input_round = roundGrid(yi,cellSize)
    
    i = np.nonzero( y == y_input_round )[0][0]
    j = np.nonzero( x == x_input_round )[0][0]
    
    sub  = np.array((i,j))
    node = np.ravel_multi_index(sub,(y.size,x.size))
    
    return node




