# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:35:56 2023

@author: Genio
"""


import numpy as np

class NeuralNetwork:
    
    def __init__(self,network):
        
        self.network = network
        self.L       = len(network)
        self.W       = [ np.random.rand(rows,cols) for rows,cols in zip(network[1:],network[:-1]) ]
        self.B       = [ np.random.rand(rows,1   ) for rows      in network[1:]                   ]
        
    def feedForward(self, a):
        
        for W,b in zip(self.W,self.B):
             
            z =  W @ a + b
            a = self.sigmoid( z )
         
        return a
        
    def sigmoid(self,zVec):
        
        return 1/(1+np.exp(-zVec))
    


    
layers = [2,3,4,5]
inp1   = np.random.rand( layers[0], 1)
net1   = NeuralNetwork(layers)
out1   = net1.feedForward(inp1)




def fun1(x,y):
    
    return x**2+y**2


delta = 1e-6

x = 2
y = 2

dfx = (fun1(x+delta,y)-fun1(x,y))/delta