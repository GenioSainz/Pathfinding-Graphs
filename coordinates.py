# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:16:01 2023

@author: Genio
"""
    
import numpy as np
from utils import roundGrid


cellSize = 5

x   = np.arange(10,55,cellSize)
y   = np.arange(60,95,cellSize)
X,Y = np.meshgrid(x,y) 
a   = np.arange(y.size*x.size).reshape(y.size,x.size)

print('################')
print('Mat a \n',a,'\n')

# sub2ind          rows    cols
sub1  = np.array([[0,0,5],[0,5,0]])
ind1  = np.ravel_multi_index(sub1, a.shape)

print('################')
print('ind1 \n',ind1,'\n')

# ind2sub       index
ind2 = np.array([2,29,60])
sub2 = np.unravel_index(ind2, a.shape)
print('################')
print('sub2 \n',sub2,'\n')


# subs to coordinates
xi = X[sub2[0],sub2[1]]
yi = Y[sub2[0],sub2[1]]


x_input = 33
y_input = 67

x_input_round = roundGrid(33,cellSize)
y_input_round = roundGrid(67,cellSize)


f = np.random.randint(1,10, size=(3,5))

gY,gX = np.gradient(f,cellSize)






