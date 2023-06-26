# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:16:01 2023

@author: Genio
"""
    
import numpy as np
from utils import roundGrid,moving_average
import matplotlib.pyplot as plt

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






xx = np.array([0.2,0.5,0.8,1.3,1.7,2.2,2.8,3.5])
xx = np.linspace(0,4,50)
yy = xx**2
slope = np.diff(yy)/np.diff(xx)
slope0 = np.insert(slope, 0, 0, axis=0)

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot()
ax.grid();ax.set_xlabel('X');ax.set_ylabel('Z')
ax.plot(xx,yy,'r',linewidth=1,marker='+')



ax2 = ax.twinx()
ax2.plot(xx ,slope0,'b',linewidth=1,marker='o')
ax2.plot(xx[np.arange(len(slope))] ,slope,'b--',linewidth=1,marker='+')


plt.show()


array = np.array([5,3,8,10,2,1,5,1,0,2])
ma    = moving_average(array, 2)