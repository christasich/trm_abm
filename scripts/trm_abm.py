# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:48:23 2017

@author: Chris Tasich
"""
#==============================================================================
# IMPORT PACKAGES
#==============================================================================

import numpy as np
import pandas as pd
import trm
import random as rd

#==============================================================================
# LOAD TIDES
#==============================================================================

file = '../data/p32_tides.dat'
parser = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
start = pd.datetime(2015,5,15,1)
end = pd.datetime(2016,5,14,1)

tides = trm.load_tides(file,parser,start,end)

#==============================================================================
# DEFINE ENVIRONMENT
#==============================================================================

X = 21
Y = 21
EBH = 3
grid = np.zeros((X,Y),dtype=float)
cell_area = 100 # in meters
time = 50 # in years


grid[0,:] = EBH
grid[:,0] = EBH
grid[X-1,:] = EBH
grid[:,Y-1] = EBH

#alpha = 0.5
#wx = 2 * np.pi / (X * 2 - 2)
#wy = 2 * np.pi / (Y * 2 - 2)
#
#for i in range(X):
#    for j in range(Y):
#        grid[i,j] = alpha - alpha * np.sin(wx*i) * np.sin(wy*j) + rd.uniform(
#                0,0.01)

#==============================================================================
# TIDAL RIVER MANAGEMENT
#==============================================================================

gs = 0.03
ws = ((gs/1000)**2*1650*9.8)/0.018
rho = 700
SSC = 0.4
dP = 0
dO = 0
z0 = 0

breach_X = 10
breach_Y = 20
grid[breach_Y,breach_X] = 0

breach_dist = np.zeros((X,Y),dtype=float)
breach_dist_X = np.zeros((X,Y),dtype=float)
breach_dist_Y = np.zeros((X,Y),dtype=float)
D = np.zeros((X,Y),dtype=float)

count = 0
for t in range(time):
    grid[breach_Y,breach_X] = trm.delta_z(tides.pressure,tides.index,ws,rho,SSC,dP,dO,
            grid[breach_Y,breach_X])
    for i in range(X)[1:-1]:
        for j in range(Y)[1:-1]:
            if count == 0:
                breach_dist_X[i,j] = abs(i - breach_Y)
                breach_dist_Y[i,j] = abs(j - breach_X)
                breach_dist[i,j] = np.sqrt(breach_dist_X[i,j]**2 + breach_dist_Y[i,j]**2)
                D[i,j] = breach_dist[i,j] # in km
            grid[i,j] = (D[i,j] ** -1.3609) * grid[breach_Y,breach_X]
    grid[breach_Y-1:breach_Y+1,breach_X-1:breach_X+1] = np.min(grid)
    count =+ 1
    
polder = grid[1:X-1,1:Y-1]

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================

# Expected Utility
def eu(w,t,p,r):
    u = (w + t * p) * (1 - r)**t
    return u