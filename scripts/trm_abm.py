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
# DEFINE ENVIRONMENT
#==============================================================================

X = 11
Y = 11
grid = np.zeros((X,Y),dtype=float)

alpha = 0.5
wx = 2 * np.pi / (X * 2 -2)
wy = 2 * np.pi / (Y * 2 - 2)

for i in range(X):
    for j in range(Y):
        grid[i,j] = alpha - alpha * np.sin(wx*i) * np.sin(wy*j) + rd.uniform(
                0,0.01)

#==============================================================================
# LOAD TIDES
#==============================================================================

file = '../data/p32_tides.dat'
parser = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
start = pd.datetime(2015,5,15,1)
end = pd.datetime(2016,5,14,1)

tides = trm.load_tides(file,parser,start,end)

#==============================================================================
# TIDAL RIVER MANAGEMENT
#==============================================================================

gs = 0.03
ws = ((gs/1000)**2*1650*9.8)/0.018
rho = 700
SSC = 0.4
dP = 0
dO = 0

breach_X = 50
breach_Y = 100

for i in range(X):
    for j in range(Y):
        grid[i,j] = trm.delta_z(tides.pressure,tides.index,ws,rho,SSC,dP,dO,
            grid[i,j])

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================

# Expected Utility
def eu(w,t,p,r):
    u = (w + t * p) * (1 - r)**t
    return u