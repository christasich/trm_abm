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
import matplotlib.pyplot as plt

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

X = 500
Y = 300
dx = 10

grid = np.zeros((Y,X),dtype=float)

#
#alpha = 0.5
#wx = 2 * np.pi / ((X-1) * 2 - 2)
#wy = 2 * np.pi / ((Y-1) * 2 - 2)
#
#for i in range(X-2):
#    for j in range(Y-2):
#        intZ[i,j] = alpha - alpha * np.sin(wx*i) * np.sin(wy*j) + rd.uniform(
#                0,0.01)
        
#==============================================================================
# TIDAL RIVER MANAGEMENT
#==============================================================================

time = 2 # in years
gs = 0.03
ws = ((gs/1000)**2*1650*9.8)/0.018
rho = 700
SSC = 0.4
dP = 0
dO = 0
z0 = 0

breachX = 0
breachY = Y/2
    
breach_dist_X = np.zeros((Y,X),dtype=float)
breach_dist_Y = np.zeros((Y,X),dtype=float)
D = np.zeros((Y,X),dtype=float)
DNorm = np.zeros((Y,X),dtype=float)

count = 0
for t in range(time):
    A = (trm.delta_z(tides.pressure,tides.index,ws,rho,SSC,dP,dO,
            grid[breachY,breachX])-grid[breachY,breachX])
    for i in range(Y):
        for j in range(X):
            if count == 0:
                breach_dist_X[i,j] = abs(j - breachX)
                breach_dist_Y[i,j] = abs(i - breachY)
                D[i,j] = np.hypot(breach_dist_X[i,j],breach_dist_Y[i,j])*dx
                DNorm[i,j] = D[i,j]/1000 + 1
            grid[i,j] = grid[i,j] + (DNorm[i,j] ** -1.3609) * A/DNorm[i,j]
    count = count + 1

#==============================================================================
# DEFINE HOUSEHOLDS
#==============================================================================
minsize = 100
maxsize = 500
polderHH = np.zeros((X,Y),dtype=int)

HH = 1
for i in range(X):
    for j in range(Y):
        randSize = rd.randint(minsize,maxsize)
        if polderHH[i,j] == 0:
            polderHH[i,j] = HH
            for ri in range(randSize+1):
                for rj in range(randSize+1):
                    try:
                        if polderHH[i+ri,j+rj] == 0:
                            polderHH[i+ri,j+rj] = HH
                    except:
                        pass
                    try:
                        if polderHH[abs(i-ri),abs(j-rj)] == 0:
                            polderHH[abs(i-ri),abs(j-rj)] = HH
                    except:
                        pass
            HH = HH + 1
#
##==============================================================================
## UTILITY FUNCTIONS
##==============================================================================
#
## Expected Utility
#def eu(w,t,p,r):
#    u = (w + t * p) * (1 - r)**t
#    return u