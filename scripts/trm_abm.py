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

#file = '../data/p32_tides.dat'
#parser = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
#start = pd.datetime(2015,5,15,1)
#end = pd.datetime(2016,5,14,1)
#
#tides = trm.load_tides(file,parser,start,end)

#==============================================================================
# DEFINE ENVIRONMENT
#==============================================================================

X = 101
Y = 101
EBH = 3
grid = np.zeros((X,Y),dtype=float)
polder = grid[1:X-1,1:Y-1]
cell_area = 100 # in meters
time = 50 # in years


grid[0,:] = EBH
grid[:,0] = EBH
grid[X-1,:] = EBH
grid[:,Y-1] = EBH
    
minsize = 5    
maxsize = 20
polderHH = np.zeros((X-1,Y-1),dtype=int)

HH = 1
HHx = np.zeros(1000)
HHy = np.zeros(1000)
HHr = np.zeros(1000)
for i in range(X-1):
    for j in range(Y-1):
        randSize = rd.randint(minsize,maxsize)
        if polderHH[i,j] == 0:
            HHx[HH] = i
            HHy[HH] = j
            HHr[HH] = randSize
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
                          

#HH = 1
#while 0 in polderHH:    
#    HHx = rd.randint(0,X-2)
#    HHy = rd.randint(0,Y-2)
#    HHr = rd.randint(2,maxrad)
#    if polderHH[HHx,HHy] == 0:
#        polderHH[HHx,HHy] = HH
#        for rx in range(HHr+1):
#            for ry in range(HHr+1):
#                try:
#                    if polderHH[HHx+rx,HHy+ry] == 0:
#                        polderHH[HHx+rx,HHy+ry] = HH
#                except:
#                    pass
#                try:
#                    if polderHH[abs(HHx-rx),abs(HHy-ry)] == 0:
#                        polderHH[abs(HHx-rx),abs(HHy-ry)] = HH
#                except:
#                    pass
#            HH = HH + 1
        
    

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

#gs = 0.03
#ws = ((gs/1000)**2*1650*9.8)/0.018
#rho = 700
#SSC = 0.4
#dP = 0
#dO = 0
#z0 = 0
#
#breach_X = 10
#breach_Y = 20
#grid[breach_Y,breach_X] = 0
#    
#
#breach_dist = np.zeros((X,Y),dtype=float)
#breach_dist_X = np.zeros((X,Y),dtype=float)
#breach_dist_Y = np.zeros((X,Y),dtype=float)
#D = np.zeros((X,Y),dtype=float)
#
#count = 0
#for t in range(time):
#    grid[breach_Y,breach_X] = trm.delta_z(tides.pressure,tides.index,ws,rho,SSC,dP,dO,
#            grid[breach_Y,breach_X])
#    for i in range(X)[1:-1]:
#        for j in range(Y)[1:-1]:
#            if count == 0:
#                breach_dist_X[i,j] = abs(i - breach_Y)
#                breach_dist_Y[i,j] = abs(j - breach_X)
#                breach_dist[i,j] = np.sqrt(breach_dist_X[i,j]**2 + breach_dist_Y[i,j]**2)
#                D[i,j] = breach_dist[i,j] # in km
#            grid[i,j] = (D[i,j] ** -1.3609) * grid[breach_Y,breach_X]
#    grid[breach_Y-1:breach_Y+1,breach_X-1:breach_X+1] = np.min(grid)
#    count =+ 1
#    
#polder = grid[1:X-1,1:Y-1]

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================

# Expected Utility
def eu(w,t,p,r):
    u = (w + t * p) * (1 - r)**t
    return u