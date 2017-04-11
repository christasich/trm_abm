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
from scipy import ndimage

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

X = 50
Y = 30
dx = 1

polderZ = np.zeros((Y,X),dtype=float)

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

time = 5 # in years
gs = 0.03
ws = ((gs/1000)**2*1650*9.8)/0.018
rho = 700
SSC = 0.4
dP = 0
dO = 0
z0 = 0

breachX = 0
breachY = Y/2
    
breachX_dist = np.zeros((Y,X),dtype=float)
breachY_dist = np.zeros((Y,X),dtype=float)
D = np.zeros((Y,X),dtype=float)
DNorm = np.zeros((Y,X),dtype=float)

count = 0
for t in range(time):
    A = (trm.delta_z(tides.pressure,tides.index,ws,rho,SSC,dP,dO,
            polderZ[breachY,breachX])-polderZ[breachY,breachX])
    for i in range(Y):
        for j in range(X):
            if count == 0:
                breachX_dist[i,j] = abs(j - breachX)
                breachY_dist[i,j] = abs(i - breachY)
                D[i,j] = np.hypot(breachX_dist[i,j],breachY_dist[i,j])*dx
                DNorm[i,j] = D[i,j]/1000 + 1
            polderZ[i,j] = polderZ[i,j] + (DNorm[i,j] ** -1.3609) * A/DNorm[i,j]
    count = count + 1

#==============================================================================
# DEFINE HOUSEHOLDS
#==============================================================================

# not sure this makes a big difference
growth_kernels = """
010 000 010 111
111 111 010 111
010 000 010 111
"""

def patches(shape, N, maxiter=100):
    # seed patches leave a gap of N between 0 and the first patch
    out = np.zeros(shape, int)
    out.ravel()[np.random.choice(out.size, N)] = np.arange(N+1, 2*N+1)
    # unpack and transform kernels to values
    # 0   -- centre, impossible to unseat anything other than zero
    # -N  -- the former 1s, can compete for empty (zero) squares
    # -2N -- the former 0s, leave the square alone
    kernels = np.array([[int(s, 2) for s in l.strip().split()]
                        for l in growth_kernels.split('\n')
                        if l.strip()], np.uint8)
    kernels = np.unpackbits(kernels).reshape(kernels.shape + (-1,))
    ke = kernels.shape[0]
    assert np.all(kernels[..., :-ke] == 0)
    kernels = (kernels[..., -ke:].astype(int).swapaxes(0, 1) - 1) * N - N
    kernels[..., ke//2, ke//2] = 0
    # shuffle labels after each iteration, so larger numbers do not get
    #  a systematic advantage
    shuffle = np.arange(4*N+1)
    # also map negative labels to zero
    shuffle[2*N+1:] = 0
    for j in range(maxiter):
        # pick one of the kernels
        k = np.random.randint(0, kernels.shape[0])
        # grow patches
        out = ndimage.grey_dilation(
            out, kernels.shape[1:], structure=kernels[k], mode='constant')
        # shuffle
        shuffle[N+1:2*N+1] = np.random.permutation(shuffle[N+1:2*N+1])
        # newly acquired territory will be labelled 1--N lift that to N+1--2N
        shuffle[1:N+1] = shuffle[N+1:2*N+1]
        out = shuffle[out]
        if np.all(out):
            break
    return out - N - 1

N = 50
polderHH = patches((Y, X), N)
HHs = dict.fromkeys(np.unique(polderHH))
fn = ['mean z','wealth']
HHown = dict.fromkeys(fn)
HHs = [dict.fromkeys(fn) for HH in HHs]
#[a for a in range(50) if a not in HHs]

#==============================================================================
# CALCULATE MEAN ELEVATION FOR HOUSEHOLDS
#==============================================================================

for HH in HHs:
    y, x = np.where(polderHH == HH)
    z = np.zeros(len(x),dtype=float)
    for i in range(len(x)):
        z[i] = polderZ[y[i],x[i]]
    HHs[HH] = {'mean z':np.mean(z)}
        

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================
#
## Expected Utility
#def eu(w,t,p,r):
#    u = (w + t * p) * (1 - r)**t
#    return u