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
from scipy import ndimage

#==============================================================================
# LOAD TIDES
#==============================================================================

def load_tides(file,parser,start,end):
    df = pd.read_csv(file,parse_dates=['datetime'],date_parser=parser,index_col='datetime')
    df1 = df[(df.index >= start) & (df.index < end) & (df.index.minute == 0)]
    df2 = df1['pressure'] - np.mean(df1['pressure'])    
    return df2

#==============================================================================
# CHANGE IN ELEVATION
#==============================================================================

def delta_z(heads,time,ws,rho,SSC,dP,dO,z0):
    C0 = np.zeros(len(heads))
    C = np.zeros(len(heads))
    dz = np.zeros(len(heads))
    dh = np.zeros(len(heads))
    z = np.zeros(len(heads)+1)
    z[0:2] = z0
    dt = float((time[1]-time[0]).seconds)
    j = 1
    for h in heads[1:]:
        dh[j] = (h-heads[j-1])/dt
        C0[j] = 0
        if h > z[j]:
            if dh[j] > 0:
                C0[j] = 0.69*SSC*(h-z[j])
                C[j] = (C0[j]*(h-heads[j-1])+C[j-1]*(h-z[j]))/(2*h-heads[j-1]-z[j]+ws/dt)
            else:
                C[j] = (C[j-1]*(h-z[j]))/(h-z[j]+ws/dt)
        else:
            C[j] = 0
        dz[j] = (ws*C[j]/rho)*dt
        z[j+1] = z[j] + dz[j] + dO - dP
        j = j + 1
    z = z[-1]
    return (z)

#==============================================================================
# BUILD ENVIRONMENT
#==============================================================================

def build_polder(x,y,alpha):
    X = np.arange(x)
    Y = np.arange(y)
    wx = 2 * np.pi / (x * 2)
    wy = 2 * np.pi / (y * 2)
    xx,yy = np.meshgrid(X,Y)
    z = alpha - alpha * np.sin(xx*wx) * np.sin(yy*wy) + np.random.uniform(0,0.1,(y,x))
    return z,xx,yy

#==============================================================================
# BUILD HOUSEHOLDS
#==============================================================================

def build_households(shape,N,maxiter=100):
    growth_kernels = """
    555555555 543212345
    444444444 543212345
    333333333 543212345
    222222222 543212345
    111101111 543202345
    222222222 543212345
    333333333 543212345
    444444444 543212345
    555555555 543212345
    """
    # load kernels
    kernels = np.array([[[int(d) for d in s] for s in l.strip().split()]
                        for l in growth_kernels.split('\n')
                        if l.strip()], np.int)
    nlev = np.max(kernels) + 1
    # special case for binary kernels
    if nlev == 2:
        kernels = 2 - kernels
        nlev = 3
    kernels = -kernels.swapaxes(0, 1) * N
    key, kex = kernels.shape[1:]
    kernels[:, key//2, kex//2] = 0
    # seed patches leave a gap between 0 and the first patch
    out = np.zeros(shape, int)
    out.ravel()[np.random.choice(out.size, N)] = np.arange((nlev-1)*N+1, nlev*N+1)
    # shuffle labels after each iteration, so larger numbers do not get
    # a systematic advantage
    shuffle = np.arange((nlev+1)*N+1)
    # also map negative labels to zero
    shuffle[nlev*N+1:] = 0
    shuffle_helper = shuffle[1:nlev*N+1].reshape(nlev, -1)
    for j in range(maxiter):
        # pick one of the kernels
        k = np.random.randint(0, kernels.shape[0])
        # grow patches
        out = ndimage.grey_dilation(
            out, kernels.shape[1:], structure=kernels[k], mode='constant')
        # shuffle
        shuffle_helper[...] = np.random.permutation(
            shuffle[(nlev-1)*N+1:nlev*N+1])
        out = shuffle[out]
        if np.all(out):
            break
    return out % N
#==============================================================================
# CALCULATE WATER LOGGING PARAMETER
#==============================================================================

# Logit Function
def logit(z,k,mid):
    x = 1/(1+np.e**(-k*(z-mid)))
    return x

#==============================================================================
# UPDATE PROFIT
#==============================================================================

def update_profit(Z,n,max_profit):
    profit = np.zeros_like(Z)
    profit[Z >= n] = max_profit

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================

# Expected Utility
def eu(w,t,p,r):
    u = (w + t * p) * (1 - r)**t
    return u