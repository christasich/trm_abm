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
import squarify as sq
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
# DEFINE CLASSES
#==============================================================================

def extract_and_collapse(dc, p):
    x = dc[p[0]:p[1],p[2]:p[3]]
    x = x.reshape((x.shape[0] * x.shape[1], x.shape[3]))
    return x

class household(object):
    def __init__(self, id, wealth = 0, plots = None, discount = 0.03):
        self.id = id
        self.wealth = wealth
        self.discount = discount
        if self.plots is None:
            self.plots = np.array([], dtype=np.integer)
        else:
            self.plots = np.array(plots, dtype=np.integer)

    def utility(self, profit_dc):
        own_patches = np.concatenate([ extract_and_collapse(profit_dc, p) for p in self.plots ],
                                      axis = 0)
        profit = np.sum(own_patches, axis = 0)
        eu = self.wealth, np.sum(profit * np.exp(- self.discount * np.arange(len(profit))))
        return eu

class polder(object):
    def __init__(self, x, y, alpha, n_households = 0):
        self.width = x
        self.height = y
        wx = np.pi / x
        wy = np.pi / y
        self.elevation = alpha * ((1.0 - np.outer(np.sin(np.arange(y) * wy),
                                                  np.sin(np.arange(x) * wx))) +
                          np.random.normal(0.0, 0.1, (y,x)))
        self.owners = np.zeros((self.height,self.width), dtype = np.integer)
        if n_households == 0:
            self.households = []
            self.owners = np.zeros((x, y), dtype = np.integer)
        else:
            self.households = self.build_households(n_households)
            self.owners = np.zeros((x,y), dtype = np.integer)
            for hh in self.households:
                for p in hh.plots:
                    self.owners[p[0]:p[1],p[2]:p[3]] = hh.id


    def build_households(self, n, gini = 0.3):
        if len(self.households) != n:
            self.households = [household(0.0) for i in range(n)]
        if isinstance(gini, dict):
            gini_land = gini['land']
            gini_wealth = gini['wealth']
        elif isinstance(gini, (list,tuple)):
            if (len(gini) > 1):
                gini_wealth = gini[0]
                gini_land = gini[1]
            else:
                gini_wealth = gini_land = gini[0]
        else:
            gini_wealth = gini_land = gini

        alpha = (1.0 / gini_land + 1.0) / 2.0
        plot_sizes = sq.normalize_sizes(np.random.pareto(alpha, size = n),
                                        self.width, self.height)
        plots = sq.squarify(plot_sizes, 0, 0, self.width, self.height)
        plots = pd.DataFrame(plots, columns = ('x', 'y', 'dx', 'dy'))
        plots['xf'] = plots['x'] + plots['dx']
        plots['yf'] = plots['y'] + plots['dy']
        plots = plots[['x','xf','y','yf']]
        plots = np.array(np.round(plots), dtype = np.integer)
        for i in range(n):
            p = plots[i]
            self.households[i].plots = p
            self.owners[p[0]:p[1],p[2]:p[3]] = i








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

def update_profit(Z,n,best_z,max_profit):
    profit = np.zeros_like(Z)
    z_ratio = Z/best_z
    profit[Z >= n] = z_ratio[Z >= n] * max_profit
    return profit

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================

# Expected Utility
def eu(w,t,p,r):
    u = (w + p) * (1 - r)
    return u