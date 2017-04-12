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
import trm_abm as abm
import matplotlib.pyplot as plt

#==============================================================================
# SELECT TIMEFRAME AND LOAD TIDES
#==============================================================================

file = '../data/p32_tides.dat'
parser = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
start = pd.datetime(2015,5,15,1)
end = pd.datetime(2016,5,14,1)

tides = abm.load_tides(file,parser,start,end)

#==============================================================================
# GENERATE POLDER ENVIRONMENT
#==============================================================================

X = 500 # X size of polder
Y = 300 # Y size of polder
dx = 1 # spatial step
alpha = 0.5 # maximum elevation at edges of polder for initial elevation

polderZ,xx,yy = abm.build_polder(X,Y,alpha)

#==============================================================================
# GENERATE HOUSEHOLD PARCELS
#==============================================================================

N = 100 # number of households

polderHH = abm.build_households((Y, X), N)

# Initialize dataframe of household paramters
max_wealth = 10000 # initial max wealth in Taka

wealth = np.random.randint(0,max_wealth,N)
profit = np.zeros(N)
dfHH = pd.DataFrame(index=np.arange(N),data={'wealth':wealth,
                    'profit':profit})

#==============================================================================
# TIDAL RIVER MANAGEMENT
#==============================================================================

time = 5 # in years
gs = 0.03 # grain size in m
ws = ((gs/1000)**2*1650*9.8)/0.018 # settling velocity calculated using Stoke's Law
rho = 700 # dry bulk density in kg/m^2
SSC = 0.4 # suspended sediment concentration in g/L
dP = 0 # compaction
dO = 0 # organic matter deposition

# breach coordinates on polder
breachX = 0
breachY = Y/2

breachX_dist = abs(xx - breachX) # x distance to breach
breachY_dist = abs(yy - breachY) # y distance to breach
breach_dist = np.hypot(breachX_dist,breachY_dist)*dx # distance to breach in m
D = breach_dist / 1000 + 1 # distance to breach in km + 1 km (for scaling purposes)

# Run simulation for t time
for t in range(time):
    Z = abm.delta_z(tides,tides.index,ws,rho,SSC,dP,dO,
                    polderZ[breachY,breachX])
    A = Z - polderZ[breachY,breachX]
    polderZ = polderZ + (D ** -1.3609) * A/D

#==============================================================================
# CALCULATE PATCH VARIABLES
#==============================================================================

# Water logged parameter (logit function)

maxWL = 1
k = 5
mid = 1.5

WL = (1 - maxWL / (1 + np.e ** (-k*(polderZ-mid))))

#==============================================================================
# AGGREGATE BY HOUSEHOLD
#==============================================================================

#for hh in HH.index:
#    y, x = np.where(polderHH == hh)
#    z = np.zeros(len(x),dtype=float)
#    for i in range(len(x)):
#        z[i] = polderZ[y[i],x[i]]
#    meanZ = np.mean(z)
#    HH.set_value(hh,'meanZ',meanZ)
#    HH.set_value(hh,'profit',maxWL / (1 + np.e ** (-k*(meanZ-mid))))
#    
#plt.hist(HH.profit)