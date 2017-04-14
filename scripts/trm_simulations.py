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
from scipy.signal import argrelextrema

#==============================================================================
# SELECT TIMEFRAME AND LOAD TIDES
#==============================================================================

file = '../data/p32_tides.dat'
parser = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
start = pd.datetime(2015,5,15,1)
end = pd.datetime(2016,5,14,1)

tides = abm.load_tides(file,parser,start,end)

# Calculate Mean High Water
pressure = tides.as_matrix()
HW = pressure[argrelextrema(pressure, np.greater)[0]]
MHW = np.mean(HW)

#==============================================================================
# GENERATE POLDER ENVIRONMENT
#==============================================================================

X = 500 # X size of polder
Y = 300 # Y size of polder
dx = 1 # spatial step
alpha = 0.5 # maximum elevation at edges of polder for initial elevation

polderZ,xx,yy = abm.build_polder(X,Y,alpha)

#==============================================================================
# GENERATE HOUSEHOLD PARCELS AND INITIALIZE PARAMETERS
#==============================================================================

N = 50 # number of households

polderHH = abm.build_households((Y, X), N)

# Initialize dataframe of household paramters
max_wealth = 10000 # initial max wealth in Taka
max_profit = 100 # max profit per 1 m^2 land in Taka

mean_z = np.zeros(N)
wealth = np.zeros(N)

# Calculate mean elevation and initial wealth of household parcels
for hh in range(N):
    mean_z[hh] = np.mean(polderZ[polderHH == hh])
    wealth[hh] = mean_z[hh]/alpha*max_wealth

profit = np.zeros(N)
HHdf = pd.DataFrame(data={'elevation':mean_z,'wealth':wealth,'profit':profit})

#==============================================================================
# TIDAL RIVER MANAGEMENT
#==============================================================================

time = 20 # in years
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
    polderZ = polderZ + (D ** -1.3) * A/D

#==============================================================================
# CALCULATE PATCH VARIABLES
#==============================================================================

# Water logged parameter (logit function)

k = 5
mid = MHW-.5

WL = 1/(1+np.e**(-k*(polderZ-mid)))

#==============================================================================
# DIAGNOSTIC PLOTS
#==============================================================================

# water logging
plt.imshow(WL,cmap='RdYlBu')

# Elevations with household parcels
plt.figure()
plt.imshow(polderZ,cmap='gist_earth')
plt.contour(polderHH,colors='black',linewidths=0.5)

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