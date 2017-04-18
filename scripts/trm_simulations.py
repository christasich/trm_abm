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
MW = np.mean(tides)
HW = pressure[argrelextrema(pressure, np.greater)[0]]
LW = pressure[argrelextrema(pressure, np.less)[0]]
MHW = np.mean(HW)
MLW = np.mean(LW)

#==============================================================================
# GENERATE POLDER ENVIRONMENT
#==============================================================================

X = 500 # X size of polder
Y = 300 # Y size of polder
dx = 1 # spatial step
dt = 1
alpha = 0.5 # maximum elevation at edges of polder for initial elevation

Z,xx,yy = abm.build_polder(X,Y,alpha)

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

mean_breach_dist = np.zeros(N)

profit = np.zeros((Y,X))

df = pd.DataFrame()

#==============================================================================
# TIDAL RIVER MANAGEMENT
#==============================================================================

time = 1 # in years
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

breach = 0
time_horizon = 1

# Calculate mean elevation and initial wealth of household parcels
for hh in range(N):
    df.set_value(hh,'elevation',np.mean(Z[polderHH == hh]))
    df.set_value(hh,'wealth',mean_z[hh]/alpha*max_wealth)

Z_breach = Z[breachY,breachX]
# Run simulation for t time
for t in range(time):
    z = abm.delta_z(tides.repeat(time_horizon),tides.index,ws,rho,SSC,dP,dO,Z_breach)
    z1 = z[8759]
    A1 = z1 - Z_breach
    Z1 = Z_breach + (D ** -1.3) * A1/D
    z5 = z[-1]
    A5 = z5 - Z[breachY,breachX]
    Z5 = Z_breach + (D ** -1.3) * A5/D
    wl_risk = (1 - abm.logit(Z1,5,1))
    flood_risk = (1 - abm.logit(Z5,5,MHW/2))
    profit = abm.logit(Z1,5,MHW/2) * max_profit
    profit_trm = abm.logit(Z5,5,MHW/2) * max_profit
    for hh in range(N):
        eu_base = abm.eu(df.loc[hh].wealth,time_horizon,
                         np.mean(profit[polderHH==hh]),np.mean(wl_risk[polderHH==hh]))
        eu_trm = abm.eu(df.loc[hh].wealth,time_horizon,
                         np.mean(profit_trm[polderHH==hh]),
                               np.mean(flood_risk[polderHH==hh]))
        if eu_base > eu_trm:
            df.set_value(hh,'vote',0)
        else:
            df.set_value(hh,'vote',1)
        df.set_value(hh,'elevation',np.mean(Z1[polderHH == hh]))
    Z_breach = z1
    
#==============================================================================
# DIAGNOSTIC PLOTS
#==============================================================================

# water logging
#plt.imshow(WL,cmap='RdYlBu')
#
## Elevations with household parcels
#plt.figure()
#plt.imshow(polderZ,cmap='gist_earth')
#plt.contour(polderHH,colors='black',linewidths=0.5)

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