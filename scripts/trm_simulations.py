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
LW = pressure[argrelextrema(pressure, np.less)[0]]
MHW = np.mean(HW)
MLW = np.mean(LW)

#==============================================================================
# GENERATE POLDER ENVIRONMENT
#==============================================================================

X = 500 # X size of polder
Y = 300 # Y size of polder
dx = 1 # spatial step
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
profit = np.zeros(N)
df = pd.DataFrame(data={'elevation':mean_z,'wealth':wealth,'profit':profit,
                           'distance_to_breach':mean_breach_dist})

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
time_horizon = 5
water_log = abm.logit(Z,5,MLW/2)
profit = water_log*max_profit
flood_risk = abm.logit(Z,5,MHW/2)

# Calculate mean elevation and initial wealth of household parcels
for hh in range(N):
    df.set_value(hh,'elevation',np.mean(Z[polderHH == hh]))
    df.set_value(hh,'wealth',mean_z[hh]/alpha*max_wealth)
    df.set_value(hh,'profit',np.sum(profit[polderHH==hh]))
    df.set_value(hh,'flood_risk',np.sum(flood_risk[polderHH==hh]))
    df.set_value(hh,'utility',abm.eu(df.loc[hh].wealth,time_horizon,
                                     df.loc[hh].profit,df.loc[hh].flood_risk))
    

# Run simulation for t time
for t in range(time):
    if breach == 0:
        print 'No breach'
    else:
        z = abm.delta_z(tides,tides.index,ws,rho,SSC,dP,dO,
                    Z[breachY,breachX])
        A = z - Z[breachY,breachX]
        Z = Z + (D ** -1.3) * A/D
        water_log = abm.logit(Z,5,MLW/2)
        flood_risk = abm.logit(Z,5,MHW/2)
    for hh in range(N):
        df.set_value(hh,'elevation',np.mean(Z[polderHH == hh]))
        df.set_value(hh,'profit',np.sum(profit[polderHH==hh]))
        df.set_value(hh,'wealth',df.loc[hh].wealth + df.loc[hh].profit)
        
        
        
        df.set_value(hh,'expected utility',abm.eu(df.loc[hh].wealth,5,df.loc[hh].profit,
                                         0.05))
        df.set_value(hh,'new elevation',np.mean(Z[polderHH == hh]))
        df.set_value(hh,'expected profit',np.sum(WL[polderHH==hh]*max_profit))
        df.set_value(hh,'expected utility',abm.eu(df.loc[hh].wealth,5,df.loc[hh].profit,
                                         0.05))

    
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