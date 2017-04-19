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
year = 8759
alpha = 0.5 # maximum elevation at edges of polder for initial elevation

Z,xx,yy = abm.build_polder(X,Y,alpha)

#==============================================================================
# GENERATE HOUSEHOLD PARCELS AND INITIALIZE PARAMETERS
#==============================================================================

N = 25 # number of households

polderHH = abm.build_households((Y, X), N)

# Initialize dataframe of household paramters
max_wealth = 10000 # initial max wealth in Taka
max_profit = 100 # max profit per 1 m^2 land in Taka

mean_z = np.zeros(N)
wealth = np.zeros(N)

mean_breach_dist = np.zeros(N)

profit = np.zeros((Y,X))
profit_trm = np.zeros((Y,X))

df = pd.DataFrame()

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

breach = 0
time_horizon = 5

# Calculate mean elevation and initial wealth of household parcels
for hh in range(N):
    df.set_value(hh,'elevation',np.mean(Z[polderHH == hh]))
    df.set_value(hh,'wealth',df.loc[hh].elevation/alpha*max_wealth)


# Run simulation for t time
for t in range(time):
    Z_breach = np.zeros(time_horizon+1)
    Z_breach[0] = Z[breachY,breachX]
    z = np.zeros(time_horizon)
    Z_all= [Z]
    A = np.zeros(time_horizon)
    flood_risk = np.zeros(time_horizon)
    wl_risk = (1 - abm.logit(Z,5,MW/2))
    profit = []
    for tt in range(time_horizon):
        z[tt] = abm.delta_z(tides,tides.index,ws,rho,SSC,dP,dO,Z_breach[tt])
        Z_breach[tt+1] = z[tt]
        A[tt] = z[tt] - Z_breach[tt]
        Z_all.append(Z_breach[tt] + (D ** -1.3) * A[tt]/D)
        flood_risk[tt] = (1 - abm.logit(Z_all[tt],5,MHW/2))
        profit.append(abm.update_profit(Z_all[tt],MHW-0.2,max_profit))
        
        
        
    for hh in range(N):
        eu_base = abm.eu(df.loc[hh].wealth,time_horizon,
                         np.mean(profit[polderHH==hh]),
                               np.mean(wl_risk[polderHH==hh]))
        eu_trm = abm.eu(df.loc[hh].wealth,time_horizon,
                         np.mean(profit_trm[polderHH==hh]),
                               np.mean(flood_risk[polderHH==hh]))
        df.set_value(hh,'utility_base',eu_base)
        df.set_value(hh,'utility_trm',eu_trm)
        if eu_base >= eu_trm:
            df.set_value(hh,'vote',0)
        elif eu_base < eu_trm:
            df.set_value(hh,'vote',1)
        df.set_value(hh,'elevation',np.mean(Z1[polderHH == hh]))
    breach_vote = np.mean(df.vote)
    
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