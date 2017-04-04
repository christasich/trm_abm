# -*- coding: utf-8 -*-
"""

Created on Apr 19, 2016
@author: tasichcm
"""
#==============================================================================
# IMPORT PACKAGES
#==============================================================================
import pandas as pd
import numpy as np

#==============================================================================
# FUNCTIONS
#==============================================================================

# Load and process tidal information
def load_tides(file,parser,start,end):
    df = pd.read_csv(file,parse_dates=['datetime'],date_parser=parser,index_col='datetime')
    df1 = df[(df.index >= start) & (df.index < end) & (df.index.minute == 0)]
    df1.pressure -= np.mean(df['pressure'])    
    return df1

# Calculate chane in elevation over a time period
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