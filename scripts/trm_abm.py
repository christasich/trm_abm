# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:48:23 2017

@author: Chris Tasich
"""
#==============================================================================
# IMPORT PACKAGES
#==============================================================================

import numpy as np
import numpy.ma as ma
import pandas as pd
import squarify as sq
# from itertools import izip, count
# from scipy import ndimage


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

def  aggrade_patches(heads,time,ws,rho,SSC,dP,dO,z0, z_breach):
    z = z0.copy()
    C_last = np.zeros_like(z0)
    dt = float((time[1]-time[0]).seconds)
    delta_h = (heads.values[1:] - heads.values[:-1])
    for h, dh in zip(heads[1:], delta_h):
        if h > z_breach:
            delta_z = ma.masked_less_equal(h-z, 0.0)
            delta_z.set_fill_value(0.0)
            if dh > 0:
                # C0 = 0.69 * SSC * delta_z
                C_next = ( delta_z * ( 0.69 * dh * SSC + C_last) ) / (delta_z + dh + ws/dt)
            else:
                C_next = ( C_last * delta_z ) / (delta_z + ws/dt)
        else:
            C_next = ma.array(np.zeros_like(z0), ma.nomask)
        C_last = C_next.filled()
        dz = C_last * ws * dt / rho
        z += dz + dO - dP
        # print "Sum(dz) = ", np.sum(dz), ", Sum(C_last) = ", np.sum(C_last)
    return (z)

#==============================================================================
# CALCULATE WATER LOGGING RISK
#==============================================================================

# Logit Function
def logit(z,k,mid):
    x = 1.0 / (1.0 + np.exp(-k*(z-mid)))
    return x

#==============================================================================
# DEFINE CLASSES
#==============================================================================

class household(object):
    def __init__(self, id, wealth = 0, plots = None, discount = 0.03):
        self.id = id
        self.wealth = wealth
        self.discount = discount
        if plots is None:
            self.plots = np.zeros((0,5), dtype=np.integer)
        else:
            self.plots = np.array(plots, dtype=np.integer)

    def utility(self, profit_dc):
        own_patches_profit = np.concatenate([ self.extract_and_collapse(profit_dc, p) for p in self.plots ],
                                      axis = 0)
        profit = np.sum(own_patches_profit, axis = 0)
        eu = self.wealth + np.sum(profit * np.exp(- self.discount * np.arange(len(profit))))
        return eu

    #==========================================================================
    # EXTRACT A RECTANGULAR SECTION THROUGH A CUBE AND COLLAPSE
    #==========================================================================
    
    # Given a cube dc[z,y,x], extract a rectangula prism in (x,y), that extends
    # through all z-values, then ravel the x and y dimensions to produce a
    # 2D array with rows = z and columns = raveled x,y.

    @staticmethod    
    def extract_and_collapse(dc, p):
        x = dc[:,p[2]:p[3],p[0]:p[1]]
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        return x

class breach(object):
    def __init__(self, pldr, breach_x, breach_y, breach_z):
        self.pldr = pldr
        self.x = breach_x
        self.y = breach_y
        self.z_breach = breach_z
        xx,yy = np.meshgrid(np.arange(pldr.width), np.arange(pldr.height))
        delta_x = xx - breach_x
        delta_y = yy - breach_y
        self.dist = np.hypot(delta_x, delta_y)
        self.scaled_dist = self.dist / 1000. + 1.
        self.A = 0.0
        

class polder(object):
    def __init__(self, x, y, 
                 time_horizon,
                 border_height = 1.0,
                 n_households = 0, 
                 max_wealth = 1.0E4, max_profit = 100.,
                 gini = 0.3):
        self.width = x
        self.height = y
        self.border_height = border_height
        self.max_wealth = max_wealth
        self.max_profit = max_profit
        self.time_horizon = time_horizon
        self.breach_duration = 0
        self.current_period = 0
        self.plots = np.zeros(shape = (0,5), dtype = np.integer)
        self.breaches = []
        self.initialize_elevation()
        self.initialize_hh(n_households)

    def initialize_elevation(self, noise = 0.1):
        wx = np.pi / self.width
        wy = np.pi / self.height
        self.elevation = self.border_height * \
            ( \
             (1.0 - \
               np.outer(np.sin(np.arange(self.height) * wy),
                        np.sin(np.arange(self.width) * wx))) + \
              noise * np.random.normal(0.0, 1.0, (self.height, self.width)) \
            )
        self.elevation_cube = np.zeros((self.time_horizon + 1, self.height, self.width))
        self.elevation_cube[0] = self.elevation
        self.current_period = 0

    def set_elevation(self, elevation, plots, n_households = None):
        if n_households is None:
            n_households = len(self.households)
        self.elevation = elevation
        self.owners = np.zeros_like(self.elevation, dtype = np.integer)
        self.plots = plots
        self.initialize_hh_from_plots(n_households)
        self.elevation_cube = np.zeros((self.time_horizon + 1, self.height, self.width))
        self.elevation_cube[0] = self.elevation
        self.current_period = 0

    def initialize_hh(self, n_households):
        self.owners = np.zeros_like(self.elevation, dtype = np.integer)
        self.households = []
        if n_households > 0:
            self.build_households(n_households)
            self.owners.fill(-1)
            for hh in self.households:
                for p in hh.plots:
                    self.owners[p[1]:(p[1]+p[3]),p[0]:(p[0]+p[2])] = hh.id

    def initialize_hh_from_plots(self, n_households):
        assert max(self.owners) < n_households
        self.households = [household(id = i) for i in range(n_households)]
        self.set_hh_plots()

    def set_households(self, households):
        self.households = households
        self.owners = np.zeros_like(self.elevation, dtype = np.integer)
        self.set_owners_wealth()

    def set_hh_wealth(self, hh):
        z = self.elevation[self.owners == hh.id]
        if z.size == 0:
            hh.wealth = 0
        else:
            hh.wealth = self.max_wealth * np.sqrt(z.size) * z.mean() / self.border_height

    def set_owners_wealth(self):
        self.owners.fill(-1.0)
        for hh in self.households:
            for p in hh.plots:
                self.owners[p[1]:(p[1]+p[3]),p[0]:(p[0]+p[2])] = hh.id
            self.set_hh_wealth(hh)

    def set_hh_plots(self):
        for hh in self.households:
            plots = self.plots[self.plots[:,4] == hh.id]
            hh.plots = plots
        self.set_owners_wealth()

    @staticmethod
    def build_subplots(weights, x0, y0, dx, dy, ix0 = 0):
        plot_sizes = sq.normalize_sizes(weights, dx, dy)
        plots = sq.squarify(plot_sizes, x0, y0, dx, dy)
        plots = pd.DataFrame(plots, columns = ('x', 'y', 'dx', 'dy'))
        plots['dx'] = np.round(plots['x'] + plots['dx']) - np.round(plots['x'])
        plots['dy'] = np.round(plots['y'] + plots['dy']) - np.round(plots['y'])
        plots = plots[['x','y','dx','dy']]
        plots = np.array(np.round(plots), dtype = np.integer)
        plots = np.concatenate( \
                   ( \
                    plots, \
                    np.expand_dims( np.arange(plots.shape[0], dtype=np.integer),
                                   axis = 1)  + int(ix0) \
                   ), \
                 axis = 1)
        return plots
        

    def build_plots(self, weights, n_boxes = 10):
        n = weights.size / n_boxes
        remainder = weights.size % n
        w = np.random.choice(weights, weights.size, False)
        wr = w[0:remainder]
        w = w[remainder:]
        w_list = np.random.choice(w, size = (n_boxes, n), replace = False)
        w_list = [ w_list[i] for i in range(w_list.shape[0])]
        if remainder > 0:
            i_dest = np.random.choice(n_boxes, remainder, replace = True)
            for i, j in enumerate(i_dest):
                w_list[j] = np.append(w_list[j], wr[i])
        grid_weights = [ np.sum(x) for x in w_list ]
        scaled_grid_weights = sq.normalize_sizes(grid_weights, self.width, self.height)
        grid = sq.squarify(scaled_grid_weights, 0, 0, self.width, self.height)
        grid = pd.DataFrame(grid, columns = ('x', 'y', 'dx', 'dy'))
        grid['dx'] = np.round(grid['x'] + grid['dx']) - np.round(grid['x'])
        grid['dy'] = np.round(grid['y'] + grid['dy']) - np.round(grid['y'])
        grid = np.array(np.round(grid), dtype = np.integer)
#        self.grid = grid.copy()
#        self.w_list = w_list
#        self.grid_weights = grid_weights
        
        cum_len = np.cumsum( np.concatenate( (np.zeros((1,)), [len(ww) for ww in w_list[:-1]]) ) )
        
        plot_list = [ self.build_subplots(w_list[i], \
                          grid[i,0], grid[i,1], grid[i,2], grid[i,3],
                          ix0 = cum_len[i]) \
                      for i in range(len(w_list)) ]
#        self.plot_list = plot_list
        plots = np.concatenate(plot_list, axis = 0)
        self.plots = plots

    def build_households(self, n = None, gini = 0.3):
        if n is not None and n != len(self.households):
            print "Initializing", n, "households"
            self.households = [household(id = i) for i in range(n)]
        else:
            print "n = ", type(n), ", ", n, ", length = ", len(self.households)
        if isinstance(gini, dict):
            gini_land = gini['land']
        elif isinstance(gini, (list,tuple)):
            if (len(gini) > 1):
                gini_land = gini[0]
            else:
                gini_land = gini[0]
        else:
            gini_land = gini

        alpha = (1.0 / gini_land + 1.0) / 2.0
        weights = np.random.pareto(alpha, size = len(self.households))

        self.build_plots(weights)
        self.set_hh_plots()

    def calc_profit(self, water_level, k):
        self.profit = self.max_profit * logit(self.elevation_cube, k, water_level / 2.0)

    def calc_eu(self):
        eu = [hh.utility(self.profit) for hh in self.households]
        return eu
    
    def add_breach(self, breach_x, breach_y, duration):
        self.breach_duration = duration,
        self.breaches.append(breach(self, breach_x, breach_y, self.border_height))
    
    def aggrade(self, heads, ws, rho, SSC, dP, dO, period = -1):
        if period < 0:
            period = self.current_period + 1
        assert(period > 0 and period <= self.time_horizon)
        sed_load = np.zeros_like(self.elevation)
        for b in self.breaches:
            sed_load += SSC * b.scaled_dist ** -2.3
        new_layer = self.elevation_cube[period - 1]
        new_layer = aggrade_patches(heads, heads.index, ws, rho, sed_load, dP, dO, new_layer, self.border_height)
        self.elevation_cube[period] = new_layer


def test():
    global pdr
    global tides
    global ws
    global rho
    global SSC
    global dP
    global dO
    
    file = '../data/p32_tides.dat'
    parser = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
    start = pd.datetime(2015,5,15,1)
    end = pd.datetime(2016,5,14,1)
    
    tides = load_tides(file,parser,start,end) + 0.25
    
    # Calculate Mean High Water
#    pressure = tides.as_matrix()
#    MW = np.mean(tides)
#    HW = pressure[argrelextrema(pressure, np.greater)[0]]
#    LW = pressure[argrelextrema(pressure, np.less)[0]]
#    MHW = np.mean(HW)
#    MLW = np.mean(LW)
    
    X = 500 # X size of polder
    Y = 300 # Y size of polder
    year = 8759 # hours in a year
    N = 100 # number of households
    
    # Initialize dataframe of household paramters
    max_wealth = 10000 # initial max wealth in Taka
    max_profit = 100 # max profit per 1 m^2 land in Taka    
    
    time = 10 # in years
    gs = 0.03 # grain size in m
    ws = ((gs/1000)**2*1650*9.8)/0.018 # settling velocity calculated using Stoke's Law
    rho = 700 # dry bulk density in kg/m^2
    SSC = 0.4 # suspended sediment concentration in g/L
    dP = 0 # compaction
    dO = 0 # organic matter deposition
    
    # breach coordinates on polder
    breachX = 0
    breachY = Y/2

    pdr = polder(x = X, y = Y, time_horizon= time, n_households = N, max_wealth=max_wealth, max_profit = max_profit)
    pdr.add_breach(breachX, breachY, time)
