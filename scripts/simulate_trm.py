# -*- coding: utf-8 -*-
"""
Created on Mon May 22 19:34:34 2017

@author: jonathan
"""
import os
import pickle
import matplotlib.pyplot as plt

from trm_abm import *


def calc_trm(pdr, discount, horizon, trm_k = 2.0, wl_k = 1.0):
    global euc
    for hh in pdr.households.values(): hh.discount = discount
    euc = pdr.calc_eu_series(MHW, trm_k, MW, wl_k, horizon)
    d_euc = euc[1:] - euc[0]
    dem = max(d_euc.max(), -d_euc.min())
    for i in range(d_euc.shape[0]):
        plt.figure();
        plt.imshow(d_euc[i], "seismic_r", vmin = -dem, vmax = dem)
        plt.axis("off")
        cbar = plt.colorbar()
        cbar.set_label("Net present value")


def test(ec = None):
    global pdr
    global tides
    global ws
    global rho
    global SSC
    global dP
    global dO
    global MW
    global HW
    global LW
    global MHW
    global MLW
    global elevation_cube
    global euc
    # global profit
    global trm_profit
    global wl_profit

    file = '../data/p32_tides.dat'
    parser = lambda x: pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
    start = pd.datetime(2015,5,15,1)
    end = pd.datetime(2016,5,14,1)

    tides = load_tides(file,parser,start,end) + 0.25

    # Calculate Mean High Water
    pressure = tides.as_matrix()
    MW = np.mean(tides)
    HW = pressure[argrelextrema(pressure, np.greater)[0]]
    LW = pressure[argrelextrema(pressure, np.less)[0]]
    MHW = np.mean(HW)
    MLW = np.mean(LW)

    X = 500 # X size of polder
    Y = 300 # Y size of polder
    year = 8759 # hours in a year
    N = 100 # number of households

    # Initialize dataframe of household paramters
    max_wealth = 10000 # initial max wealth in Taka
    max_profit = 100 # max profit per 1 m^2 land in Taka

    t = 10 # in years
    gs = 0.03 # grain size in m
    ws = ((gs/1000)**2*1650*9.8)/0.018 # settling velocity calculated using Stoke's Law
    rho = 700 # dry bulk density in kg/m^2
    SSC = 0.4 / 4.0 # suspended sediment concentration in g/L
    dP = 0 # compaction
    dO = 0 # organic matter deposition

    # breach coordinates on polder
    breachX = 0
    breachY = Y/2

    pdr = polder(x = X, y = Y, time_horizon= t, n_households = N,
                 max_wealth=max_wealth, max_profit = max_profit,
                 border_height = 0.5, amplitude = 1.5, noise = 0.05)
    pdr.add_breach(breachX, breachY, t)

    if ec is None:
        t0 = time.time()
        t1 = t0
        for i in range(pdr.time_horizon):
            pdr.aggrade(tides, ws, rho, SSC, dP, dO, i + 1)
            t2 = time.time()
            print ("%2d: %.02f, %.02f" % (i, float(t2 - t1), float(t2 - t0)))
            t1 = t2
        elevation_cube = pdr.elevation_cube.copy()
    else:
        pdr.elevation_cube = ec.copy()
        pdr.elevation = ec[0].copy()

    calc_trm(pdr, 0.15, 4, trm_k = 5.0)
    trm_profit = pdr.calc_profit(MHW, 5.0, elevation_cube, False)
    wl_profit = pdr.calc_profit(MW, 1.0, elevation_cube, False)

def save_images(folder = "csdms_figures", euc = None, ec = None, wl_profit = None, trm_profit = None, note = None):
    plt.ioff()
    if euc is not None:
        d_euc = euc[1:] - euc[0]
        dem = max(d_euc.max(), -d_euc.min())
        for i in range(d_euc.shape[0]):
            f = plt.figure();
            plt.imshow(d_euc[i], "seismic_r", vmin = -dem, vmax = dem)
            plt.axis("off")
            cbar = plt.colorbar()
            cbar.set_label("Net present value")
            plt.draw()
            if note is not None:
                caption = "polder_%02d_du_%s.png" % (i, note)
            else:
                caption = "polder_%02d_du.png" % i
            plt.savefig(os.path.join(folder, caption), dpi = 300)
            plt.close(f)
    if ec is not None:
        el_min = ec.min()
        el_max = ec.min()
        scale = max(el_max - MW, MW - el_min)
        for i in range(ec.shape[0]):
            f = plt.figure();
            plt.imshow(ec[i], "terrain", vmin = MW - scale, vmax = MW + scale)
            plt.axis("off")
            cbar = plt.colorbar()
            cbar.set_label("Elevation (m) rel. MW")
            plt.draw()
            if note is not None:
                caption = "polder_%02d_elev_%s.png" % (i, note)
            else:
                caption = "polder_%02d_elev.png" % i
            plt.savefig(os.path.join(folder, caption), dpi = 300)
            plt.close(f)
    if wl_profit is None:
        pmax = 0.0
    else:
        pmax = wl_profit.max()
    if trm_profit is not None:
        pmax = max(pmax, trm_profit.max())

    if wl_profit is not None:
        # pmax = max(wl_profit)
        for i in range(wl_profit.shape[0]):
            f = plt.figure();
            plt.imshow(wl_profit[i], "plasma", vmin = 0, vmax = pmax)
            plt.axis("off")
            cbar = plt.colorbar()
            cbar.set_label("Profit")
            plt.draw()
            if note is not None:
                caption = "polder_%02d_wl_profit_%s.png" % (i, note)
            else:
                caption = "polder_%02d_wl_profit.png" % i
            plt.savefig(os.path.join(folder, caption), dpi = 300)
            plt.close(f)
    if trm_profit is not None:
        # pmax = max(wl_profit)
        for i in range(trm_profit.shape[0]):
            f = plt.figure();
            plt.imshow(trm_profit[i], "plasma", vmin = 0, vmax = pmax)
            plt.axis("off")
            cbar = plt.colorbar()
            cbar.set_label("Profit")
            plt.draw()
            if note is not None:
                caption = "polder_%02d_trm_profit_%s.png" % (i, note)
            else:
                caption = "polder_%02d_trm_profit.png" % i
            plt.savefig(os.path.join(folder, caption), dpi = 300)
            plt.close(f)
    plt.ion()

def runit(force = False):
    global elevation_cube, a, v, a_res, v_res

    plt.ioff()
    if os.path.exists('elevation_cube.pickle') and not force:
        elevation_cube = pickle.load(open('elevation_cube.pickle', 'rb'))
        test(elevation_cube)
    else:
        test()
        pickle.dump(elevation_cube, open('elevation_cube.pickle', 'wb'))


    plt.draw()
    plt.close('all')
    plt.ion()

    a = auction(pdr.households)
    a_res = a.auction()

    v = election(pdr.households)
    v_res = v.vote()

def batch(force = False, trm_k = 5.0):
    global vres_list, ares_list
    vres_list = []
    ares_list = []
    if not os.path.exists('batch'):
        os.mkdir('batch')
    pickle.dump(elevation_cube, open(os.path.join('batch', 'elevation_cube.pickle'), 'wb'))
    pickle.dump(pdr.owners, open(os.path.join('batch', 'owners.pickle'), 'wb'))
    pickle.dump(pdr.plots, open(os.path.join('batch', 'plots.pickle'), 'wb'))
    # trm_profit = pdr.calc_profit(MHW, 5.0, elevation_cube, False)
    # wl_profit = pdr.calc_profit(MW, 1.0, elevation_cube, False)
    for horizon in range(3,7):
        calc_trm(pdr, 0.15, 4, trm_k = trm_k)

        a = auction(pdr.households)
        v = election(pdr.households)
        ares = a.auction()
        vres = v.vote()
        vres_list.append(vres)
        ares_list.append(ares)
        print "Vote: winner = ", vres[0], " min utility = ", min(vres[1].values()), ", ", v.count_unhappy(vres[0]), " unhappy households"
        print "Auction: winner = ", ares[0], " min utility = ", min(ares[1].values()), ", ", a.count_unhappy(ares[0]), " unhappy households"
