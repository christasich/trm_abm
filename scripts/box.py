# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:09:20 2017

@author: chris
"""
import numpy as np

X = 50
Y = 30

max = 10
min = 5

box = np.zeros([Y,X],dtype=int)

HH = 1
while np.min(box) == 0:
    x = np.random.randint(min,max)
    y = np.random.randint(min,max)
    xx,yy = np.where(box==0)
    randind = np.random.randint(0,len(xx))
    ycor = xx[randind]
    xcor = yy[randind]
    box[ycor:ycor+y,xcor:xcor+x] = HH
    HH = HH + 1