#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:33:50 2020

@author: darrenhsu
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

#%%

def func(x, a, b, c, d, e, f, IRF):
    return (a * np.exp(-x/b) + c * np.exp(-x/d) + e * np.exp(-x/f)) * (1+erf(x/IRF)) / 2

xdata = np.linspace(-100,300,801)
y = func(xdata, 2.3, 2, 0.7, 15, 1, 120, 1.1) # unitless, ps, ps, ...
np.random.seed()
y_noise = np.random.normal(size=xdata.size) * 0.1
ydata = y + y_noise
y_err = 1 / np.sqrt(np.linspace(2,230,801)) + 0.05 * np.linspace(1,2,801)

fig = plt.figure(figsize=(12,8))
plt.errorbar(xdata, ydata, yerr=y_err, ecolor='#999999', capsize=3)
plt.show()
#%%

popt, pcov = curve_fit(func, xdata, ydata, p0=[2, 3, 1, 10, 1, 100, 0.3], 
                       sigma=y_err, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
print(popt)
print(perr)
#plt.plot(xdata, ydata, 'b-')
plt.errorbar(xdata, ydata, yerr=y_err, ecolor='#999999',capsize=3,zorder=5)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f, f=%5.3f, IRF=%5.3f' % tuple(popt),
         zorder=10)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=0, fontsize='x-small')
plt.xlim(-40,150)
plt.show()
