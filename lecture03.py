#!/usr/bin/env python3

#import standard python libraries.
import numpy as np
import matplotlib.pyplot as plt

dx = .1
x = np.arange(0,6*np.pi,dx)
sinx=np.sin(x)
cosx=np.cos(x)
#the hard way.........
#fwd_diff = np.zeros(x.size -1)
#for i in range(x.size - 1):
#    fwd_diff[i] = x[i+1] - x[i]

#the easier way...
fwd_diff = (sinx[1:]-sinx[:-1]) / dx
bkd_diff = (sinx[1:]-sinx[:-1]) / dx
plt.plot(x, cosx, label = 'analytical derivative of $\sin{x}$')
plt.plot(x[:-1],fwd_diff, label = 'forward diff approx')
plt.plot(x[1:], bkd_diff, label = ' backward diff approx')
plt.legend(loc='best')
plt.show()