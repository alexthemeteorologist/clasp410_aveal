#!/usr/bin/env python3
'''
this script will be containing written functions in order to explore a single layer atmospheric model.
'''
#as always, we must import all standard python libraries.
import numpy as np
import matplotlib.pyplot as plt
#because dan said it looked nice use fivethirtyeight for your plotting style
#first, let's start off with a set of a few arrays.
# ensure that this is our time array. from 1900 to 2000.
year = np.array([1900, 1950, 2000])
#this is our flux value range
s_0 = np.array([1365, 1366.5, 1368])
#stefan-boltzmann constant
sigma = 5.67E-8
#albedo ratio
alpha = .33
#temperature anomaly.
t_anom = np.array([-.4, 0, .4])
def temp_function(s_0 = 1365, albedo = .33):
    '''
    given the surface temperature of the earth (denoted by s_0), determine the temperature of the Earth's surface by using a single-layer
    perfectly absorbing energy balanced atmosphere model.
    '''
    temp_earth = (s_0 * (1-alpha) / (2*sigma))**(.25)
    return temp_earth

def compare_warming():
    '''
    this function is designed to create beautifullllllllllllll plots about changes in solar driving
    accounting for any type of climate change, and if we can subsequently measure them!
    '''
    temp_model = temp_function(s_0 = s_0)
    temp_observation = temp_model[1] + t_anom

    fig, ax = plt.subplots(1,1, figsize = (10,10))
    ax.plot(year, temp_observation, label = 'Observed Temperature Change')
    ax.plot(year, temp_model, label = 'Predicted Temperature Change')
    ax.legend(loc='best')
    ax.set_xlabel('Year')
    ax.set_ylabel('Surface Temperature ($in K$)')
    fig.tight_layout()
    plt.show()
    print(f'the final predicted temperature in 2000 is {temp_model[2]} degrees $K$')
    print(f'the final actual observed temperature in 2000 is {temp_observation[2]} degrees $K$')
    print(f'the actual temperature increase over the 100 year span is {temp_observation[2] - temp_model[2]} degrees Kelvin')