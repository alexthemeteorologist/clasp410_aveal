#!/usr/bin/env python3

'''
A set of tools for solving Newton's law of cooling, i.e.,

$\frac{d T(t)}{dt} = k \left(T_{env} - T(t) \right)$

...where values and units are defined below.

Within this module are functions to return the analytic solution of the heat
equation, numerically solve it, find the time to arrive at a certain
temperature, and visualize the results.

The following table sets the values and their units used throughout this code:

| Symbol | Units  | Value/Meaning                                            |
|--------|--------|----------------------------------------------------------|
|T(t)    | C or K | Surface temperature of body in question                  |
|T_init  | C or K | Initial temperature of body in question                  |
|T_env   | C or K | Temperature of the ambient environment                   |
|k       | 1/s    | Heat transfer coefficient                                |
|t       | s      | Time in seconds                                          |

DAN TEACHING NOTES:
1) Start by mapping out what we want as our end goal:
   - Temperature vs. time for each scenario. <- FUNCTION #1!
   - The time it takes in seconds to cool to our target. <- FUNCTION #2!
2) List out the inputs and outputs for this!
    - Inputs: time, ambient temp, initial temp, heat transfer coeff.
    - Outputs: temperature for each time in input time.
3) Start by just hard coding everything.
    - Everything as a top-to-bottom script, hard coded.
    - Boy, changing this stuff by hand sure sucks.
4) Then, turn into two functions.
    - Show how easy it is to quickly change things.
5) Finally, build a function to answer the final question.
    - Create the final plot and deliverable.
    - Build the plot component wise, adding better titles and labels as we
      go. Contrast against initial plot that sucks.
'''

# Standard imports:
import numpy as np
import matplotlib.pyplot as plt

# Set the plot style
# (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)
plt.style.use('fivethirtyeight')


def solve_temp(t, k=1/300., T_env=20, T_init=90):
    '''
    For a given scalar or array of times, `t`, return the analytic solution
    for Newton's law of cooling:

    $T(t)=T_env + \left( T(t=0) - T_{env} \right) e^{-kt}$

    ...where all values are defined in the docstring for this module.

    Parameters
    ==========
    t : Numpy array
        Array of times, in seconds, for which solution will be provided.


    Other Parameters
    ================
    k : float
        Heat transfer coefficient, defaults to 1/300. s^-1
    T_env : float
        Ambient environment temperature, defaults to 20°C.
    T_init : float
        Initial temperature of cooling object/mass, defaults to °C

    Returns
    =======
    temp : numpy array
        An array of temperatures corresponding to `t`.
    '''

    return T_env + (T_init - T_env) * np.exp(-k*t)


def time_to_temp(T_target, k=1/300., T_env=20, T_init=90):
    '''
    Given an initial temperature, `T_init`, an ambient temperature, `T_env`,
    and a cooling rate, return the time required to reach a target temperature,
    `T_target`.

        Parameters
    ==========
    T_target : scalar or numpy array
        Target temperature in °C.


    Other Parameters
    ================
    k : float
        Heat transfer coefficient, defaults to 1/300. s^-1
    T_env : float
        Ambient environment temperature, defaults to 20°C.
    T_init : float
        Initial temperature of cooling object/mass, defaults to °C

    Returns
    =======
    t : scalar or numpy array
        Time in s to reach the target temperature(s).
    '''

    return (-1/k) * np.log((T_target - T_env)/(T_init - T_env))
# from this point forward ...... your code will be typed on your own with your own functions and what not.