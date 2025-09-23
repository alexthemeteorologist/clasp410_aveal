#!/usr/bin/env python3
'''
the file will be used for multiple function definitions and plotting figures based on atmospheres and atmospheric definitions
of varying layers, solar constants and emissivities along with reflectivity values. a few function definitions were constructed for
n-layer atmospheres in regards to Earth's atmosphere, and one special case scenario

in regards to this Lab, questions 1 & 2 can be ran by the function n_layer_atmos(nlayers = 5, epsilon = 1), where the number of layers
is a REQUIRED input. we want to ensure our function matrix works for ANY value of emissivity and ANY number of layers.

question 3; part 1's adjacent running code can be found outside the previous function definition. part 2 had another function created for comparison.
simply run the algorithm ("run lab1.py") once in order to gather the plot for question 3, experiment 1. the second plot for number 3 can be found by
calling the function revised_n_layer_atmos, and will produce a plot of varying emissivities with respect to temperature.

question 4: this function definition is intended to create and display a matrix based on Venus' atmosphere. we will simply call the function
named as 'venus_layer_atmos('nlayers=52')', and will subsequently produce a matrix with similar parameters; enter in a value (REQUIRED) for nlayers,
and with that value we can now understand how many layers of venus' atmosphere.... given an albedo change & solar forcing change.....is needed
in order to construct a Venus' surface temperature of 700K. 

question 5: given this question has two parts, we would like to first return a matrix of temperatures using the nuclear war conditions.
you may run the matrix of temperatures and check any values using any atmospheric layer condition by typing in said function 
'nuclearwinter(nlayers = *REQUIRED PARAMETER*, epsilon = .5) in order to satisfy the parameters of the question's ask.
the second function for question 5 will be ran by nuclearwinterplot(nlayers = 5, epsilon = .5), and a nuclear winter plot will show up.


COLLABORATORS: Katherine Paff, Tyler Overbeek
'''
#let's import standard python libraries to help with calculations and plotting.
import numpy as np
import matplotlib.pyplot as plt

# stefan-boltzmann constant 
sigma = 5.67 * 10**(-8)
#solar constant
s0 = 1350
#the reflectivity of the body we want to use
albedo = .33
# lets create an array of coefficients (temperatures) using the matrix and function definition
def n_layer_atmos(nlayers, epsilon = 1, albedo = .33, s0 = 1350, debug = 0,):
    '''
    we will be able to make a matrix of 'n-layer' atmospheric temperatures in order to represent EARTH at a particular emissivity
    and an arbitrary number of 'n' layers of the atmosphere. the equation for the matrix in 'Ax = b' form is derived by the 
    'else' condition in the function definition.

    CONSTANTS DEFINED: albedo; the reflectivity of the atmosphere
    s0 = solar constant, in terms of Flux with units W*m**(-2)
    epsilon = emissivity of the atmosphere; black body emissivity is = 1. energy in = energy out.

    REQUIRED PARAMETERS: 
    nlayers; enter a number of layers of the atmosphere you want to enter.
    epsilon; of course, the function definition defaults emissivity to 1, but this can voluntarily be changed.

    RETURNS:
    matrix of arbitrary fluxes converted into temperatures.
    '''
    #create Matrix Ax = b
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    # the loop ensures each row and co
    for i in range(nlayers+1): #take into account all rows based on n-layers
        for j in range(nlayers+1): #take into account all columns based on n-layers
            if i == j: #if we're in the first row
                A[i,j] = -2 + (j == 0)
            else: #if the rows do NOT equal the first row....then execute
                 A[i,j] = (epsilon**(i>0)) * ((1-epsilon)**(np.abs(j-i)-1))
    if debug:
        print(f'A[i={i},j={j}] = {A[i, j]}')
        print(A)
    #CONVERTING FLUXES TO TEMPERATURES
    #use now the solar flux equation for the first,'b' value
    b[0] = -1/4*(1-albedo)*s0
    #conduct A^(-1) operation on the matrix
    Ainv = np.linalg.inv(A)
    #our flux values based on multiplying A(A^(-1))*b
    fluxes = np.matmul(Ainv, b)

    #return temperatures now
    temps =(fluxes/epsilon/sigma)**(1/4)
    temps[0] = (fluxes[0]/sigma)**(1/4)
    return temps


# Question 3 - Experiment 1; range of emissivities
# Question 3 - Experiment 1; range of emissivities
# Question 3 - Experiment 1; range of emissivities
# Question 3 - Experiment 1; range of emissivities
# Question 3 - Experiment 1; range of emissivities


#empty array of temperatures.
temps_array = []

#use ranges of emissivities from .01 to 1, with a spacing of 100 different emissivities possible.
epsilon_range = np.linspace(0.01, 1, 100)
for i in epsilon_range:
    plotted_temps = n_layer_atmos(nlayers = 1, epsilon = i)
    temps_array.append(plotted_temps[0])
#plot each temperature from the temperature array now, and subsequently plot with respect to emissivity range
plt.figure(figsize=(8, 5))
plt.plot(epsilon_range, temps_array, color='darkgreen')
plt.axhline(288, color='red', linestyle='--', label='Observed Earth Temp (288K)') #denote 288K in order to see decrease in temperature
plt.xlabel('Atmospheric Emissivity')
plt.ylabel('Surface Temperature (K)')
plt.title('Surface Temperature vs Emissivity (Single-Layer Atmosphere)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
# Question 3 - Experiment 2;
# Question 3 - Experiment 2; 
# Question 3 - Experiment 2; 
# Question 3 - Experiment 2; 
# Question 3 - Experiment 2;  
def revised_n_layer_atmos(nlayers = 5, revised_epsilon = .225, albedo = .33, s0 = 1350, debug = 0,):
    '''
    constructing plot with new revised emissivity value for Earth's atmosphere. use the altitude array now to determine what number of layers
    will replicate a 288K Earth atmosphere given the new emissivity, and use 'nlayers = 5' and 'revised_epsilon = .225'

    PARAMETERS: 
    n-layers; change the REQUIRED n-layers parameter as needed in order to change the number of layers and construct the plot.

    OUTPUT;
    returns a plot with altitude (atmospheric layers) with respect to surface temperatures.
    '''
    #the difference is now we use the revised epsilon to conduct the same function.
    #use the same n_layer function thereafter in order to have the same temperatures with respect to altitude & layers.
    altitude_temps = n_layer_atmos(nlayers, epsilon = revised_epsilon)
    altitude= np.arange(0,60,10)
    plt.figure(figsize=(8, 5))
    plt.plot(altitude_temps, altitude, color='darkgreen')
    plt.axvline(288, color='red', linestyle='--', label='Observed Earth Temp (288K)')
    plt.xlabel('Temperature in Kelvin')
    plt.ylabel('Altitude in kilometers')
    plt.title('Atmospheric Height of Earth Compared to Surface Temperature')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Question 4 - Experiment 1; 
# Question 4 - Experiment 1; 
# Question 4 - Experiment 1; 
def venus_layer_atmos(nlayers, epsilon = 1, albedo = .6, s0 = 2600, debug = 0,):
    '''
    changes the solar constant and albedo to that of Venus's atmospheric composition. function creates new atmospheric layeral
    matrix in order to determine how many layers of atmosphere Venus must have in order to achieve surface temperature of 700K.

    PARAMETERS:
    nlayers; REQUIRED PARAMETERS ONCE AGAIN in order to change the layeral and surface temperature composition.

    OUTPUTS:
    matrix of n-layer venus atmospheric temperatures (layeral variant). since we wanted 52 layers of venus' atmosphere to create 
    a 700K surface temperature, we would get a matrix that produces 52 values (one for each atmospheric layer)
    '''
    #extremely similar to the previous function; but the difference is the actual output due to the different albedo and solar constant values.
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j:
                A[i,j] = -2 + (j == 0)
            else:
                 A[i,j] = (epsilon**(i>0)) * ((1-epsilon)**(np.abs(j-i)-1))
    if debug:
        print(f'A[i={i},j={j}] = {A[i, j]}')
        print(A)
    #very very similar to the previous function, HOWEVER keynote the previous parameters in order to change the final outputs
    b[0] = -1/4*(1-albedo)*s0
    Ainv = np.linalg.inv(A)
    fluxes = np.matmul(Ainv, b)
    venustemps =(fluxes/epsilon/sigma)**(1/4)
    venustemps[0] = (fluxes[0]/sigma)**(1/4)
    return venustemps

def nuclearwinter(nlayers, epsilon = .5, albedo = 0, s0 = 1350, debug = 0,):
    '''
    changes the solar constant and albedo to that of a nuclear atmospheric composition. in this case, we want the actual
    albedo value to change and decrease to 0, given a nuclear atmosphere will be one that has ZERO absorption due to all the 
    materials of the radiation and debris.

    PARAMETERS:
    nlayers; REQUIRED PARAMETERS ONCE AGAIN in order to change the layeral and surface temperature composition.

    OUTPUTS:
    matrix of n-layer nuclear** atmospheric conditions (layeral variant), and outputs temperatures of each atmospheric layer.
    '''
    #once again the temperature profile matrix will appear but using slightly different values for epsilon and albedo
    # keep in mind the nuclear atmospheric conditions , and temperature profiles will look different.
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j: #if we're in the first row
                A[i,j] = -2 + (j == 0)
            else:
                 A[i,j] = (epsilon**(i>0)) * ((1-epsilon)**(np.abs(j-i)-1))
    if debug:
        print(f'A[i={i},j={j}] = {A[i, j]}')
        print(A)
    b[0] = 0
    b[-1] = -1/4*(1-albedo)*s0
    Ainv = np.linalg.inv(A)
    fluxes = np.matmul(Ainv, b)
    nucleartemps =(fluxes/epsilon/sigma)**(1/4)
    nucleartemps[0] = (fluxes[0]/sigma)**(1/4)
    return nucleartemps

def nuclearwinterplot(nlayers, epsilon = .5, albedo = 0, s0 = 1350, debug = 0,):
    '''
    changes the solar constant and albedo to that of a nuclear atmospheric composition. in this case, we want the actual
    albedo value to change and decrease to 0, given a nuclear atmosphere will be one that has ZERO absorption due to all the 
    materials of the radiation and debris.

    PARAMETERS:
    nlayers; REQUIRED PARAMETERS ONCE AGAIN in order to change the layeral and surface temperature composition.

    OUTPUTS:
    plot of the nuclear conditional atmosphere layers with respect to temperature and the specific layeral heights.
    '''
    nuclearaltitude_temps = nuclearwinter(nlayers = 5)
    nuclearaltitude= np.arange(0,60,10)
    plt.figure(figsize=(8, 5))
    plt.plot(nuclearaltitude_temps, nuclearaltitude, color='darkgreen')
    plt.xlabel('Temperature in Kelvin')
    plt.ylabel('Altitude in kilometers')
    plt.title('Nuclear Atmosphere - Surface Temperature vs. Height')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    