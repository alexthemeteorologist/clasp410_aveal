#!/usr/bin/env python3
'''
'''
#let's import standard python libraries to help with calculations and plotting.
import numpy as np
import matplotlib.pyplot as plt

# stefan-boltzmann constant 
sigma = 5.67 * 10**(-8)
#solar constant
s0 = 1370
#the reflectivity of the body we want to use
albedo = .33
# ------------------------------------------------------------------------------

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
    IN THE UPDATED FINAL project, we want to VARY EPSILON PER LAYER and thenceforth calculate 

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

def venus_layer_atmos(nlayers = 52, epsilon = 1, albedo = .6, s0 = 2600, debug = 0,):
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

nlayers_range = np.arange(0, 21, 1)
k = 0.2  # coupling constant that allows for emissivity to exponentially increase PER LAYER
# as layers increase, epsilon values, or emissivity increases

# we want to create empty arrays
temps_by_layers = []
eps_by_layers = []
newvenustemps_by_layers = []
for nL in nlayers_range:
    eps = 1.0 - np.exp(-k * nL)  # monotonic increase toward 1 AS the number of layers increases
    eps_by_layers.append(eps) # add the new associated emissivity value to a list
    temps = n_layer_atmos(nlayers=nL, epsilon=eps)
    temps_by_layers.append(temps[0])
    newvenustemps = venus_layer_atmos(nlayers = nL, epsilon = eps)
    newvenustemps_by_layers.append(newvenustemps[0])

# reverse the temperature array that way our original temperature array matches the other plots we created 
reversedtemps = list(reversed(temps))
print(temps_by_layers)
print(temps)
print(len(temps))
print(newvenustemps)
print(len(newvenustemps))
plt.figure(figsize=(8,7))
plt.plot(nlayers_range, temps_by_layers, label= ' new earth layered profile ', color='darkgreen')
plt.twinx()
plt.plot(nlayers_range,temps, reversedtemps, label='original Earth temp profile', color='steelblue', linestyle='--')
plt.plot(nlayers_range,newvenustemps_by_layers, label = 'venus epsilon changes/layer', color = 'gold', linestyle = '--')
plt.xlabel('Number of layers')
plt.ylabel('Temperature & Epsilon')
plt.title('Coupled epsilon & resulting surface temperature')
plt.grid(True)
plt.tight_layout()
plt.legend(loc ='best')
plt.show()
