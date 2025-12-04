#!/usr/bin/env python3
'''
this code script has a few different functions runnable to produce all figures for lab 5.
KEEP IN MIND THAT THE "gamma" parameter in snowball_earth IS ONLY NEEDED FOR QUESTION 4. IT NEEDS TO BE REMOVED
FOR ALL OTHER problems unless you are running the code script for question 4 outputs only.

for question 1, simply run the program and type into the terminal 
problem1() 
and the example graph will be replicated as shown in the lab handoutt.

for question 2, simply run in the terminal 
problem2() 
and then the corresponding replica graph will appear for that question

for question 3, run in the terminal 
problem3() and then the comparison plots will show for the different dynamic albedos and the 
radiative factors that have changed.

for question 4, ADD TO the snowball_earth function definition "gamma", so that way it could be recognized
when acknowledging the changes in gamma per every single step.

now, you can run in the terminalproblem4() and then the final plot of snowball earth conditions with the Gamma multiplier will appear.

COLLABORATORS: Tyler Overbeek and Katherine Paff.
'''
#for the last last last time, import our standard python libraries.
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
#some constants:
radearth = 6357000 #the radius of earth in meters.
mxdlayer = 50 #the depth of the mixed layer 
sigma = 5.67*10**-8 # stefan-boltzmann constant
C = 4.2*10**6 #specific heat capacity of water :)
rho = 1020 # density of water in kg/m^(3)
solar = 1370

def generategrid(npoints = 18):
    '''
    create an evenly space grid with the cell centers as npoints.
    grid always runs from zero to 180 degrees as being the edges of the grid. this also means that the first gridpoint will be dLat/2
    and the last point is 180 - dLat/2

    parameters: 
        npoints : int, defaults to 18
            number of grid points to create
    
    returns
        dlat: float
            grid spacing in latitude degrees
        lats: numpy array
            locations of all the grid centers
    '''
    dlat = 180/npoints # latitude spacing
    lats = np.linspace(dlat/2., 180-dlat/2.,npoints) # lat cell centers
    return dlat, lats

def temp_warm(lats_in):
    '''
    '''
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    #get base grid
    npoints = T_warm.size
    dlat, lats = generategrid(npoints)
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

    #set initial temperature curve:
def insolation(s0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    s0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = s0 * angle.sum()
    s0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = s0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = s0_avg * insolation / 365

    return insolation


def snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr=False, albice=.6,
                   albgnd=.3, apply_insol=False, solar=1370):
    '''
    Solve the snowball Earth problem.

    Parameters
    ----------
    nlat : int, defaults to 18
        Number of latitude cells.
    tfinal : int or float, defaults to 10,000
        Time length of simulation in years.
    dt : int or float, defaults to 1.0
        Size of timestep in years.
    lam : float, defaults to 100
        Set ocean diffusivity
    emiss : float, defaults to 1.0
        Set emissivity of Earth/ground.
    init_cond : function, float, or array
        Set the initial condition of the simulation. If a function is given,
        it must take latitudes as input and return temperature as a function
        of lat. Otherwise, the given values are used as-is.
    apply_spherecorr : bool, defaults to False
        Apply spherical correction term
    apply_insol : bool, defaults to False
        Apply insolation term.
    solar : float, defaults to 1370
        Set level of solar forcing in W/m2
    albice, albgnd : float, defaults to .6 and .3
        Set albedo values for ice and ground.

    Returns
    --------
    lats : Numpy array
        Latitudes representing cell centers in degrees; 0 is south pole
        180 is north.
    Temp : Numpy array
        Temperature as a function of latitude.
    '''

    # Set up grid:
    dlat, lats = generategrid(nlat)
    # Y-spacing for cells in physical units:
    dy = np.pi * radearth / nlat

    # Create our first derivative operator.
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0, :] = B[-1, :] = 0

    # Create area array:
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    # Set number of time steps:
    nsteps = int(tfinal / dt)

    # Set timestep to seconds:
    dt = dt * 365 * 24 * 3600

    # Create insolation:
    insol = insolation(solar, lats)

    # Create temp array; set our initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond

    # Create our K matrix:
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    # Boundary conditions:
    K[0, 1], K[-1, -2] = 2, 2
    # Units!
    K *= 1/dy**2

    # Create L matrix.
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # Set initial albedo.
    albedo = np.zeros(nlat)
    loc_ice = Temp <= -10  # Sea water freezes at ten below.
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # SOLVE!
    for istep in range(nsteps):
        # Update Albedo:
        loc_ice = Temp <= -10  # Sea water freezes at ten below.
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albgnd

        # Create spherical coordinates correction term
        if apply_spherecorr:
            sphercorr = (lam*dt) / (4*Axz*dy**2) * np.matmul(B, Temp) * dAxz
        else:
            sphercorr = 0

        # Apply radiative/insolation term:
        if apply_insol:
            radiative = (1-albedo)*insol - emiss*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho*C*mxdlayer)

        # Advance solution.
        Temp = np.matmul(Linv, Temp + sphercorr)

    return lats, Temp

def problem1():
    '''
    Create solution figure for Problem 1 (also validate our code qualitatively)
    '''

    # Get warm Earth initial condition.
    dlat, lats = generategrid()
    temp_init = temp_warm(lats)

    # Get solution after 10K years for each combination of terms:
    lats, temp_diff = snowball_earth()
    lats, temp_sphe = snowball_earth(apply_spherecorr=True)
    lats, temp_alls = snowball_earth(apply_spherecorr=True, apply_insol=True,
                                     albice=.3)

    # Create a fancy plot!
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_init, label='Initial Condition')
    ax.plot(lats-90, temp_diff, label='diffusion')
    ax.plot(lats-90, temp_sphe, label='diffusion and spherical')
    ax.plot(lats-90, temp_alls, label='diffusion, spherical and Radiative')

    # add style and labels to all things we need thanks.
    ax.set_title('problem 1 grid solution')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')
    plt.show()
# ----------------------------------------------------
#this is just simply a test function to ensure all of our latitudes and grids print correctly
def test_functions():
    print('test gen_grid')
    print('for npoints = 5:')
    dlat_correct, lats_correct = 36.0, np.array([18., 54., 90., 126., 162.])
    result = gen_grid(5)
    if (result[0] == dlat_correct) and np.all(result[1] == lats_correct):
        print('\tPassed!')
    else:
        print('\tFAILED!')
        print(f"Expected: {dlat_correct}, {lats_correct}")
        print(f"Got: {gen_grid(5)}")
# -----------------------------------------------------
# lets continue through the rest of the lab and answer the rest of the prompts.
def problem2(nlat=18, tfinal=10000, dt=1.0, lam=100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr=False, albice=.6,
                   albgnd=.3, apply_insol=False, solar=1370):
    '''
    in problem two, we create the replica of the original initial condition graph but with varied epsilon and lambda values that end up remaining constant
    for the remainder of the lab to come.

    parameters: nlat, tfinal - constant values for time steps and number of latitude groups
    emiss - constant at 1 until redefined at the lats, temp_alls line.
    '''
    # vary the parameters of lambda and epsilon

    dlat, lats = generategrid()
    temp_init = temp_warm(lats)
    #original solution in green
    lats, temp_alls = snowball_earth(emiss = .7, lam = 88.9, apply_spherecorr=True, apply_insol=True,)
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_init, label='Initial Condition')
    ax.plot(lats-90,temp_alls, '--',label = 'new radiative solution', color = 'green')
    ax.set_xlabel('latitudes')
    ax.set_ylabel('degrees centigrade')
    ax.set_title('revised warm earth curve!')
    plt.legend(loc='best')
    plt.show()

def temp_hot(lats_in):
    '''
    modifies our original temperature array to alll hot temperatures at 60 degrees for all latitudes as well.
    '''
    T_hot = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                       60, 60, 60, 60, 60, 60, 60, 60])
    #get base grid
    npoints = T_hot.size
    dlat, lats = generategrid(npoints)

    coeffs = np.polyfit(lats, T_hot, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def temp_cold(lats_in):
    '''
    records the array that is updated for a cold snowball earth model.
    bear in mind all of the latitudes we want is already defined in the inpput of this model as well.

    additionally, this model replicates the "temp_hot" function denoted above.
    '''
    T_cold = np.array([-60, -60, -60, -60, -60, -60, -60, -60, -60, -60,
                       -60, -60, -60, -60, -60, -60, -60, -60])
    #get base grid
    npoints = T_cold.size
    dlat, lats = generategrid(npoints)
    coeffs = np.polyfit(lats, T_cold, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def problem3():
    '''
    Create solution figure for Problem 3 (also validate our code qualitatively)
    All plotting methods and different graphs are also representative on this function alone.
    '''

    # Get warm Earth initial condition.
    dlat, lats = generategrid()
    temp_init = temp_warm(lats)
    
    # Get solution after 10K years for each combination of terms:
    #lats, temp_diff = snowball_earth()
    #lats, temp_sphe = snowball_earth(apply_spherecorr=True)
    lats, temp_alls = snowball_earth(emiss = .7, lam = 88.9, apply_spherecorr=True, apply_insol=True,
                                     albgnd = .6 )
    lats, cold_temp = snowball_earth(emiss = .7, lam= 88.9,init_cond= temp_cold, apply_spherecorr=True, apply_insol=True,)
    lats, hot_temp = snowball_earth(emiss = .7, lam= 88.9,init_cond= temp_hot, apply_spherecorr=True, apply_insol=True,
                                     )
                                    

    # Create a fancy plot!
    fig, ax = plt.subplots(1, 1, figsize = (9,9))
    ax.plot(lats-90, temp_alls, label='flash freeze')
    #ax.plot(lats-90, temp_diff, label='diffusion')
    #ax.plot(lats-90, temp_sphe, label='diffusion and spherical')
    ax.plot(lats-90, cold_temp, label='cold earth, spherical and Radiative')
    ax.plot(lats-90, hot_temp, label='hot earth, spherical and Radiative')


    # Customize like those annoying insurance commercials
    ax.set_title('10000 year problem 3 grid solution')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')
    plt.show()

def problem4():
    '''
    this function will run the fourth problem of this lab with all the forcings added and
    the gamma multiplier steps also added as well. 
    '''
    dlat, lats = generategrid()
    temp_init = temp_cold(lats)

    #define the solar forcing equation (insolation)
    gamma = .4 # use this as the starting value for gamma before the for loop increases everything
    insol = gamma * insolation(solar, lats)

    #create ranges for solar forcing
    solar_multiplier_inc = np.arange(.4, 1.45, .05) #index the increasing gamma multiplier
    solar_multiplier_dec = np.linspace(1.4, .35, len(solar_multiplier_inc)) #index the decrease as well.


    #now, we can store temperature increases and decreases respectively
    temp_inc = []
    temp_dec = []

    #store the average temperatures, which we will end up plotting as well.
    avg_temp_inc = []
    avg_temp_dec = []

    # use the initial cold earth solution for initial conditions.
    init_cond = temp_cold(lats)
    temp_inc.append(init_cond)
    avg_temp_inc.append(np.mean(init_cond))


    #temp_dec.append(init_cond)
    #avg_temp_dec.append(np.mean(init_cond))

    #use the increasing solver now.
    for i in solar_multiplier_inc[1:]:
        lats, init_cond = snowball_earth(init_cond=init_cond, emiss = .7, lam = 89, apply_spherecorr= True, apply_insol= True, gamma = i)
        temp_inc.append(init_cond)
        avg_temp_inc.append(np.mean(init_cond))

    #repeat the process for the decreases:
    init_cond = temp_cold(lats)
    temp_dec.append(init_cond)
    avg_temp_dec.append(np.mean(init_cond))
    for i in solar_multiplier_dec[1:]:
        lats, init_cond = snowball_earth(init_cond=init_cond, emiss= .7 , lam = 89,
                                        apply_spherecorr= True, apply_insol= True, gamma = i)
        temp_dec.append(init_cond)
        avg_temp_dec.append(np.mean(init_cond))

    print(avg_temp_inc)
    print(avg_temp_dec)
    
    #time for plotting :)
    fig, ax = plt.subplots(1, 1, figsize= (10,8))
    ax.plot(solar_multiplier_inc, avg_temp_inc, label='increasing solar multiplier (gamma)')
    ax.plot(solar_multiplier_dec, avg_temp_dec, label = 'Decreasing solar multiplier')

    # differ the style again :)
    ax.set_title('earth equilibrium using cold temperature model')
    ax.set_ylabel(r'Temp($^{\circ}C$)')
    ax.set_xlabel('Solar Multiplier Range')
    ax.legend(loc = 'best')
    plt.show()


