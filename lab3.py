#!/usr/bin/env python3
'''
Question 1: simply run the program first, then type into the terminal heat_solve() in order for the three 
arrays to print out. The three arrays will be dependent on the function temp_kanger, but you would need 
to change the value of 'offset' in order to gather the condtions from the beginning, along with the 
conditions that are set to be modified based on the global temperature increases that are set to be 
outlined in question 3. The three arrays requested for question 1 will print in succinct order. Bear in mind,
the three arrays is a collection of time points, space points, and temperatures altogether.

Question 2: a few steps to run and generate the different plots;
first, you will run the program in its entirety - run lab3.py . 
then you will type directly into the terminal -
t, x, U = heatsolver()

and then type for the plotting
plot_heatsolver(t,x,U, vmin = -25, vmax = 25)

and then beautiful plots will appear! question 2 asks for the generic case, so ALLLLLLL cases of 'offset' have been set to 0.
they can be changed to represent temperature differences and global warming conditions for the next question.

Question 3: similar process as question 2 to run the program and produce different plots....just edit a few
values in the code snippet to end up changing values and making our plots look different for each condition asked.

AFTER YOU HAVE CHANGED UNIFORM CONDITIONS (changing ALL instances of 'offset' to the desired value for the temperature changes),
then ensure the program is re-ran ---- run lab3.py

still type in 
t, x , U = heatsolver()

and finally type in 
plot_heatsolver(t, x , U, vmin = -25, vmax = 25)

then the updated heatmap and graphic of kanger temperatures will show up!
'''
#first, we must always import standard python libraries.
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# heres the solution matrix seen below. what the output is supposed to return if the kanger solution is indeed correct.
sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
           [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
           [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
           [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
           [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
           [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
           [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
           [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
           [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
           [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
           [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]
sol10p3 = np.array(sol10p3).transpose()

# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
offset = 0 #factor in the degrees celsius offset
adjusted_t_kanger = t_kanger + offset #modify your kanger curve to account for the shift in surface temp.
def temp_kanger(t):
    '''
    this specific function sets up the kangerlussaq average temperature functions.
    Parameters
    ---------------------
    t : optional - the temperatures actually output in the matrix once the function is called.
    you can also have an offset set here in case you'd like to change the kanger function for the heatmap
    BEFORE you run and call your plotting function.

    IF YOU WANT TO JUST SEE THE ARRAYS, YOU CAN SIMPLY RUN THE FOLLOWING FUNCTION IN THE TERMINAL AS SUCH;
    heatsolver()

    this temp_kanger() function only has the computations needed for the average temperature profile.

    Returns
    ---------------------
    after calling the function, a matrix of temperatures will be returned in the terminal. simply run the function to return the matrix array!
    we want to return the amplitude of the functions along with their respective means.
    '''
    offset = 0
    t_amp = (adjusted_t_kanger - adjusted_t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + adjusted_t_kanger.mean()

def heatsolver(xstop = 100, tstop = 365*75, dx = .9, dt = .9, c2 = .0216, lowerbound = temp_kanger, upperbound = 5, initial = 0):
    '''
    A function for solving the heat equation.
    Apply Neumann boundary conditions such that dU/dx = 0.
    You can also change the boundary conditions so that the kanger solver coded in above will end up matching
    the conditions for the upperbounds as well. Keep in mind that these constants and step changes are 
    editable, granted our entire solver is based on this function.

    Parameters
    ----------
    c2 : float
        c^2, the square of the diffusion coefficient.

    initial : Numpy array or None
              The inital temperature profile. If None, the example problem's 
              profile 4x - 4x^2 is used. The default = 0. 

    upperbound, lowerbound : None, scalar, or func
        Set the lower and upper boundary conditions. If either is set to
        None, then Neumann boundary condtions are used and the boundary value
        is set to be equal to its neighbor, producing zero gradient.
        Otherwise, Dirichlet conditions are used and either a scalar constantp
        is provided or a function should be provided that accepts time and
        returns a value. 

    tstop: float or integer
    the amount of time, in days, it will take to reach stable equilibrium of the permafrost. this can easily be changed in order to
    account for the diverging heatmap graphical representation. this will eventually be changed to answer question 2, to see how the depth will change
    for the equilibrium state at 0 degrees celsius ...and how long it will take to get there.

    xstop: float or integer
    the depth of the ground layers. (in meters) we would like to keep this value generally constant, especially because our graphical
    representation would want to sink down to 100 meters deep....just for a clean, accurate representation.

    dx: float or integer
    the timestep of meters for each graph; the heat plot and the kanger curve for the sake of resolution will be generally constant as well.

    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    '''

    # Check our stability criterion:
    dt_max = dx**2 / (2*c2) / dt
    if dt > dt_max:
        raise ValueError(f'DANGER: dt={dt} > dt_max={dt_max}.')

    # Get grid sizes (plus one to include "0" as well.)
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1

    # Set up space and time grid:
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create solution matrix
    U = np.zeros([M, N])

    # Set initial conditions
    if initial is None:
        U[:,0] = 4*x - 4*x**2 # default initial condition based on example problem
    else:
        U[:,0] = initial

    offset = 0 #in degrees celsius.
    U[:,0] = U[:,0] + offset #apply the offset conditions TO ALLLLL figures in order to change for different initial temperature 

    # Get our "r" coeff:
    r = c2 * (dt/dx**2)

    # Solve our equation!
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])

        # Apply boundary conditions:
        # Lower boundary
        if lowerbound is None:  # Neumann
            U[0, j+1] = U[1, j+1]
        elif callable(lowerbound):  # Dirichlet/constant
            U[0, j+1] = lowerbound(t[j+1])
        else:
            U[0, j+1] = lowerbound

        # Upper boundary
        if upperbound is None:  # Neumann
            U[-1, j+1] = U[-2, j+1]
        elif callable(upperbound):  # Dirichlet/constant
            U[-1, j+1] = upperbound(t[j+1])
        else:
            U[-1, j+1] = upperbound


    # Return our pretty solution to the caller:
    return t, x, U


def plot_heatsolver(t, x, U, title=None, **kwargs):
    '''
    Plot the 2D solution for the 'heat_solver' function.

    Extra kwargs handed to pcolor.

    Paramters
    ---------
    t, x : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    title : str, default is None
        Set title of figure.

    Returns
    -------
    fig, ax : Matplotlib figure & axes objects
        The figure and axes of the plot.

    map : Matplotlib color bar object AND map that will be produced
        The color bar on the final plot
    '''

    # Check our kwargs for defaults:
    # Set default cmap to hot
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'

    # Create and configure figure & axes:
    # Create a figure/axes object
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (18,9))
    # Create a color map and add a color bar.
    map = ax1.pcolor(t, x, U, cmap='seismic', vmin=-25, vmax=25)
    ax1.invert_yaxis()
    fig.colorbar(map, ax=ax1, label='Temperature ($C$)')
    ax1.set_xlabel('Time Elapsed (days)')
    ax1.set_ylabel('depth (meters)')
    ax1.set_title('Spatial Heatmap of Kangerlussuaq')
    # Set indexing for the final year of results:
    dt = .2
    loc = int(-365/dt) # Final 365 days of the result.
    # Extract the min values over the final year:
    winter = U[:, loc:].min(axis=1)
    summer = U[:, loc:].max(axis=1)
    # Create a temp profile plot:
    #fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.invert_yaxis()
    ax2.plot(winter, x, label='Winter')
    ax2.plot(summer, x, '--', label = 'summer difference')
    ax2.set_xlabel('Degrees ($C$)')
    ax2.set_ylabel('Depth in $meters$')
    ax2.set_title('Depth profile of Kangerlussauq Permafrost')
    plt.legend(loc = 'best')
    plt.show()
    return fig, ax1, ax2, map

def validate_solver():
    '''
    This function validates the heat equation solver by comparing its solution
    element-by-element to the example problem solution provided in the Lab 3 pdf.
    This validation function has no returns but instead prints whether or not the 
    solver was able to be validated successfully.

    Parameters
    ---------
    None

    Returns
    -------
    None
    '''

    # Define variable values (based on example problem)
    xstop=1
    tstop=0.2
    dx=0.2
    dt=0.02
    c2=1
    lowerbound=0
    upperbound=0

    # Call the function and store the returns
    t, x, U = heatsolver(xstop=xstop, tstop=tstop, dx=dx, dt=dt, c2=c2, lowerbound=lowerbound, upperbound=upperbound, initial=None)

    # Check that array shapes match first before validating
    if U.shape == sol10p3.shape:
        print('Shape of heatsolver() result matches shape of desired solution.')
        
        # Compare the solution values in U and sol10p3 to see if they match
        if np.allclose(U, sol10p3, atol=1e-5): # atol is the allowed tolerance to account for rounding errors
            print('heatsolver() function validated!!!\nAll values calculated from the solver match the given solution values.\n')
        else:
            print('Mismatching values found.\n')
    
    else:
        raise ValueError('Shape of heatsolver() result DOES NOT match shape of desired solution. Cannot validate.')