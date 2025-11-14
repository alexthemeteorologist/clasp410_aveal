#!/usr/bin/env python3
'''
Collaborators: Tyler Overbeek & Katherine Paff
Our forest fire and disease model is going to be analyzed below.

Question 1 can be ran specifically by running the script first. Then, you will type into the terminal the conditions
of the forest, with percentage bare, ignition and spreading possibilities.

type into the terminal run lab04.py, then type in the forest conditions
forest = forest_fire(pbare=0, isize=3, jsize = 3, pignite = 0)

then type in the terminal     
forest
and an array of the fire spread will appear as defined. 

for the larger matrix, i typed in the terminal after i re-ran the program;
forest = forest_fire(pbare=0, isize=4, jsize = 10, pignite = 0)
then type
forest

and you will see the array reprinted out as well with this new grid shown. additionally, the steps will also be shown from iteration 0 to 1 and 1 to 2.

Question 2 has a few different elements needed to run these simulations: LEAVE pignite CONSTANT at .2 for ALL DIFFERING VALUES
first plot with pspread changing, just run in the terminal the same pignite as .2, BUT CHANGE THE PSPREAD VALUE IN THE FUNCTION DEFINITION
as pspread = .09,
as pspread = .5,
as pspread = .8,
and as pspread = .96 ALL TO REPLICATE THE DIFFERENT OUTCOMES of spreading probability.
the graphs that show progression are also attached in the adjacent report.

you will also see the comparisons numerically of fire spreads for the differences of pbare as well. 
all you will do for this one is change what is ran in the terminal.

for each different pbare spread: you will now change in the terminal accordingly;
forest = forest_fire(pbare=.15, isize=9, jsize = 9, pignite = .2)
forest = forest_fire(pbare=.5, isize=9, jsize = 9, pignite = .2)
forest = forest_fire(pbare=.8, isize=9, jsize = 9, pignite = .2)

and then it will replicate exactly the progression shown over a 5-period time interval; reminder that pignite is remaining constant in this case.

Question 3 has one more different element; this time, just simply run in the terminal;
forest = forest_fire(pfatal = .5, pbare=.3, isize = 9, jsize=9, pignite = .2)

this will generate the grid based on different time steps, and you can increase time steps using the plot_forest2d in order to see the completed timestep plots.
'''
#first, import standard python libraries.
import numpy as np
#an intermediate library needed would also entail the random function for our grid spaces.
from numpy.random import rand
import matplotlib.pyplot as plt

#declare our function
def forest_fire(nstep=5, isize=3, jsize=3, pfatal = .5 , pignite = 0.0, pspread = 1.0, pbare = 0.15):
    '''
    Create a forest fire.
    Also, make a specific map that would be called by simply assigning the conditions to an array called

    forest

    Parameters
    ----------
    pfatal , pignite : float between 0 and 1
        these are probabilities that our population BEGINS DEAD at the beginning of our simulation
    isize, jsize : int, defaults to 3
        Set size of forest in x and y direction, respectively.
    nstep : int, defaults to 4
        Set number of steps to advance solution.
    pspread : float, defaults to 1.0
        Set chance that fire can spread in any direction, from 0 to 1
        (i.e., 0% to 100% chance of spread.)
    pbare: float , ranges from 0 and 1
        can be used to define an immune population or one that has already been burned in the wildfire context
    '''
    # Creating a forest and making all spots have trees.
    forest = np.zeros((nstep, isize, jsize)) + 2.0

    #print('INITIAL FOREST:')
    #print(forest)

    if pignite > 0.0: #scatter the fire randomly :)
        print(f'Setting fires randomly using pignite={pignite}')
        loc_ignite = np.zeros((isize, jsize), dtype = bool)
        while loc_ignite.sum() == 0: # keep in mind the randon fire spreading function
            loc_ignite = rand(isize, jsize) <= pignite
        print(f'Starting with {loc_ignite.sum()} points on fire')
        forest[0, loc_ignite] = 3
    else:
        #set initial fire to center:
        forest[0,isize//2,jsize//2] = 3
        print('Lighting center of forest on fire')
    #the proportion of locations that are bare as well.
    loc_bare = rand(isize, jsize) <= pbare
    forest[0, loc_bare] = 1



    # Loop through time to advance our fire.
    for k in range(nstep-1):
        # Assume the next time step is the same as the current:
        forest[k+1, :, :] = forest[k, :, :]
        # Search every spot that is on fire and spread fire as needed.
        for i in range(isize):
            for j in range(jsize):
                # Are we on fire?
                if forest[k, i, j] != 3:
                    continue
                # Ah! it burns. Spread fire in each direction.
                # Spread "up" (i to i-1)
                if (pspread > rand()) and (i > 0) and (forest[k, i-1, j] == 2):
                    forest[k+1, i-1, j] = 3
                # Spread "Down" (i to i+1)
                if (pspread > rand()) and (i > 0) and (i < isize-1) and (forest[k, i+1, j] == 2):
                    forest[k+1, i+1, j] = 3
                # Spread "East" (j to j-1)
                if (pspread > rand()) and (i > 0) and (forest[k, i, j-1] == 2):
                    forest[k+1, i, j-1] = 3
                # Spread "West" (j to j+1)
                if (pspread > rand()) and (i > 0) and (j < jsize-1) and (forest[k, i, j+1] == 2):
                    forest[k+1, i, j+1] = 3
                if pfatal > rand():
                    forest[k+1,i,j] = 0
                # Change burning to burnt:
                forest[k+1, i, j] = 1

    return forest


def plot_progression(forest):
    '''
    calculate the time dynamic of forest fires and now plot them on a linear set of axes.
    three plot lines will be made on a single set of axes;

    parameters
    --------------------
    forest: is the only parameter used to call our progression plot and refer to this matrix of data.
    '''
    #dead indviduals is represented by pfatal
    ksize, isize, jsize = forest.shape
    npoints = isize*jsize
    #find all spots that are forested or healthy
    loc_forested = forest == 2
    forested = 100* loc_forested.sum(axis=(1,2)) /npoints

    # bare locations total
    loc_bare = forest == 1
    bare = 100* loc_bare.sum(axis=(1,2)) / npoints
    #burning loations total
    loc_burning = forest == 3
    burning = 100* loc_burning.sum(axis=(1,2)) / npoints

    #plt.title(f'Progression Plot of 10x10 coverage over 5 iterations')
    plt.plot(forested, label ='forested', color = 'darkgreen')
    plt.plot(bare, label = 'bare', color = 'tan')
    plt.plot(burning, label= 'burning its butt off', color = 'red')
    plt.xlabel('time (in some arbitrary units)')
    plt.ylabel('percentage total forest')
    plt.title('Progression of pignite = .2 & pbare = .8')
    plt.legend()
    plt.show()

def plot_forest2d(forest, itime=8):
    '''
    Given a forest of size (ntime, nx, ny), plot the itime-th moment as a
    2d pcolor plot.
    '''

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig.subplots_adjust(left=.117, right=.974, top=.929, bottom=0.03)
    from matplotlib.colors import ListedColormap
    forest_cmap = ListedColormap(['black', 'tan', 'darkgreen', 'crimson'])
    # Add our pcolor plot, save the resulting mappable object.
    map = ax.pcolor(forest[itime, :, :], vmin=0, vmax=3, cmap=forest_cmap)

    # Add a colorbar by handing our mappable to the colorbar function.
    cbar = plt.colorbar(map, ax=ax, shrink=.8, fraction=.08,
                        location='bottom', orientation='horizontal')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Death','Bare/Burnt/Immune', 'Forested/Healthy', 'Burning/Infected'])

    # Flip y-axis (corresponding to the matrix's X direction)
    # And label stuff.
    ax.invert_yaxis()
    ax.set_xlabel('Eastward ($km$) $\\longrightarrow$')
    ax.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax.set_title(f'Our Forest Mapped at T={itime}')

    # Return figure object to caller:
    return fig

def make_all_2dplots(forest, folder='results/'):
    '''
    For every time frame in `forest_in`, create a 2D plot and save ther image
    in folder.
    '''

    import os

    # Check to see if folder exists, if not, make it!
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Make a buncha plots.
    ntime, nx, ny = forest.shape
    for i in range(ntime):
        print(f"\tWorking on plot #{i:04d}")
        fig = plot_forest2d(forest, itime=i)
        fig.savefig(f"{folder}/forest_i{i:04d}.png")
        plt.close('all')
