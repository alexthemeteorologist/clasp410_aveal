#!/usr/bin/env python3
'''
this script is written in efforts to create models and function definitions to create plots regarding Lotka- Volterra competition models, predator-prey models
and comparing phase diagrams in an effort to see what equilibrium states we can accomplish with laid out equations that are already defined and constructed.
there are a few function definitions in order to just define time arrays, derivative functions, and plotting at the end for our representation.

Question 1: in order to generate the standard plot as shown in the example modelA AFTER running the script, type in the terminal; makeplots()
and the initial conditions will show as given in the example model.

in order to change timesteps, as instructed in the lab handout to show proof...you must then change the parameters of dT inside the function definitions
of euler_solve and solve_rk8 in order to mirror the changes in timestep across both methods for the competition model. bear in mind, the standard dT for
the competition model is 1 year, as specified in the lab handout.

the way to change the dT for the predator-prey model is to directly modify variable 'dT' inside of the actual makeplots() function BEFORE running it and calling it. 
dT is expressed as a variable inside of a function here, which means it will be referenced inside of the outer function definition for ease of change once called.
in example, i chose a dT value of .1 for BOTH models to compare them, and you can reflect those changes in both areas of modification listed above for ease of access.

-----------------------------
Question 2: to generate the standard plot as shown in the example model, simply type in the terminal; makeplots()
in order to see the equilibrium state, you must edit the function definitions for BOTH dNdt_comp AND solve_rk8 to reflect the following coefficients.

a = 2, b = 1, c = 2, d = 1.

this will cause the competition model to go into stable equilibrium for both species AND both methods of differential equation solving. 
both graphs have been attached to reflect such changes, and will be explained qualitatively/quantitatively with evidence to back results up. 

-----------------------------
Question 3: run the script, then call makephaseplots() to generate the phase diagrams.

this question specifically deals with function makephaseplots() and its variables inside the function.
in order to generate the standard plot, that has already been encoded in to remain unchanged. type in makephaseplots(), and the standard 
phase diagram will come up regardless of your other choice of input.

in order to change the parameters of N1 & N2 coefficients, modify the first and second parameters, respectively for 
compare_phasetime, compare_phaseN1euler, compare_phaseN2euler, = euler_solve(predatorpreymodel, N1,N2,dT)
and the change will be reflected in the phase diagrams comparing the population changes with respect to each species. 

testing different N1, N2, and dT values in the comwill produce different phase graphs, as additionally outlined with the color coding in legend.
for some simplicity, i have already tested a different set of coefficients on the same axes as the standard example model. 

Collaborators: Tyler Overbeek and Katherine Paff.
'''
#bear in mind this is the Lotka-Volterra model for the two species, as defined in the lab pdf.
#but first, we must import standard python libraries as usual.
import numpy as np
import matplotlib.pyplot as plt
#create a function that imports the specific equations using the four-letter parameters outlined below in order to satisfy competition model
def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    Lab 1 - 2
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
    The current time (not used here).
    N : two-element list
    The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    #the variable dN1dt represents how much our step we will utilize for the competition plot.
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]
    return dN1dt,dN2dt
#the secondary function definition is written as follows;
#use the derivation from euler's method in order to create initial conditions and timestep conditions.
def euler_solve(func, N1_init=.5, N2_init=.5, dT=1, t_final=100.0):
    '''
    euler_solve is designed to have a specific encoded method via Euler's method in order to take every step (dT) of recorded population proportion (N1 & N2)
    for each species and calculate the actual value for each. We want to record an initial condition of population for both species, and we should return a 
    100-year time array with populationn values (N1 & N2) for each specific interval and change of time for each model we decide to run. one of the N values will end
    at 1 after the 100-year simulation completes, and the other species will die and diminish to 0.....unless there is a combination of factors that provides 
    an equilibrium solution! varying the parameters will help us find out.

    IN ORDER TO CHANGE THE DIFFERENT N1, and N2 initial conditions, you must run the script, then change the values in the terminal when calling function.

    in example, you can change initial conditions as such;
    euler_solve(dNdt_comp, .3,.6,1,100) 

    is different from this set of initial conditions
    euler_solver(dNdt_comp, .5,.5, 1, 100)

    and is also different from this set of initial conditions
    euler_solver(dNdt_comp, .3,.6, .35, 100)


    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init : initial value for the proportion of population that is occupied by the species defined as the prey,
    that is contextually the bunny specimen in this Lotka-Volterra competition model. If N1 = 0, bunnies would go extinct.

    N2_init : initial value for the proportion of the population that is defined as the predatory in the Lotka-Volterra competition model.
    We use N2 to denote the number of 'wolves' contextually in this model compared as a specimen that could potentially go extinct. 
    That is, N2 = 0
   
    dT: since our models run over a course of time, we want our time step of models to vary in order to get a more accurate solver. smaller dT values will give us
    a more accurate estimate over the course of our time interval from [0, t_final].

    t_final: the final year, or array of time we want to record for our population proportion in regards to bunnies and wolves.
    standard for this problem is 100, and should be left as such.

    Returns
    ------------
    returns time: all array of time elapsed in years, and can vary based on the dT time step.
    returns N1 : all array of outputs for our prey specimen
    returns N2: all array of outputs for our predator specimen

    '''
    #first, make your time array from 0, 100 years. dT is the variable that will change based on the step you desire.
    time = np.arange(0, t_final, dT)
    #create the array of all possible outcomes based on your amount of years/timesteps you desire
    N1 = np.zeros(time.size)
    #assigns the first population proportion as the initial value declared in the function definition.
    N1[0] = N1_init

    #the second array of time is constructed here
    N2 = np.zeros(time.size)
    #the initial condition for the predator species defined as N2[0] and variable N2_init
    N2[0] = N2_init


    for i in range(1, time.size): #this loop is required in order to iterate throughout all of the N1, N2 outputs and their derivatives
         dN1, dN2 = func(i, [N1[i-1], N2[i-1]] )
         N1[i] = N1[i-1] + dT * dN1
         N2[i] = N2[i-1] + dT * dN2
    #function is returning all time array values, and N1/N2 values at the end of the function being called.
    return time, N1, N2
#we now need a function for our rk-8 solver - lets create using the proper parameters and values.
def solve_rk8(func, N1_init=.5, N2_init=.5, dT=1, t_final=100.0,
 a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init, N2_init : float
    Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
    Largest timestep allowed in years.
    t_final : float, default=100
    Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
    Lotka-Volterra coefficient values
    Returns
    -------
    time : Numpy array
    Time elapsed in years.
    N1, N2 : Numpy arrays
    Normalized population density solutions.
    '''
    from scipy.integrate import solve_ivp
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
    args=[a, b, c, d], method='DOP853', max_step=dT)
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
    # Return values to caller.
    return time, N1, N2

#once we've finished the dNdT competition model, we now must replicate our predator-prey equations so we can euler 
#and rk-8 solve them when it's time to plot!
def predatorpreymodel(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    Lab 1 - 2
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
    The current time (not used here).
    N : two-element list
    The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b* N[0]* N[1]
    dN2dt = -c*N[1] + d*N[1]*N[0]
    return dN1dt,dN2dt
#time to create plots and run the equlibrium states!
def makeplots():
    '''
    the following is a plotting function with no neccesary inputs or return statements. we want to acquire two plots to compare the
    the predator-prey models and the competition models. additionally, we will plot both the euler and rk-8 graphs on both of their respective axes.
    additionally, we want to redefine our constants given that our euler_solve, and dNdt_comp all depended on the time array, that we ALSO wanted to plot later on.

    both plots will be created on a set of two graphs separately BUT with similar axes for reference.

    in order to call the function and create the plots, just specifically run in the terminal:
    makeplots()

    and an output of two graphs will appear.
    given the conditions entered in the assigning of arrays which holds the calling of 
    function euler_solve(dNdt_comp, N1, N2) or calling euler_solve(predatorpreymodel, N1, N2, dT)
    it is IMPORTANT AND IMPERATIVE to change the conditions inside of the function BEFORE running the script and generating the plots with makeplots().
    '''
    #re-define the final year (100 years) we should have time assigned for
    t_final = 100
    #redefine dT as a variable condition
    dT= .05
    #our time array redefined as such
    time = np.arange(0, t_final, dT)

    #the actual timearray for the euler-competition method is defined as eulertimearray
    #the euler N1 and N2 value arrays are set as variables as well ALONG WITH THEIR RESPECTIVE INITIAL CONDITIONS
    
    #the rk times, and initial N1 & N2 Values are also shown as parameters when we call our solver functions to plot them.
    eulertimearray , eulerN1, eulerN2 = euler_solve(dNdt_comp,0.3,0.6)
    rktimearray, rkN1, rkN2 = solve_rk8(dNdt_comp,0.3,0.6)

    #now, time to plot!
    fig, axes = plt.subplots(1,2, figsize = (18,9))
    # set an array now to plot the first set of information.
    # we want to plot the euler times vs both the N1 predatory and N2 prey specimen on one singular set of axes.
    axes[0].plot(eulertimearray,eulerN1, label = 'competition euler N1 output')
    axes[0].plot(eulertimearray,eulerN2, label = 'competition euler N2 output')
    #the same exact piece of information should be plotted on the same set of axes as well, however we should use dashed lines to differentiate methods.
    axes[0].plot(rktimearray, rkN1, '--', label = 'competition RK-8 N1 output', color = 'lightblue') 
    axes[0].plot(rktimearray,rkN2, '--', label = 'competition RK-8 N2 output', color = 'red')
    # lets take attention to labelling and style now.
    axes[0].set_title('Competition Model ; $Lotka-Volterra$')
    axes[0].set_xlabel('Timescale ($Years$)')
    axes[0].set_ylabel('Population Density of Specimen')
    axes[0].legend(loc='best')

    # we are also going to redefine the variables for the predator-prey model, very similar to the competition plots.
    predpreytime , predpreyN1euler, predpreyN2euler = euler_solve(predatorpreymodel,0.3,0.6,dT)
    predpreytimerk, predpreyN1rk, predpreyN2rk = solve_rk8(predatorpreymodel,0.3,0.6,dT)
    #plot the predator-prey models now to depict both species with respect to time.
    #this line covers the euler method's way of outputting the specimen
    axes[1].plot(predpreytime, predpreyN1euler, label = 'predator-prey euler N1 output') 
    axes[1].plot(predpreytime,predpreyN2euler, label = 'predator-prey euler N2 output')
    #plot the rk-8th order depiction model again using the predator-prey method of solving this time.
    axes[1].plot(predpreytimerk,predpreyN1rk,'--',label = 'predator-prey rk N1 output', color = 'lightblue')
    axes[1].plot(predpreytimerk,predpreyN2rk, '--', label = 'predator-prey rk N2 output', color = 'red')
    # take attention to the styling and axes labelling once again for the second set of plots.
    axes[1].set_xlabel('Timescale ($Years$)')
    axes[1].set_ylabel('Population Density of Specimen')
    axes[1].set_title('Predator-Prey Model ; $Lotka-Volterra$')
    axes[1].legend(loc='best')

    plt.style.use('fivethirtyeight')
    plt.tight_layout()
    plt.show()

def makephaseplots():
    '''
    this function also generates plots without any return values and no parameters to be entered inside of the function when calling it.
    RATHER , YOU MUST EDIT PARAMETERS AND CHANGE the N1 and N2 values for different phase diagrams to generate.
    in example, for euler_solve(predatorpreymodel, N1, N2, dT) values N1, N2 and dT must vary if we want different graphs for the phase diagrams.

    keep in mind that the paramters needed to change the values are INSIDE of the function definition. therefore, they must be changed and the program re-ran in order to 
    apply those changes to any figure or graph replicated and curated within this function.
    '''
    # keep these lines unmodified ; these are the standard graphs we are comparing our revised information to.
    predpreytime , predpreyN1euler, predpreyN2euler = euler_solve(predatorpreymodel,0.6,0.3,.05) # LEAVE THIS LINE UNCHANGED - gives standard graphic.
    predpreytimerk, predpreyN1rk, predpreyN2rk = solve_rk8(predatorpreymodel,0.6,0.3,.05) # LEAVE THIS LINE UNCHANGED - gives standard, example visual

    #MODIFY THE TWO LINES BELOW (last three parameters) to get different phase graphs and different models.
    compare_phasetime, compare_phase_N1euler, compare_phase_N2euler = euler_solve(predatorpreymodel,0.2,0.7,.05) #modify the last three parameters to change N1, N2 and dT timesteps, respectively.
    compare_phasetime_rk, compare_phase_N1rk, compared_phase_N2rk = solve_rk8(predatorpreymodel,0.2,0.7,.05) #modify the last three parameters to change N1, N2 and dT timesteps, respectively.
    # we want to plot a comparison condition and the original set of conditions on one set of axes
    # following that, we can compare figures for upload into the report.
    fig , ax = plt.subplots(1,1, figsize = (15,7))
    #plot the N1 conditions against the N2 conditions to see the actual lineup for both specimen - euler method
    ax.plot(predpreyN1euler, predpreyN2euler, label = 'standard phase initial cond. for N1') 
    # plot the N1 conditions against the N2 conditions to see lineup for both specimen simultaneously - rk8 method
    ax.plot(predpreyN1rk,predpreyN2rk, label = 'standard phase initial cond. for N2')

    # THESE ARE COMPARISON CONDITIONS - the last three parameters of N1 initial, N2 initial, and the timestep dT all changed. 
    ax.plot(compare_phase_N1euler, compare_phase_N2euler, linewidth = 2, label = 'compared phase N1 initial cond.' , color = 'pink' )
    ax.plot(compare_phase_N1rk, compared_phase_N2rk, linewidth = 2 , label = 'compared phase N2 initial cond.', color = 'purple')
    #finally, add some style to the plot.
    ax.set_xlabel('population density/quantity of N1 prey species')
    ax.set_ylabel('population density/quantity of N2 predatory species')
    ax.set_title('Phase Diagram of Prey vs. Predatory Species')
    plt.style.use('fivethirtyeight')
    plt.tight_layout()
    plt.legend(loc= 'best')
    plt.show()
