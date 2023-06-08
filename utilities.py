from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from global_constants import *
import time
from IPython.display import HTML
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def fcc(num_cells, rho):
    """
    Initializes a system of atoms on an FCC (face-centered cubic) lattice.

    Parameters
    ----------
    num_cells : int
        The number of unit cells in each dimension
    rho : float
        The density of the system

    Returns
    -------
    pos_vec : np.ndarray
        Array of particle coordinates
    """
    
    num_atoms = 4 * num_cells**3
    

    l = (4.0/rho)**(1/3)    # lattice constant
    
    pos_vec_unit = np.array(
        [[0.0, 0.0, 0.0], [l / 2, l / 2, 0.0], [l / 2, 0.0, l / 2], [0.0, l / 2, l / 2]]
    )
    pos_vec = []

    for i in range(num_cells):
        for j in range(num_cells):
            for k in range(num_cells):
                for n in range(4):
                    pos = pos_vec_unit[n] + np.array([i, j, k]) * l
                    pos_vec.append(pos)

    return np.array(pos_vec)


def gaussian_velocity(num_atoms, T):
    """
    Initializes the velocities on each dimension of all the atoms in the system which obey
    a Maxwell-Boltzmann distribution at temperature T.
    The velocity on each dimension of the atom is given by a gaussian distribution with temperature
    unitless T, and the velocity of center of mass (velocity_com) is set to zero

    Parameters
    ----------
    num_atoms : int
        The total number of particles in the system
    T : float
        The temperature of the system

    Returns
    -------
    velocity_array : np.ndarray, shape = (num_atoms, dim)
        Array of initial atom velocities in each coordinate
    """
    velocity_array = np.random.normal(0, (T) ** 0.5, size=(num_atoms, dim))  # initiate velocity
    velocity_com = np.mean(velocity_array, axis=0)
    velocity_array -= velocity_com
    return velocity_array


def rel_pos_dist(pos_t, L):
    """
    Calculates the relative position and distance between atoms with a given absolute position array

    Parameters
    ----------
    pos_t : ndarray, shape = (num_atoms, dim)
        The position of each atoms in the system at a single time step
    L: float
        The length of the simulation box in each dimension

    Returns
    -------
    rel_pos : np.ndarray, shape = (num_atoms, num_atoms, dim)
        The relative position between each pair of atoms in all dimensions
    dist:  np.ndarray, shape = (num_atoms, num_atoms)
        The scalar (absolute) distance between each pair of atoms
    """
    rel_pos = (pos_t[:, np.newaxis, :] - pos_t[np.newaxis, :, :] + L/2) % L - L/2
    dist = np.linalg.norm(rel_pos, axis=2)
    return rel_pos, dist


def cal_force(rel_pos_t, dist_t):
    """
    Calculates the total force on each atom of the system, based on pairwise Lennard-Jones interactions.

    Parameters
    ----------
    rel_pos_t : np.ndarray, shape = (num_atoms, num_atoms, dim)
        The relative position between each pair of atoms in each dimensions at a single timestep
    dist_t : np.ndarray, shape = (num_atoms, num_atoms)
        The scalar (absolute) distance between each pair of atoms at a single timestep

    Returns
    -------
    F: np.ndarray, shape = (num_atoms, num_atoms, dim)
        The force vector of each pair of two atoms in all dimensions at a single timestep
    """
    dist_t = np.tile(dist_t[:,:,np.newaxis], (1,1,3))
    dU = 4 * (-12 * dist_t ** (-13) + 6 * dist_t ** (-7))
    f = - dU * rel_pos_t / dist_t
    f = np.nan_to_num(f, copy=False, nan=0)
    F = np.sum(f, axis=1)
    return F


def cal_next_pos_vel_force(i, pos_array, vel_array, force_array, L):
    """
	Parameters
    ----------
    i : int
        The index of the atom in the system
    pos_array : np.ndarray, shape = (num_atoms, dim)
        Array of particle coordinates at the current time step
    vel_array : np.ndarray, shape = (num_atoms, dim)
        Array of particle velocities at the current time step
    force_array : np.ndarray, shape = (num_atoms, dim)
        Array of particle forces at the current time step
    L : float
        The length of the simulation box in each dimension

    Returns
    -------
    pos_array : np.ndarray, shape = (num_atoms, dim)
        Array of particle coordinates at the next time step
    vel_array : np.ndarray, shape = (num_atoms, dim)
        Array of particle velocities at the next time step
    force_array : np.ndarray, shape = (num_atoms, dim)
        Array of particle forces at the next time step
	d_i : np.ndarray, shape = (num_atoms, num_atoms)
		Array of pairwise distances between the particles (considering minimal image convention)
    """
    
    pos_array[i] = (pos_array[i - 1] + time_step * vel_array[i - 1] + time_step**2 / 2 * force_array[i - 1])
    rel_pos_i, d_i = rel_pos_dist(pos_array[i], L)

    force_array[i] = cal_force(rel_pos_i, d_i)
    pos_array[i] = pos_array[i] % L

    vel_array[i] = vel_array[i - 1] + time_step / 2 * (force_array[i - 1] + force_array[i])
    
    return pos_array, vel_array, force_array, d_i


def summ_dUr(dist_t):
    """
    Calculates the summation term on the pressure function, as per the virial theorem, of a particular timestep.

    Parameters
    ----------
    dist_t: np.ndarray, shape = (num_atoms, num_atoms)
        The matrix of distances between atoms (considering minimal image convention)

    Returns
    -------
    dU_r: np.ndarray, shape = (num_atoms, num_atoms)
        The array witht he result from the summation in the virial's theorem.
    """

    dU = 4*(-12*dist_t**(-13) + 6*dist_t**(-7))
    dU = np.nan_to_num(dU, copy=False, nan=0)
    dU_r = np.sum(dU*dist_t)

    return dU_r


def potential(dist_t):
    """
    Calculates the Lennard Jones potential from pairwise interactions in the system given the system's configuration.

    Parameters
    ----------
    dist_t: np.ndarray, shape = (num_atoms, num_atoms)
        The matrix of distances between atoms (considering minimal image convention)

    Returns
    -------
    U: np.ndarray, shape = (num_atoms, num_atoms)
        The scalar potential of each pair of two atoms at a single timestep
    """
    u_ij = 4 * (dist_t**(-12) - dist_t**(-6))
    u_ij = np.nan_to_num(u_ij, copy=False, nan=0)
    U = 0.5 * np.sum(u_ij)
    return U



def kinetic(vel_t):
    """
    Calculates the kinetic energy at a single timestep

    Parameters
    ----------
    vel_t : ndarray, shape = (num_atoms, dim)
        The velocity vector of all the atoms in the system at a single time step

    Returns
    -------
    np.ndarray: shape = (num_atoms)
        The kinetic energy of all the atoms in the system at a single timestep
    """
    return np.sum(vel_t**2) / 2


def E_plot(rho, T, N, K, pot, U, Ekin, steps):
	"""
	Plots the time evolution of the kinetic, potential and total energy of the system

    Parameters
    rho : float
        The density of the system
    T : float
        The temperature of the system
	K : np.ndarray, shape = (steps)
        The kinetic energy of the system over all timesteps
	pot : np.ndarray, shape = (steps)
        The potential energy of the system over all timesteps
	U : np.ndarray, shape = (steps)
        The total energy of the system over all timesteps
	N : float
		Number of particles in the system
	steps : float
		Number of timesteps to simulate over

    """

	fig = plt.figure(figsize=(12,6))
	
	time_array = np.linspace(0, steps * time_step, steps)

	plt.plot(time_array, K / N, label="$K$", lw=1)
	plt.plot(time_array, pot / N, label=r'$U_{LJ}$', lw=1)
	plt.plot(time_array, U / N, lw=1, label = '$E$')
	plt.axhspan(0.9 * Ekin / N, 1.1 * Ekin / N, alpha=0.15, color="blue")

	plt.vlines(
		x=10,
		ymin=min(pot / N),
		ymax=max(K / N),
		color="r",
		linestyle="--",
		lw=0.5,
	)
	plt.legend(loc="best")
	plt.title(r"{} particles with $\rho$ ={}, $T$ ={}".format(N, rho, T))
	plt.ylabel('Energy per particle')
	plt.xlabel(r'$\tau$')


def simulate(num_cells, rho, T, steps, rescale, energy_plot=True, animation=False):
    """

    Simulates the evolving system with a given initial positions, and initial velocities of all the atoms.
    If rescale is True, we rescale the velocity such that the kinetic energy is rescaled to a value close to
    the theoretical value. The rescaling condition check is set to be every 20 unit steps.
    Referring to the book, we can turn rescaling off safely after 10 unit time and say the system enters equilibrium.

    Parameters
    ----------
 num_cells : int
        The number of unit cells in each direction (x, y, and z) of the simulation box
    rho : float
        The density of the system
    T : float
        The temperature of the system
    steps : int
        The number of time steps to simulate
    rescale : bool, optional
        Whether to turn on the velocity rescaling to adjust kinetic energy to the theoretical value.
        If True, rescaling will be performed at regular intervals.
        If False, the simulation will run without velocity rescaling. Default is True.

    Returns
    -------
    tuple
        A tuple containing:
        - N: int, the total number of atoms in the simulation
        - T: float, the temperature of the system
        - pos_array : np.ndarray, shape = (steps, N, dim)
            The position coordinates of all the atoms in the system over all timesteps
        - vel_array: np.ndarray, shape = (steps, N, dim)
            The velocity vectors of all the atoms in the system over all timesteps
        - K : np.ndarray, shape = (steps)
            The kinetic energy of the system in equilibrium
        - sum_dUr : np.ndarray, shape = (steps)
            The cumulative potential energy difference of the system in equilibrium
		- ani3d :
    """

    N = 4*num_cells**3
    L = (4.0/rho)**(1/3) * num_cells
    Ekin = 3.0 / 2.0 * float(N - 1) * T
    
    pos_array, vel_array, f_array = [np.zeros((steps, N, 3)) for i in range(3)]
    pos_array[0] = fcc(num_cells, rho)
    vel_array[0] = gaussian_velocity(N, T)
    rel_pos0, d0 = rel_pos_dist(pos_array[0], L)
    f_array[0] = cal_force(rel_pos0, d0)
    
    K, pot, sum_dUr = [np.zeros(steps) for i in range(3)]
    K[0], pot[0], sum_dUr[0] = kinetic(vel_array[0]), potential(d0), summ_dUr(d0)

    n0 = 0
    ani3d = 0

    for i in tqdm(range(1, steps)):
        
        pos_array, vel_array, f_array, d_i = cal_next_pos_vel_force(i, pos_array, vel_array, f_array, L)
        
        sum_dUr[i], K[i], pot[i] = summ_dUr(d_i), kinetic(vel_array[i]), potential(d_i)
        
        
        if rescale:            # rescaling criteria from Jos Thijssen's book
            if i%20==0:
                re_factor = np.sqrt(Ekin * 2 / np.sum(vel_array[i] ** 2))
                vel_array[i] *= re_factor
                
            if i*time_step == 10.0:
                n0 = i
                rescale = False

    U = K + pot


    if energy_plot:
        E_plot(rho, T, N, K, pot, U, Ekin, steps)  

    if animation:
        print('animating')
    
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
    
        def animate3d(i):
            ax3d.clear()
            ax3d.set_xlim(0, L)
            ax3d.set_ylim(0, L)
            ax3d.set_zlim(0, L)
            ax3d.scatter(pos_array[i, :, 0], pos_array[i, :, 1], pos_array[i, :, 2], 'b')


            # Create the animation object
        ani3d = FuncAnimation(fig3d, animate3d, frames=steps, interval=10)
        plt.close(fig3d)
            
    return N, T, pos_array, vel_array, K[n0:], sum_dUr[n0:], ani3d


def pressure(sum_dUr, T, N):
    """
    Parameters
    ----------
    sum_dUr : ndarray, shape = (num_steps)
        The sum term obtained from the virial theorem
    T : float
        The temperature of the system
    N : int
        The number of particles in the system

    Returns
    -------
    p : float
        The average pressure of the system
    """
    
    p_t = np.zeros(len(sum_dUr))
    
    def pressure_t(sum_dUr):
        p_t = 1 - (1/(3*T*N) * sum_dUr/2)
        return p_t
    
    for i in range(len(p_t)):
        p_t[i] = pressure_t(sum_dUr[i])
                
    return np.mean(p_t)



def cv(K_eq, N):
    """
    Calculates the specific heat observable after system reaches equilibrium

    Parameters
    ----------
    K_eq : np.ndarray, shape = (num_equi_steps, )
        The kinetic energy over time after reaches equilibrium

    Returns
    -------
    cv : float
        The specific heat of the system
    """
    dK = np.var(K_eq)
    cv = (3*np.mean(K_eq)**2) / (2*np.mean(K_eq)**2 - 3*N*dK) 
    
    return cv


def block_data_error(data, b):
    """
    calculates the block-averaged version of an observable A at each timestep

    Parameters
    ----------
    data : np.ndarray, shape = (steps,1)
        A physical observable among all time steps (e.g. observable can be pressure, specific heat)
    b: int
        The size of the block used in averaging the observable

    Returns
    -------
    a_avg : np.ndarray, shape = (num_block, 1)
        The block-averaged value of this observable
    a_error: float
        The block data error of this observable
    """
	
    num_blocks = len(data) // b
    data = data[:num_blocks*b]  # truncate data to have integer number of blocks

    data = data.reshape((num_blocks, b))
    a_avg = np.mean(data, axis=1)
    a_error = np.sqrt(1 / (num_blocks - 1) * np.var(a_avg, axis=0))
    
    return a_avg, a_error


def block_bootstrap_observable(observable_func, data, num_datasets, b):
    """
    Computes the average value and statistical error of the observable of choice.

    Parameters
    ----------
    observable_func: function
        the function that computes the value of the observable of choice ('cv' or 'pressure')
    num_datasets: int
        the number of random uncorrelated sample sets we want to make
    b : int
        the size of the blocks for bootstrapping

    Returns
    -------
    avg_observable: float
        The averaged value of the observable calculated with block bootstrap
    sigma: float
        The statistical error of this result.
    """

    if (observable_func != pressure and observable_func != cv):
        raise ValueError("Invalid observable function")
        
    
    blocks, error = block_data_error(data[0], b)
    num_blocks = len(data[0]) // b
    samples_observable = np.zeros(num_datasets)
    
    for j in range(num_datasets):
        uncorr_data = blocks[np.random.choice(range(num_blocks), size=len(data[0]))]
        samples_observable[j] = observable_func(uncorr_data, *data[1:])

    avg_observable = np.mean(samples_observable)
    error = np.std(samples_observable) # or std?

    return avg_observable, error


def animate3d(i):
    """
    Creates a FuncAnimation to animate 3D movements of atoms for i th frame
    
    Parameters:
    -----------
    i : int
        The i th frame as index

    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        The animation object for the ith frame
    """

    ax3d.clear()
    ax3d.set_xlim(0, L)
    ax3d.set_ylim(0, L)
    ax3d.set_zlim(0, L)
    ax3d.scatter(pos[i, :, 0],pos[i, :, 1], pos[i, :, 2], 'b')

  


 # functions for Euler convention, used in a different code setup 

def cal_pos_vel_euler(i, pos_t, vel_t, L):
    """
    Calculates the position of atom i over all time steps using Euler method.

    Parameters
    ----------
    i: int
        the index i of the particle in the whole particle list
    pos_t : np.ndarray
        the starting position of the particle i
    vel_t : np.ndarray
        the starting velocity of the particle i
    L : float
        the length of the simulation box

    Returns
    -------
    pos_t: np.ndarray
        The position of particle i at each time step
    vel_t: np.ndarray
        The velocity of particle i at each time step
    """
    # calculate the new position at time step i using Euler method
    pos_t[i] = pos_t[i-1] + vel_t[i-1]*time_step
    
    # apply periodic boundary conditions to the new position
    pos_t[i] = (pos_t[i] + L/2.0) % L - L/2.0
    
    # calculate the new velocity at time step i using Euler method and the force at the previous time step
    vel_t[i] = vel_t[i-1] + force(pos_t[i-1],L)*time_step
    
    return pos_t, vel_t



def simulate_euler(init_pos, init_vel, L):
    """
    Simulates the motion of all particles over all time steps using Euler method.
    The animation is presented in 2D with box length 0 to L

    Parameters
    ----------
    init_pos : np.ndarray
        the initial position of all the particles
    init_vel : np.ndarray
        the initial velocity of all the particles
    L : float
        The length of the simulation box

    Returns
    -------
    pos_array: np.ndarray
        The position of all the particles at each time step
    vel_array: np.ndarray
        The velocity of all the particles at each time step
    K:np.array
        The kinetic energy of all the particles at each time step
    pot:np.array
        The potential of all the particles at each time step
    U:np.array
        The total energy of all the particles at each time step
    ani2d: FuncAnimation
        An animation showing the motion of particles in 2D at each time step
    """
    pos_array = np.zeros((steps, len(init_pos), 3))  # array to store positions of particles over time
    pos_array[0] = init_pos # set initial positions of particles
    vel_array = np.zeros((steps, len(init_pos), 3)) # array to store velocities of particles over time
    vel_array[0] = init_vel # set initial velocities of particles
    
    K = np.zeros(steps) # array to store the kinetic energy of particles over time
    K[0] = kinetic(vel_array[0]) # set initial kinetic energy of particles
    pot = np.zeros(steps) # array to store the potential of particles over time
    pot[0] = potential(pos_array[0], L) # set initial potential of particles
    
    for i in range(1, steps): # loop over all time steps
        pos_array, vel_array = cal_pos_vel_euler(i, pos_array, vel_array, L) # calculate position and velocity in each time step
        K[i] = kinetic(vel_array[i]) # calculate kinettic energy in each time step 
        pot[i] = potential(pos_array[i], L) # calculate potential in each time step
        

    # 2D Animation              
    fig2d = plt.figure(figsize=(5, 5))
    ax2d = plt.subplot(1, 1, 1)

    def animate2d(i):
        ax2d.clear()
        ax2d.set_xlim([0, L])
        ax2d.set_ylim([0, L])
        ax2d.scatter(pos_array[i, 0, 0], pos_array[i, 0, 1], s=10)
        ax2d.scatter(pos_array[i, 1, 0], pos_array[i, 1, 1], s=10)


    ani2d = animation.FuncAnimation(fig2d, animate2d, frames=steps,
                                  interval=10, blit=False, repeat=False)

    plt.close(fig2d)
    # ani.save('animation.mp4',writer='ffmpeg')
        
        
    # Calculate the total energy.   
    U = K + pot
    
    return pos_array, vel_array, K, pot, U, ani2d