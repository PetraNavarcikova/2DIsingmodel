import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from scipy.optimize import curve_fit

def initialize_lattice(n):
    """
    Randomly initialises an n x n  Ising spin lattice with spins (+1 or -1).

    Parameters: 
    -----------
    n: int
        Lattice size (in 1dim).
    
    Returns:
    ----------
    lattice: 2d array
        Ising spin lattice with size n x n.
    """
    lattice = np.random.choice([-1, 1], size=(n, n))
    return lattice

def impurities(lattice, concentration_percent, seed=None):
    """
    Randomly sets a percentage of spins in the lattice to 0, simulating impurities.

    Parameters:
    ------------
    lattice: np.ndarray
        2D array representing the spin lattice (e.g. values of +1 and -1).
    concentration_percent: float 
        Desired impurity concentration (between 0 and 100).
    seed: int (optional)
        If provided, sets the random seed for reproducible impurity placement.

    Returns:
    ------------
    lattice: np.ndarray
        A new 2D lattice with the same shape, but with some elements set to 0 (impurities).
    """

    # Sanity check: ensure the concentration is a valid percentage
    if not (0 <= concentration_percent <= 100):
        raise ValueError("Concentration must be between 0 and 100")

    # Set the seed for reproducibility, if specified
    if seed is not None:
        np.random.seed(seed)

    # Create a copy of the input lattice to avoid modifying the original
    new_lattice = lattice.copy()

    # Total number of spins in the lattice
    total_sites = new_lattice.size

    # Number of sites to be set to 0 (impurities), based on the concentration
    num_impurities = int((concentration_percent / 100.0) * total_sites)

    # Generate a flat list of all indices (as if the lattice were a 1D array)
    indices = np.arange(total_sites)

    # Randomly choose 'num_impurities' unique indices to become impurities
    # 'replace=False' ensures no duplicate indices are chosen, so each impurity affects a different site
    impurity_indices = np.random.choice(indices, size=num_impurities, replace=False)

    # Flatten the 2D lattice into a 1D array so we can index it easily
    flat = new_lattice.flatten()

    # Set the chosen indices to 0 to simulate impurities
    flat[impurity_indices] = 0

    # Reshape the 1D array back to its original 2D lattice shape
    return flat.reshape(new_lattice.shape)

def calculate_energy(lattice, J=1, H=0):
    """
    Vectorized energy calculation for a given Ising spin lattice configuration.
    Energy includes the nearest neighbour interaction via J (ferro/antiferromagnetism) with periodic boundary conditions and external magnetic field, H, summed over all spins.
    Nerest neighbours are implemented by including just the up and left neighbour per spin position. Iterating over all spins results in all NN contributions (so including down and  right) only once.
   

    Parameters: 
    -----------
    lattice: np.ndarray
        Ising spin lattice (2D).
    J: float (optional)
        Nearest neighbours coupling constant, default 1.
    H: float (optional)
        External magnetic field strength, default 0.
    
    Returns:
    ----------
    E_interaction + E_field: float
        Total energy of the spin lattice.
    """
    # Periodic nearest neighbors (roll shifts the array)
    interaction = (
        np.roll(lattice, 1, axis=0) +  # up neighbor
        np.roll(lattice, 1, axis=1)    # left neighbor
    )

    E_interaction = -J * np.sum(lattice * interaction)
    E_field = -H * np.sum(lattice)

    return E_interaction + E_field

def acceptance(delta_E, T):
    """
    Calculates the acceptance probability for a spin flip based on change in energy.

    Parameters: 
    -----------
    delta_E: float
        Change in energy due to spin flip (non-dimensionalised).
    T: float
        Temperature (non-dimensionalised).

    Returns:
    ----------
    ratio: float
        Probability of accepting the flip.
    """
    ratio = min(1, np.exp(-delta_E / T))

    return ratio

def metropolis_step(lattice, T, J=1, H=0):
    """
    Performs one Metropolis Monte Carlo sweep over the entire lattice.
    Selects from N*N spins randomly and flips their spins if they result in lower energy otherwise the spin flip happens with Boltzmann prob=e^(-deltaE), where deltaE is the energy penalty.

    Parameters: 
    -----------
    lattice: np.ndarray
        Ising spin lattice (2D).
    T: float
        Temperature (non-dimensionalised).
    J: float (optional)
        Nearest neighbours coupling constant, default 1.
    H: float (optional)
        External magnetic field strength, default 0.
    

    Returns:
    ----------
    lattice: np.ndarray
        Updated lattice configuration (2D).
    acceptance ratio: float
        Number of accepted flips per number of all MC steps in 1 sweep.
    """

    #lattice size
    n = lattice.shape[0]
    accepted = 0

    #sampling outside the loop to increase code efficiency-> one big sampling call for all random spin flips 
    sample=np.random.randint(low=0, high=n, size=2*n*n, dtype=int)

    for k in range(n*n):
        # Choose a random spin
        i = sample[2*k]
        j = sample[2*k+1]
        s = lattice[i,j]
        
        #NN
        neighbors = lattice[(i+1)%n, j] + lattice[(i-1)%n, j] + \
                    lattice[i, (j+1)%n] + lattice[i, (j-1)%n]

        # ΔE = E_after-E_before, where E_before=E_local(s) and E_after=E_local(-s)
        #Energy change if we flip this spin 
        delta_E = 2*s * (J * neighbors +H)
    
        # Accept flip with Metropolis probability
        if delta_E <= 0 or np.random.rand() < acceptance(delta_E, T):
            lattice[i, j] *= -1  # Flip the spin
            accepted += 1
            
    acceptance_ratio = accepted / (n*n) 
    
    return lattice, acceptance_ratio

def wolff_step(lattice, T, J=1,):
    """
    Note that this version hardcodes H=0, when compared to Metropolis. 
    
    Performs one Wolff step:
    Randomly selects one spin to define a single cluster. 
    Grows this cluster from this initial point by only considering nearest neighbours of parallel spin. 
    Neighbours are added with probability p=1-e^(-2J/kbT). 
    At the end of the Wolff step the full cluster flips its spin.
       
    Parameters: 
    -----------
    lattice: np.ndarray
        Ising spin lattice.
    T: float
        Temperature (non-dimensionalised).
    J: float (optional)
        Nearest neighbours coupling constant, default 1.
   
    Returns:
    ----------
    lattice: np.ndarray
        Updated lattice configuration.
    """
    #count 
    count=1
    
    n = lattice.shape[0]
    #keeps track of spins already added to the cluster
    visited=np.zeros((n,n))
    #creates queue to keep track of spins to be considered 
    q = Queue(maxsize = n*n)

    #Cluster initialisation
    
    #select random spin to start with
    cluster_start = list(np.random.randint(0, n, size=2))
    #define cluster spin
    cluster_spin=lattice[cluster_start[0],cluster_start[1]]
    #put into the queue so it can be used to grow the cluster
    q.put(cluster_start)
    #mark starting point as visited
    visited[cluster_start[0],cluster_start[1]]=1
    #flip starting point spin with 100% probability
    lattice[cluster_start[0],cluster_start[1]]=-lattice[cluster_start[0],cluster_start[1]]
    
    while q.qsize()>0:
        # get current spin (q.get() deletes from the queue, so you can only use it once)
        i,j=q.get()
    
        #get nearest neighbours
        neighbors_idx = [[(i+1)%n, j], [(i-1)%n, j], [i, (j+1)%n], [i, (j-1)%n]]
        #get not yet visited nearest neighbours
        to_add_idx=[ idx for idx in neighbors_idx if visited[idx[0],idx[1]]!=1]
        
        #only look at neighbours not yet in the cluster
        for element in to_add_idx:
            #add to cluster with p=1-e^-2J if the current spin=cluster spin
            if lattice[element[0], element[1]]==cluster_spin:
                if np.random.rand()<(1-np.exp(-2*J/T)):
                    #only mark as visited after adding it to the cluster, (we must allow all 4 nn to try to add a this spin to the cluster in order not to bias the distribution)
                    visited[element[0],element[1]]=1
                    count+=1
                    #flip spin=add to cluster
                    lattice[element[0], element[1]]=-lattice[element[0], element[1]]
                    #add to the queue so it can be used to grow the cluster further (via its nearest neighbours)
                    q.put(element)
   
    return lattice, count


def bin_error(x, bin_size):
    """
    Estimates the error of (the mean of) x using the data blocking method.

    Parameters:
    -----------
    x: np.ndarray
        Array of observable to be used for computation.
    bin_size: int
        Size of bins to be used in the data blocking.

    Returns:
    ---------
    error: np.ndarray
        Calculated associated error to each value from x.
    """
    x = np.array(x)
    N = len(x)
    n_bins = N // bin_size # integer number of bins
    # ensure at least 2 bins 
    if n_bins < 2:
        raise ValueError("Bin size too large or too few data points")

    # Reshape into bins and compute mean of each bin
    binned_data = x[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1) # only takes full bins (leftovers ignored)
    error = np.std(binned_data, ddof=1) / np.sqrt(n_bins)
    
    return error

def bin_variance_error(x, bin_size):
    """
    Estimates the error of the variance of x using the data blocking method.

    Parameters:
    -----------
    x: np.ndarray
        Array of observable to be used for computation.
    bin_size: int
        Size of bins to be used in the data blocking.

    Returns:
    ---------
    error: np.ndarray
        Calculated associated variance error to each value from x.
    """
    x = np.array(x)
    N = len(x)
    n_bins = N // bin_size # integer number of bins
    # ensure at least 2 bins 
    if n_bins < 2:
        raise ValueError("Bin size too large or too few data points")

    # Reshape and compute variance in each bin
    binned_data = x[:n_bins * bin_size].reshape(n_bins, bin_size)
    bin_vars = np.var(binned_data, axis=1)
    error = np.std(bin_vars, ddof=1) / np.sqrt(n_bins)
  
    return error

def detect_thermalization(observable, window=100, tol=1e-2):
    """
    Computes the average per rolling window and determines if the change in value between suscessive windows is below the tolerance to determined when thermalization occurs.

    Parameters:
    -----------
    observable: np.ndarray
        Array of observable to be used for computation
    window: int 
        Size of rolling window (default: 100)
    tol: float
        Tolerance on the difference between averages of succesive windows (default: 0.01)

    Returns:
    --------
    Boolean of if thermalisation has occured, index (or None) of when thermalisation occurs (or doesnt occur)
    """
    obs = np.array(observable)
    for i in range(len(obs) - 2 * window): # to ensure windows dont go out of bounds
        avg1 = np.mean(obs[i:i+window]) 
        avg2 = np.mean(obs[i+window:i+2*window]) # the next window from avg1
        if np.abs(avg2 - avg1) < tol:
            return True, i + 2 * window  # Returns index after second window
            
    return False, None

def temp_sweep_dynamic_thermalization_w_binerrors(n, T_range, n_steps, window_size, bin_size, J=1, H=0, alg='W'):
    """
    Runs the Ising model across a range of temperatures and collects observables. Thermalisation steps are determined using a sliding window.
    Error bars are calculated using data blocking.

    Parameters:
    ------------
    n: int
        Size of Ising lattice (n x n).
    T_range: np.ndarray
        Range of non-dimensional temperatures over which to perform simulations.
    n_steps:
        Number of steps of Metropolis algorithm.
    window_size: int
        Size of window for the sliding window used for determining thermalization steps.
    bin_size: int
        Size of bins to be used in the data blocking.
    J: float (optional)
        Nearest neighbours coupling constant, default 1.
    H: float (optional)
        External magnetic field strength, default 0.
    alg: str (optional, default=W)
        Determines which update algorithm to use, if set to W uses Wolff, uses Metropolis otherwise.

    Returns:
    ------------
    Observables: 
        T: Range of temperatures used
        E: Average energies
        M: Average magnetisations 
        C: Specific heats
        Chi: Susceptibility
        E_rr, M_err, C_err, Chi_err: Errors associated to E, M, C, and Chi respectively.
    """
    E_avg = []
    M_avg = []
    C = []
    chi = []

    E_errors = []
    M_errors = []
    C_errors = []
    chi_errors = []

    for T in T_range:
        lattice = initialize_lattice(n)

        energies = []
        magnetizations = []
        max_therm = 5000 # an arbitrary max steps to stop the loop 
        
        for step in range(max_therm): 
            if alg=='W':
                wolff_step(lattice, T, J=1,)
            else:
                metropolis_step(lattice, T, J, H)[0]

            E = calculate_energy(lattice, J, H)
            M = np.sum(lattice)

            energies.append(E)
            magnetizations.append(M)
            
        is_thermalized, thermalized_step = detect_thermalization(magnetizations, window_size)

        if is_thermalized:
            leftover = max_therm - thermalized_step
            if leftover < n_steps: # ensures enough for n_steps
                for further_steps in range(n_steps - leftover):
                    metropolis_step(lattice, T, J, H)[0]

                    E = calculate_energy(lattice, J, H)
                    M = np.sum(lattice)

                    energies.append(E)
                    magnetizations.append(M)
        else: # if never thermalized just take max_therm as thermalisation steps
            for further_steps in range(n_steps):
                metropolis_step(lattice, T, J, H)[0]

                E = calculate_energy(lattice, J, H)
                M = np.sum(lattice)

                energies.append(E)
                magnetizations.append(M)
    
        N = n * n
        E_arr = np.array(energies[-n_steps:])
        M_arr = np.array(magnetizations[-n_steps:])

        e_mean = np.mean(E_arr) / N
        m_mean = np.mean(np.abs(M_arr)) / N
        c_heat = (np.var(E_arr) / (T**2)) / N
        susceptibility = ((np.mean(M_arr**2) - np.mean(M_arr)**2) / T) / N

        # calculate error bars
        E_err = bin_error(E_arr, bin_size) / N
        M_err = bin_error(M_arr, bin_size) / N
        C_err = bin_variance_error(E_arr, bin_size) / (T**2 * N)
        chi_err = bin_variance_error(M_arr, bin_size) / (T * N)

        # store for plotting
        E_avg.append(e_mean)
        M_avg.append(m_mean)
        C.append(c_heat)
        chi.append(susceptibility)

        E_errors.append(E_err)
        M_errors.append(M_err)
        C_errors.append(C_err)
        chi_errors.append(chi_err)

    return {
        'T': T_range,
        'E': E_avg,
        'M': M_avg,
        'C': C,
        'Chi': chi,
        'E_error': E_errors,
        'M_error': M_errors,
        'C_error': C_errors,
        'Chi_error': chi_errors
    }

