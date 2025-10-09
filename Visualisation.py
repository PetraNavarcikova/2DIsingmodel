
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Project2 import initialize_lattice,metropolis_step


def animate_lattice(N, T, n_steps, J=1, H=0, interval=200):
    """
    Animates the Ising lattice such that -1 and 1 correspond to blue and red respectively.
    
    Parameters:
    -----------
    N: int
        Size of lattice (N x N).
    T: int
        Non-dimensional temperature.
    n_steps: int
        Number of repetitions of metropolis step.
    interval: int
        Time between each frame of the animation in milliseconds.

    Returns: 
    -----------
    Animation of ising lattice over n_steps of metropolis algorithm.
    """
    lattice = initialize_lattice(N)

    # form animation plot 
    fig, ax = plt.subplots()
    im = ax.imshow(lattice, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1) # animation settings (colour etc.)
    ax.set_title(f"T = {T}")
    ax.axis('off')

    # perform metropolis step as a frame of the animation
    def update(frame):
        nonlocal lattice
        lattice = metropolis_step(lattice, T, J, H)[0]
        im.set_array(lattice)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=True)
    plt.close() # to stop last frame from showing
    
    return ani


def plot_magnetization(data):
    """
    Plots average magnetization as a function of temperature.

    Parameters:
    ------------
    data (dict): A dictionary containing:
        'T' (list or array-like): Temperature values.
        'M' (list or array-like): Corresponding magnetization values.

     Returns:
    ----------
         None
    """
    plt.plot(data['T'], data['M'], marker='o')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Magnetization ⟨|M|⟩')
    plt.title('Magnetization vs Temperature')
    plt.grid(True)
    plt.show()

def plot_energy(data):
    """
    Plots average energy as a function of temperature.

    Parameters:
    ------------
    data (dict): A dictionary containing:
            'T' (list or array-like): Temperature values.
            'E' (list or array-like): Corresponding average energy values.

     Returns:
    ----------
         None
    """
    plt.plot(data['T'], data['E'], marker='o')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Energy ⟨E⟩')
    plt.title('Energy vs Temperature')
    plt.grid(True)
    plt.show()

def plot_specific_heat(data):
    """
    Plots specific heat per spin as a function of temperature.

    Parameters:
    ------------
    data (dict): A dictionary containing:
        'T' (list or array-like): Temperature values.
        'C' (list or array-like): Corresponding specific heat per spin.

     Returns:
    ----------
         None
    """
    plt.plot(data['T'], data['C'], marker='o')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Specific Heat C')
    plt.title('Specific Heat vs Temperature')
    plt.grid(True)
    plt.show()

def plot_susceptibility(data):
    """
    Plots specific magnetic susceptibility per spin as a function of temperature.

    Parameters:
    -----------
    data (dict): A dictionary containing:
        'T' (list or array-like): Temperature values.
        'Chi' (list or array-like): Corresponding magnetic susceptibility per spin.

     Returns:
    ----------
         None 
    """
    plt.plot(data['T'], data['Chi'], marker='o')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Magnetic Susceptibility χ')
    plt.title('Susceptibility vs Temperature')
    plt.grid(True)
    plt.show()



