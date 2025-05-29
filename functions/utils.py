import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Define the debug function
def debug_print(debug, *args):
    if debug:
        print(*args)

def flatten_list(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten_list(item)  # Recursively flatten lists
        else:
            yield item

def check_solution_error_numpy(A, s, b):
    """
    Calculate the residual error ||b - As|| using numpy
    """
    print(f'A shape: {A.shape}\nb shape: {b.shape}\ns shape: {s.shape}') 
    residual = np.linalg.norm(b.reshape(-1, 1) - A.dot(s.reshape(-1, 1))) 
    print(f'residual for 1 numpy: {residual}')

def check_solution_error_torch(A, s, b):
    """
    Calculate the residual error ||b - As|| after linear system is converted to graph
    """
    s = torch.tensor(s.reshape(-1,1)).to(device)
    b = torch.tensor(b.reshape(-1,1)).to(device)
    residual_ = torch.linalg.norm(b - torch.sparse.mm(A, s))
    print('Residual error 2 torch: ',residual_)
    del A, residual_

from scipy import sparse
from scipy.sparse import linalg

def is_sparse_spd(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if the matrix is symmetric
    if not (matrix != matrix.T).nnz == 0:
        return False

    try:
        # Attempt Cholesky decomposition
        linalg.cholesky(matrix)
    except:
        # If Cholesky decomposition fails, the matrix is not positive definite
        return False

    # If all checks pass, the matrix is symmetric positive definite
    return True

from scipy import sparse
import scipy.sparse.linalg as spla

def is_positive_semidefinite_sparse(matrix):
    # Check if the matrix is symmetric
    if not (matrix != matrix.T).nnz == 0:
        return False
    
    try:
        # Compute the smallest eigenvalues
        eigenvalues = spla.eigsh(matrix, k=1, which='SM', return_eigenvectors=False)
        
        # Check if the smallest eigenvalue is non-negative (allowing for numerical precision)
        return eigenvalues[0] > -1e-10
    except spla.ArpackNoConvergence:
        # If eigenvalue computation fails, we can't determine if it's positive semidefinite
        return False

def plot_one_variable(variable_list, xlabel, ylabel, label, name):

    fontsize=25
    plt.figure(figsize=(8, 6))
    plt.gca().set_facecolor('#F5F5F5') 
    plt.plot(variable_list, label=label)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(color='#D3D3D3', linestyle='-', linewidth=0.5)  # Light gray fine grid
    plt.grid(True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.yscale('log')
    plt.savefig(name, bbox_inches="tight", dpi=300)
    plt.close()
    # plt.show()

def plot_results(all_residuals_dict, save_path, episode, N, default_omega, omega_list=None, time_list=None, plot_cpu_time=False):
    """
    Plot and save the residual curves for different methods.

    Parameters:
        all_residuals_dict (dict): Dictionary containing residuals data.
        save_path (str): Directory where the plot will be saved.
        episode (int): Current episode number (used in labels).
        N (int): Key/index used to access residual data.
        default_omega (float): Parameter for the default SOR method.
        time_list (optional): Not used in this implementation.
        plot_cpu_time (bool, optional): Not used in this implementation.
    """

    file_name = f'{save_path}/residual_plot_{episode}_N{N}.png'
    fontsize = 20

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#F5F5F5')

    # Mapping from dictionary keys to labels
    data_labels = { 
        'residuals_RL': f'RL SOR ep={episode}',
        'residuals_SOR': f'default SOR w={default_omega}',
        'residuals_baseline': f'Baseline'
    }

    # Plot each dataset if available
    for key, label in data_labels.items():
        data = all_residuals_dict.get(key, {}).get(N)
        if data:
            ax.plot(data, label=label)

    # Set labels, grid, and tick parameters

    ax.set_xlabel('Iterations', fontsize=fontsize)
    ax.set_ylabel('Residual Norm', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.grid(color='#D3D3D3', linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_yscale('log')

    # Add title with size N
    ax.set_title(f'Size: N = {N}', fontsize=fontsize)

    # Optionally update x-axis ticks to display iteration and omega value
    if omega_list is not None:
    
        # Annotate each point on the RL SOR curve with its relaxation parameter every 5 iterations.
        rl_data = all_residuals_dict.get('residuals_RL', {}).get(N)
        if rl_data is not None:
            for i, omega in enumerate(omega_list):
                if i % 5 == 0 and i < len(rl_data):
                    ax.annotate(f'{omega:.2g}', xy=(i, rl_data[i]),
                                xytext=(0, 5), textcoords='offset points',
                                fontsize=8, color='blue', rotation=45)

    # Save and close the figure
    fig.savefig(file_name, bbox_inches="tight", dpi=300)
    # plt.close(fig)


def plot_results_no_RL(all_residuals_dict, save_path, data_size_list, default_omega):
    """
    Plot and save the residual curves for different methods.

    Parameters:
        all_residuals_dict (dict): Dictionary containing residuals data.
        save_path (str): Directory where the plot will be saved.
        episode (int): Current episode number (used in labels).
        N (int): Key/index used to access residual data.
        default_omega (float): Parameter for the default SOR method.
        time_list (optional): Not used in this implementation.
        plot_cpu_time (bool, optional): Not used in this implementation.
    """

    for N in data_size_list:
        file_name = f'{save_path}/residual_plot_N{N}.png'
        fontsize = 20

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor('#F5F5F5')


        # Mapping from dictionary keys to labels
        data_labels = {
            'residuals_SOR': f'default SOR w={default_omega}',
            'residuals_baseline': f'Baseline'
        }

        # Plot each dataset if available
        for key, label in data_labels.items():
            data = all_residuals_dict.get(key, {}).get(N)
            if data:
                ax.plot(data, label=label)

        # Set labels, grid, and tick parameters

        ax.set_xlabel('Iterations', fontsize=fontsize)
        ax.set_ylabel('Residual Norm', fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.grid(color='#D3D3D3', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_yscale('log')

        # Add title with size N
        ax.set_title(f'Size: N = {N}', fontsize=fontsize)

        # Save and close the figure
        fig.savefig(file_name, bbox_inches="tight", dpi=300)
        # plt.close(fig)

def plot_durations(episode_durations, path, is_ipython, show_result=True):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig(f"{path}/training.png")
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    plt.ioff()
    plt.close()

def load_policy_net(policy_net, filepath):
    policy_net.load_state_dict(torch.load(filepath))
    policy_net.eval()  # Set the network to evaluation mode
    print("Policy network weights loaded successfully.")
    return policy_net

def parse_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore empty lines or lines that start with a comment
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line into key and value
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Convert the value to the appropriate type
            if value.isdigit():
                value = int(value)  # Convert to integer
            elif value.replace('.', '', 1).isdigit() and '.' in value:
                value = float(value)  # Convert to float
            elif value.lower() == 'true':
                value = True  # Convert to boolean True
            elif value.lower() == 'false':
                value = False  # Convert to boolean False
            elif value.startswith('[') and value.endswith(']'):
                # Convert to a list (assumes list of integers)
                value = eval(value)  # Use eval cautiously; safer alternatives exist
            elif value.lower() == 'none':
                value = None  # Convert to None
            else:
                value = str(value)  # Keep as string

            config[key] = value

    return config