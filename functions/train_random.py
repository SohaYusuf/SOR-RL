# python main.py --train_RL_model 1 --run_FGMRES_baseline 1 --run_FGMRES_default_SOR 1 --num_episodes 100 --data_size_list 576 --target_tol 1e-8 --n_actions 5 --learning_rate 1e-7

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from sklearn.metrics import mean_squared_error
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from functions.read_data import *
from functions.preconditioners import *
from functions.env import *
from functions.model import *
from functions.utils import *
from functions.paths import *
from functions.fgmres import *


# Set all seeds for reproducibility
# seed = 42  # You can choose any fixed seed value

# Python's built-in random module
# random.seed(seed)

# # PyTorch (both CPU and CUDA)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # NumPy (if used in any imported functions)
# import numpy as np
# np.random.seed(seed)


def train_(config):

    # ================================================================================================================================
    #                                                   Matplotlib Setup
    # ================================================================================================================================
    # Enable interactive plotting mode and configure for IPython
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()

    print('\n Running main ..... \n')

    # ================================================================================================================================
    #                                                   Configuration and Argument Extraction
    # ================================================================================================================================
    # Extract relevant command line arguments from the configuration
    args = {k: config[k] for k in ["seed", "batch_size", "gamma", "eps_start", "eps_end", "eps_decay",
                               "tau", "learning_rate", "num_episodes", "target_tol", "default_omega",
                               "debug", "data_path", "save_path", "dataset", "checkpoint", "mode",
                               "data_size_list", "n_actions", "train_RL_model", "run_FGMRES_baseline", 
                               "run_FGMRES_default_SOR"]}



    # ================================================================================================================================
    #                                                   Result Path Setup
    # ================================================================================================================================
    # Set up the directory to save results
    save_path = config["save_path"]
    os.makedirs(save_path, exist_ok=True)
    seed = config["seed"]
    ddtype = np.float32

    # ================================================================================================================================
    #                                                   Save config file
    # ================================================================================================================================
    # Convert the dictionary into a list of key-value pairs, converting values to strings
    config_list = [[key, str(value)] for key, value in args.items()]

    # Convert the list to a NumPy array
    config_array = np.array(config_list)

    # Save the configuration array to a text file using np.savetxt
    # fmt='%s' ensures all entries are treated as strings,
    # and the delimiter separates the key and value with ": "
    np.savetxt(f"{save_path}/config.txt", config_array, fmt="%s", delimiter=": ")

    # ================================================================================================================================
    #                                                   Load the data
    # ================================================================================================================================

    train_loader = get_train_loader(train_path=config["data_path"], sizes=config["data_size_list"], batch_size=config["batch_size"], mode=config["mode"], device=device)

    # ================================================================================================================================
    #                                                   Set up the reinforcement learning hyparparameters
    # ================================================================================================================================

    Transition = namedtuple('Transition',
                    ('state', 'action', 'next_state', 'reward'))


    class ReplayMemory(object):

        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.memory.append(Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    BATCH_SIZE = config["batch_size"]
    GAMMA = config["gamma"]
    EPS_START = config["eps_start"]
    EPS_END = config["eps_end"]
    EPS_DECAY = config["eps_decay"]
    TAU = config["tau"]
    LR = config["learning_rate"]
    num_episodes = config["num_episodes"]

    # Get number of actions from gym action space
    # n_actions = env.action_space.n
    # Get the number of state observations
    # state, info = env.reset()
    # n_observations = len(state)
    # n_observations = 1

    n_actions = config['n_actions']
    n_observations = 2

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    # ================================================================================================================================
    #                                                   Initialize the residual dictionaries
    # ================================================================================================================================

    residuals_baseline = {}
    residuals_SOR = {}

    all_residuals_dict = {}

    for A, A_tensor, x_true, b in train_loader:
        
        N = A.shape[0]
        print(f' ------------------ N : {A.shape[0]} --------------------')

        # ================================================================================================================================
        #                                                   GMRES Baseline Solution
        # ================================================================================================================================
        if config["run_FGMRES_baseline"]==True:
            # Solve using Flexible GMRES without preconditioner
            gmres_baseline = FlexibleGMRES_original(A, max_iter=A.shape[0], tol=config["target_tol"])
            x_baseline, residuals_baseline[N] = gmres_baseline.solve(b)

            # Calculate baseline solution error
            solution_error_baseline = np.sqrt(mean_squared_error(x_true, x_baseline))
            print(f'solution_error baseline: {solution_error_baseline}')
            print(f'residuals baseline: {residuals_baseline[N][-1]}')

            # plot_one_variable(residuals_baseline[N], xlabel='iteration', ylabel='residual norm', label='no preconditioner', name='baseline_FGMRES')

        # ================================================================================================================================
        #                                                   GMRES with SOR Preconditioner
        # ================================================================================================================================
        if config["run_FGMRES_default_SOR"]==True:
            # Construct SOR preconditioner
            M_sor_ = M_sor(A, omega=config["default_omega"])

            # Solve using Flexible GMRES with SOR preconditioner
            gmres_SOR = FlexibleGMRES_original(A, max_iter=A.shape[0], tol=config["target_tol"], M=M_sor_,)
            x_SOR, residuals_SOR[N] = gmres_SOR.solve(b)

            # Calculate solution error with SOR
            solution_error_SOR = np.sqrt(mean_squared_error(x_true, x_SOR))
            print(f'solution_error with SOR: {solution_error_SOR}')
            print(f'residuals using SOR: {residuals_SOR[N][-1]}')

        
    all_residuals_dict['residuals_baseline'] = residuals_baseline
    all_residuals_dict['residuals_SOR'] = residuals_SOR

    # ================================================================================================================================
    #                                Plot and save the residuals for Baseline and SOR for all sizes
    # ================================================================================================================================

    plot_results_no_RL(all_residuals_dict=all_residuals_dict, 
                 save_path='results/', 
                 data_size_list=config['data_size_list'], 
                 default_omega=config['default_omega'])

    # Save residuals dictionary to a text file
    with open(f"{save_path}/residuals_baseline.txt", "w") as f:
        json.dump(residuals_baseline, f, indent=4)

    with open(f"{save_path}/residuals_SOR.txt", "w") as f:
        json.dump(residuals_SOR, f, indent=4)

    
    # ================================================================================================================================
    #                                                   GMRES with RL Preconditioner
    # ================================================================================================================================
    if config["train_RL_model"]==True:

        episode_durations = []

        actions_list = []
        rewards_list = []
        counter_ = 0
        episode_rewards = []

        checkpoint_folder = f'{save_path}/policy_checkpoints/'
        os.makedirs(checkpoint_folder, exist_ok=True)

        train_results_folder = f'{save_path}/train_results/'
        os.makedirs(train_results_folder, exist_ok=True)

        torch.autograd.set_detect_anomaly(True)

        residuals_RL = {}

        env = IterativeSolverEnv(config['n_actions'], target_tol=config["target_tol"], seed=seed)
        env.save_info(filename='env_info.txt', save_path=save_path)

        total_episodes = num_episodes*len(train_loader)
        total_episode_counter = 0

        for A, A_tensor, x_true, b in train_loader:

            A = csc_matrix(A)

            N = A.shape[0]
            
            # env = IterativeSolverEnv(A, b, config['n_actions'])

            # env.save_info(filename='env_info.txt', save_path=save_path)

            solver = FlexibleGMRES_RL(A, max_iter=N, tol=config["target_tol"])

            for i_episode in range(num_episodes):

                print(f'---------------------------- Episode {i_episode} -------------------------------------------')
                
                total_rewards = 0
                episode_reward = 0

                # Initialize the environment and get its state
                state, info = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                # Initialize Flexible GMRES
                solver.initialize(b)

                if i_episode == 0 or i_episode == num_episodes // 3 or i_episode == (2 * num_episodes) // 3 or i_episode == num_episodes - 1:
                    torch.save(policy_net.state_dict(), f'{checkpoint_folder}/policy_net_weights_{i_episode}.pth')

                for t in count():

                    counter_ = counter_ + 1
                    action = select_action(state)
                    
                    observation, omega_list, reward, done, residual_list, time_list = env.step(action.item(), A, solver)

                    # print('omega_list: ',omega_list)
                
                    # Print type and value of reward
                    reward = torch.tensor([reward], device=device)
                    
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                    # if done:
                    #     next_state = None
                    # else:
                    #     next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                    # Store the transition in memory
                    memory.push(state, action, next_state, reward)

                    # Move to the next state
                    state = next_state
                    
                    # optimize_model()
                    optimize_model(Transition, memory, policy_net, target_net, optimizer, device, BATCH_SIZE, GAMMA)

                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()

                    # Save the policy_net weights
                    torch.save(policy_net.state_dict(), 'policy_net_weights.pth')
                    
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = (policy_net_state_dict[key] * TAU +
                                                    target_net_state_dict[key] * (1 - TAU))
                    
                    target_net.load_state_dict(target_net_state_dict)
                    total_rewards = total_rewards + reward.item()

                    actions_list.append(action.item())
                    rewards_list.append(reward.item())

                    episode_reward+=reward.item()
                    
                    if done:
                        episode_rewards.append(episode_reward)
                        episode_durations.append(t + 1)
                        plot_durations(episode_durations, train_results_folder, is_ipython)
                        break

                residuals_RL[N] = residual_list

                all_residuals_dict['residuals_RL'] = residuals_RL

                if i_episode % 5 == 0:
                    np.savetxt(f'{train_results_folder}/residuals_{i_episode}.txt', residual_list)
                    print('residual_list',residual_list)

                    plot_results(all_residuals_dict=all_residuals_dict, 
                                 save_path=f'{train_results_folder}', 
                                 episode=total_episode_counter, 
                                 N=N, 
                                 default_omega=config["default_omega"], 
                                 omega_list=omega_list)
                
                total_episode_counter +=1
                print(f'episode {i_episode}, action {action.item()}  ======> state: {observation} ======> reward: {episode_reward}')

            print('Complete')
            plot_durations(episode_durations,train_results_folder, is_ipython, show_result=False)
            plt.ioff()
            plt.show()

        with open(f"{save_path}/residuals_RL.txt", "w") as f:
            json.dump(residuals_RL, f, indent=4)
    
    # ================================================================================================================================
    #               Save dictionary containing all the residuals for Baslines, Default SOR and RL SOR
    # ================================================================================================================================
    with open(f"{save_path}/all_residuals.txt", "w") as f:
        json.dump(all_residuals_dict, f, indent=4)
