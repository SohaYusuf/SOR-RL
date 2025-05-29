# python main.py --mode test --data_path "data/advection_2D/test/" --data_size_list 144

# python main.py --mode test --train_RL_model 1 --run_FGMRES_baseline 1 --run_FGMRES_default_SOR 1 --num_episodes 100 --data_size_list 576 --target_tol 1e-8 --n_actions 5 --learning_rate 1e-7

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from sklearn.metrics import mean_squared_error
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import configparser


from functions.read_data import *
from functions.preconditioners import *
from functions.env import *
from functions.model import *
from functions.utils import *
from functions.paths import *


def eval_(config_test):

    print('\n Running main ..... \n')

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()

    # ================================================================================================================================
    #                                                   Configuration and Argument Extraction
    # ================================================================================================================================
    # Extract relevant command line arguments from the configuration
    args = {k: config_test[k] for k in ["seed", "batch_size", "gamma", "eps_start", "eps_end", "eps_decay",
                               "tau", "learning_rate", "num_episodes", "target_tol", "default_omega",
                               "debug", "data_path", "save_path", "dataset", "checkpoint", "mode",
                               "data_size_list", "n_actions", "train_RL_model", "run_FGMRES_baseline", 
                               "run_FGMRES_default_SOR"]}


    # ================================================================================================================================
    #                                                   Result Path Setup
    # ================================================================================================================================
    # Set up the directory to save results
    save_test_path = f'{config_test["save_path"]}/test/'
    os.makedirs(save_test_path, exist_ok=True)
    seed = config_test["seed"]
    ddtype = np.float32


    file_path = 'results/config.txt'  # Path to your config file
    # config = parse_config(file_path)
    # print(config)
    import ast

    def read_config(file_path):
        config = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(": ", 1)
                try:
                    # Convert value to proper type (int, float, bool, list, or None)
                    config[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # Keep as string if conversion fails
                    config[key] = value
        return config

    # Example usage
    config = read_config(file_path)
    print(config)

    test_loader = get_train_loader(train_path=config_test["data_path"], 
                                   sizes=config_test["data_size_list"], 
                                   batch_size=config_test["batch_size"], 
                                   mode=config_test["mode"], 
                                   device=device)

    n_observations=2
    n_actions=5
    # ================================================================================================================================
    #                                                  Initialize policy network
    # ================================================================================================================================
    policy_net = DQN(n_observations, n_actions).to(device)

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
    counter_example = 0

    for A, A_tensor, x_true, b in test_loader:
        counter_example +=1
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
                 save_path=save_test_path, 
                 data_size_list=config_test['data_size_list'], 
                 default_omega=config['default_omega'])

    # Save residuals dictionary to a text file
    with open(f"{save_test_path}/residuals_baseline.txt", "w") as f:
        json.dump(residuals_baseline, f, indent=4)

    with open(f"{save_test_path}/residuals_SOR.txt", "w") as f:
        json.dump(residuals_SOR, f, indent=4)

    checkpoint = "results/policy_checkpoints/policy_net_weights_99.pth"
    print('checkpoint path: ', checkpoint)

    # Assuming you have already defined and initialized your policy_net
    policy_net = load_policy_net(policy_net, checkpoint)

    env = IterativeSolverEnv(config['n_actions'], target_tol=config["target_tol"], seed=seed)

    num_episodes = 10

    BATCH_SIZE = config["batch_size"]
    GAMMA = config["gamma"]
    EPS_START = config["eps_start"]
    EPS_END = config["eps_end"]
    EPS_DECAY = config["eps_decay"]
    TAU = config["tau"]
    LR = config["learning_rate"]
    # num_episodes = config["num_episodes"]

    episode_durations = []
    actions_list = []
    rewards_list = []
    counter_ = 0
    episode_rewards = []
    residuals_RL = {}
    total_episode_counter = 0


    for A, A_tensor, x_true, b in test_loader:

        A = csc_matrix(A)

        N = A.shape[0]

        solver = FlexibleGMRES_RL(A, max_iter=N, tol=config["target_tol"])

        print(f'---------------------------- N: {N} -------------------------------------------')

        for i_episode in range(num_episodes):

            print(f'---------------------------- Episode {i_episode} -------------------------------------------')
            
            total_rewards = 0
            episode_reward = 0

            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            # Initialize Flexible GMRES
            solver.initialize(b)

            for t in count():

                # counter_ = counter_ + 1
                action = select_action(state)
                
                observation, omega_list, reward, done, residual_list, time_list = env.step(action.item(), A, solver)
                reward = torch.tensor([reward], device=device)
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                # memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state
                
                policy_net_state_dict = policy_net.state_dict()
                
                total_rewards = total_rewards + reward.item()

                actions_list.append(action.item())
                rewards_list.append(reward.item())

                episode_reward += reward.item()
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_durations.append(t + 1)
                    plot_durations(episode_durations, save_test_path, is_ipython)
                    break

            residuals_RL[N] = residual_list

            all_residuals_dict['residuals_RL'] = residuals_RL

            if i_episode % 5 == 0:
                np.savetxt(f'{save_test_path}/residuals_{i_episode}.txt', residual_list)
                print('residual_list',residual_list)

                plot_results(all_residuals_dict=all_residuals_dict, 
                                save_path=save_test_path, 
                                episode=total_episode_counter, 
                                N=N, 
                                default_omega=config["default_omega"], 
                                omega_list=omega_list)
            
            total_episode_counter +=1
            print(f'episode {i_episode}, action {action.item()}  ======> state: {observation} ======> reward: {episode_reward}')

        print('Complete')
        plot_durations(episode_durations,save_test_path, is_ipython, show_result=False)
        plt.ioff()
        plt.show()

    with open(f"{save_test_path}/residuals_RL.txt", "w") as f:
        json.dump(residuals_RL, f, indent=4)

    with open(f"{save_test_path}/all_residuals.txt", "w") as f:
        json.dump(all_residuals_dict, f, indent=4)


        
