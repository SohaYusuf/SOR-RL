import gym
import numpy as np
import time
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import LinearOperator, spilu, spsolve, lsqr
from scipy.stats import linregress
import random
from sklearn.linear_model import LinearRegression
from gym.utils import seeding 

from functions.preconditioners import *
from functions.fgmres import *

# from pyamg.krylov import fgmres

class IterativeSolverEnv(gym.Env):
    
    def __init__(self, n_actions, target_tol, seed=None):
        super(IterativeSolverEnv, self).__init__()

        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([np.inf, 1.0], dtype=np.float32)
       
        self.action_space = gym.spaces.Discrete(n_actions)  
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)             
       
        self.target_tol = target_tol

    def reset(self, seed=None):
        # Ensure to pass the seed to the superclass
        super().reset(seed=seed) 
        self.state = np.array([1.0, 1.0], dtype=np.float32)
        self.M = None
        self.residuals_list = []
        self.current_parameter = 1.0 
        self.omega_list = []
        return np.array(self.state, dtype=np.float32), {}  

    def step_convergence_ratio(self):
        if len(self.residuals_list) == 1:
            return 1e-10  # No convergence rate can be computed with only one residual

        r_1 = self.residuals_list[-1]
        r_2 = self.residuals_list[-2]

        print('r_1: ',r_1)
        print('r_2: ',r_2)

        r = np.absolute(r_1 / r_2)  # Relative residual
        # Convergence rate is the negative of the slope
        convergence_rate = -np.log(r)

        # if convergence_rate <= 1e-7 or convergence_rate == 0:
        #     convergence_rate = 1e-8

        return convergence_rate

    def save_info(self, filename, save_path):
        """
        Saves key environment information to a text file.
        """
        # Get number of actions if using a Discrete action space
        if hasattr(self.action_space, "n"):
            num_actions = self.action_space.n
        else:
            num_actions = "Non-discrete action space"
        
        # Call reset to obtain reset values (this will update the environment's state)
        reset_state, _ = self.reset()
        
        # Build the information string
        info = (
            f"Number of Actions: {num_actions}\n"
            f"Number of observations: {len(self.state)}\n"
            f"States: {'np.log(convergence_rate), np.log(residual_norm)'}\n"
            f"Reset State: {reset_state}\n"
        )
        
        # Write the information to the specified text file
        with open(f'{save_path}/{filename}', "w") as file:
            file.write(info)
        print(f"Environment information saved to {filename}")

    
    def step(self, action, A, solver):

        print('======================> action <=======================: ', action)
        N = A.shape[0]

        if self.action_space.n == 3:
            parameters = [0.5, 1.0, 2.0]
        if self.action_space.n == 5:
            parameters = [0.1, 0.5, 1.0, 2.0, 10.0]

        factor = parameters[action]
        self.current_parameter *= factor
        self.current_parameter = np.clip(self.current_parameter, 1e-20, 1e20)
        self.omega_list.append(self.current_parameter)

        # Solve using the selected parameter
        M_sor_ = M_sor(A, omega=self.current_parameter)
        _, residual_vector, _ , residual_norm, _ = solver.step(M_sor_)

        self.residuals_list.append(residual_norm)
        convergence_rate = self.step_convergence_ratio()
        self.state = (convergence_rate, np.log(residual_norm))
        self.state_info = str((convergence_rate, np.log(residual_norm)))

        # Reward function
        # reward = convergence_rate
        c=100
        p=2
        reward = c*(convergence_rate**p)/(1+(convergence_rate**p))
        if len(self.residuals_list) > 1:
            if residual_norm < self.residuals_list[-2]:
                reward += 10  # Bonus for reducing the residual
            else:
                reward -= 1

        if residual_norm <= self.target_tol:
            reward += 100

        # termination
        if len(self.residuals_list) > 1:
            terminated = bool(
                    len(self.residuals_list) >= N-5 or
                    convergence_rate < 1e-5)
        else:
            terminated = False

        print(f'action: {action}, relaxation parameter: {self.current_parameter}')
        print('current_residual_norm:', residual_norm)
        print('convergence_rate: ', convergence_rate)
        print('reward: ', reward)

        return np.array(self.state, dtype=np.float32), self.omega_list, reward, terminated, self.residuals_list, None
