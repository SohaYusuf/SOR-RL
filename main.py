import torch
import argparse
import numpy as np
import random
import time
import pprint
from functions.train_random import *
from functions.eval import *


def argparser():
    """
    Parses command-line arguments for training parameters.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Training parameters for the model.")
    
    # Define training and configuration parameters 21 arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of transitions sampled from the replay buffer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--eps_start", type=float, default=0.9, help="Starting value of epsilon for exploration")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final value of epsilon for exploration")
    parser.add_argument("--eps_decay", type=float, default=1000, help="Rate of epsilon decay (higher means slower decay)")
    parser.add_argument("--tau", type=float, default=0.005, help="Update rate of the target network")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for the AdamW optimizer")
    parser.add_argument("--num_episodes", type=int, default=100, help="Total number of episodes for training")
    parser.add_argument("--target_tol", type=float, default=1e-8, help="target tolerance for flexible GMRES")
    parser.add_argument("--default_omega", type=float, default=0.01, help="default value of relaxation parameter for default_SOR e.g 1e-2")

    parser.add_argument("--debug", type=int, default=True, help="Debug flag")

    parser.add_argument("--data_path", type=str, default="data/advection_2D/train/", help="Path to the dataset")
    parser.add_argument("--save_path", type=str, default="results/", help="path where results will be saved")

    # parser.add_argument("--N", type=int, default=144, help="Parameter N")
    parser.add_argument("--dataset", type=str, default="advection_2D", help="Dataset name e.g advection_2D")
    parser.add_argument("--checkpoint", type=str, help="checkpoint path that you want to use for evaluation")
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--data_size_list", type=int, nargs="+", default=[144, 576, 2304], help="Use 144 576 etc as the argument, do not add comma") 
    parser.add_argument("--n_actions", type=int, default=5, help="number of discrete actions the RL agent can take")  

    parser.add_argument("--train_RL_model", type=int, default=False, help="Train RL agent with FGMRES")
    parser.add_argument("--run_FGMRES_baseline", type=int, default=False, help="Run FGMRES with no preconditioner")
    parser.add_argument("--run_FGMRES_default_SOR", type=int, default=False, help="Run FGMRES with default SOR")

    return parser.parse_args()


if __name__ == "__main__":

    # Parse arguments
    args = argparser()
    
    # Logging
    print(f"Using device: {device}")
    print("Using config:")
    pprint.pprint(vars(args))
    print()

    if args.mode == "train":
        # train
        start_time = time.time()

        train_(vars(args))

        elapsed_time = time.time() - start_time
        print(f"Total training time: {elapsed_time:.2f} seconds")

    elif args.mode == "test":
        # evaluate
        start_time = time.time()

        eval_(vars(args))

        elapsed_time = time.time() - start_time
        print(f"Total evaluation time: {elapsed_time:.2f} seconds")