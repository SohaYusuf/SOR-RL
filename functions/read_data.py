import torch
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


def custom_collate_fn(batch):
    """
    Since the DataLoader batch_size is always 1, this collate function simply
    returns the sole element from the batch list without further processing.
    """
    return batch[0]

class LinearSystemDataset(torch.utils.data.Dataset):
    def __init__(self, train_path, sizes, mode, device):
        self.train_path = train_path
        self.sizes = sizes
        self.device = device
        self.data = []
        self.mode = "train"

        for N in sizes:
            for p in [1]:  
                A, A_tensor, x_true, b = read_A_file(self.train_path, N, p, self.mode, self.device)
                self.data.append((A, A_tensor, x_true, b))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_train_loader(train_path, sizes, batch_size, mode, device):
    dataset = LinearSystemDataset(train_path, sizes, mode, device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    return train_loader

def read_mfem_vector(path):
    vals = []
    with open(path,'r') as f:
        for line in f:
            for tok in line.strip().split():
                vals.append(float(tok))
    return np.array(vals)



def read_A_file(train_path, N, p, mode, device):
    """
    Reads a sparse matrix from a text file and returns:
    - A: scipy sparse coo_matrix
    - A_tensor: torch sparse coo tensor
    - u_mfem: vector from u file
    - b_mfem: matrix-vector product A * u_mfem
    """

    # Define file paths
    txt_file_path = os.path.join(train_path, f"A_{N}_p{p}.txt")
    u_txt_file_path = os.path.join(train_path, f"u_ex0_{N}_p{p}.txt")
    # save_path = "results/data_visualization/"
    # os.makedirs(save_path, exist_ok=True)

    # Read sparse matrix entries from the text file
    sparse_matrix_entries_ = []
    with open(txt_file_path, 'r') as file:
        for line in file:
            row, col, val = map(float, line.split())
            sparse_matrix_entries_.append((int(row-1), int(col-1), val))   # changed this for ex0.cpp
    # Extract row indices, column indices, and values
    rows, cols, vals = zip(*sparse_matrix_entries_)

    # Construct scipy sparse matrix
    A = coo_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)
    print(f'A: \n{A}\nNo. of entries: {len(vals)}')

    ################################################ added by soha for using ex1.cpp #################################
    #################################################################################################################

    # Convert A to CSR format (efficient for matrix multiplication)
    A = A.tocsr()

    # Compute A * A^T using sparse matrix multiplication
    # A = A.dot(A.T)
    # A = A.T.dot(A)

    # Compute b_mfem
    # u_mfem = np.loadtxt(u_txt_file_path)
    u_mfem = read_mfem_vector(u_txt_file_path)
    print('u_mfem shape: ',u_mfem.shape)

    b_mfem = A.dot(u_mfem)
    print(f'b_mfem: {b_mfem}')
    print(f'Error in ||A*u_mfem - b_mfem||: {np.linalg.norm(b_mfem - A.dot(u_mfem))}')

    # Construct sparse coo tensor in torch
    vals_tensor = torch.tensor(vals, dtype=torch.float64).to(device)
    rows_tensor = torch.tensor(rows, dtype=torch.int64).to(device)
    cols_tensor = torch.tensor(cols, dtype=torch.int64).to(device)
    A_tensor = torch.sparse_coo_tensor(indices=torch.stack([rows_tensor, cols_tensor]), values=vals_tensor, size=(N, N), dtype=torch.float64).to(device)

    return A, A_tensor, u_mfem, b_mfem