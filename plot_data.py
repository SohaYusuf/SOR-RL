import torch
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
    
    
txt_file_path = 'plot_data/A_ex0_1328.txt'
#  Read sparse matrix entries from the text file
N=1328
sparse_matrix_entries_ = []
with open(txt_file_path, 'r') as file:
    for line in file:
        row, col, val = map(float, line.split())
        sparse_matrix_entries_.append((int(row), int(col), val))
# Extract row indices, column indices, and values
rows, cols, vals = zip(*sparse_matrix_entries_)

# Construct scipy sparse matrix
A = coo_matrix((vals, (rows, cols)), dtype=np.float64)

print(f'A: No. of entries: {len(vals)}')

plt.figure(figsize=(6, 6))
plt.spy(A, markersize=0.4)                         # marker style is default for sparse :contentReference[oaicite:3]{index=3}
plt.title("Sparsity Pattern of A")
plt.xlabel("Column index")
plt.ylabel("Row index")
plt.show()