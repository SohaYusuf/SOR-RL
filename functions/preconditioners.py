import numpy as np
import torch

from scipy.sparse import csr_matrix, diags, eye, tril, triu
from scipy.sparse import linalg as sla
from scipy.sparse.linalg import LinearOperator, spilu, spsolve, lsqr

from scipy.sparse import csr_matrix, tril, triu, diags
from scipy.sparse.linalg import LinearOperator

def M_sor(A, omega=1.0, epsilon=1e-20):
    """
    Create an SOR preconditioner for the given matrix A, including non-symmetric matrices.
    
    Parameters:
    A : scipy.sparse.csr_matrix
        The matrix to precondition (can be non-symmetric)
    omega : float, optional
        The relaxation factor (default is 1.0, which gives Gauss-Seidel)
    
    Returns:
    M : scipy.sparse.linalg.LinearOperator
        The SOR preconditioner as a LinearOperator
    """
    # Ensure A is in CSR format
    A = csr_matrix(A)
    
    # Extract diagonal and lower triangular parts
    D = diags(A.diagonal())
    L = tril(A, k=-1)
    
    # Compute (D/omega - L)
    D_omega = D 
    L = omega * L
    DL = D_omega - L
    DL += epsilon * np.eye(DL.shape[0])  # Add small value to diagonal
    
    def M_solve(x):
        """Solve the system (D/omega - L)y = x"""
        return lsqr(DL, x)[0]
    
    def M_apply(x):
        """Apply the SOR preconditioner"""
        y = M_solve(x)
        return omega * y
    
    # Create and return the LinearOperator
    n = A.shape[0]
    M = LinearOperator((n, n), matvec=M_apply, dtype=A.dtype)
    
    return M

def M_spilu_scipy(A_matrix, fill_factor=None, drop_tol=None):
    """
    Constructs an ILU preconditioner using SciPy's spilu.
    
    Parameters:
    -----------
    A_matrix : scipy.sparse matrix
        Sparse matrix for ILU factorization.
    fill_factor : float, optional
        Fill-in level for ILU. Higher value means more memory usage.
    drop_tol : float, optional
        Drop tolerance for ILU. Lower value keeps more entries.
        
    Returns:
    --------
    Minv : scipy.sparse.linalg.LinearOperator
        Linear operator for the preconditioner.
    """
    
    # Perform ILU factorization
    ilu_preconditioner = spilu(A_matrix, fill_factor=fill_factor, drop_tol=drop_tol)

    # Define ILU solve function
    def apply_preconditioner(residual_vector):
        return ilu_preconditioner.solve(residual_vector)

    # Return as a linear operator for use in solvers
    M = sla.LinearOperator(A_matrix.shape, matvec=apply_preconditioner)

    return M