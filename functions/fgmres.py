import numpy as np
from scipy.linalg import get_blas_funcs, qr_insert
from typing import Optional, Tuple, List
from scipy.linalg import solve_triangular

from functions.preconditioners import *
from functions.env import *


class FlexibleGMRES_original:
    def __init__(self, A, max_iter, tol, M=None):
        self.A = A # use A as linear operator
        self.M = M if M is not None else np.eye(A.shape[0])  # Use identity matrix if no preconditioner is given
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, b, x0=None):
        # self.A,self.M,x,b,postprocess = make_system(self.A,self.M,x0,b)
        if x0 is None:
            x0 = np.zeros_like(b)  # Initialize x0 as an array of zeros if not provided
            
        r0 = b - self.A @ x0
        beta = np.linalg.norm(r0)
        v = np.zeros((len(b), self.max_iter + 1))
        v[:, 0] = r0 / beta
        H = np.zeros((self.max_iter + 1, self.max_iter))
        Z = np.zeros((self.A.shape[0], self.max_iter))
        residuals = []

        for j in range(self.max_iter):
            # Step 3: Compute z_j
            # check convergence 
            # Step 3: Compute z_j with the correct handling of M
            if self.M is None:  # If M is None or a matrix
                #z_j = np.linalg.solve(self.M, v[:, j].reshape(-1, 1)).flatten()  # Ensure it's 2D for solve
                z_j = v[:,j] 
            elif isinstance(self.M, np.ndarray):
                z_j = np.linalg.solve(self.M, v[:, j].reshape(-1, 1)).flatten()
            elif callable(self.M):  # If M is a linear operator (function)
                z_j = self.M(v[:, j])  # Apply the linear operator directly
            else:
                raise ValueError("Preconditioner M must be either None, a matrix, or a callable linear operator.")
                
            Z[:, j] = z_j

            # Step 4: Compute w
            w = self.A @ z_j

            # Steps 5-8: Arnoldi iteration
            for i in range(0, j+1): # check if this is j
                H[i, j] = np.dot(w, v[:, i])
                w[:] -= H[i, j] * v[:, i]

            # Step 9: Compute h_{j+1,j} and v_{j+1}
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < self.tol:
                break
            v[:, j + 1] = w / H[j + 1, j]

            # Calculate and store residual
            #  print(f'H[:j + 1, :j + 1]: {H[:j + 1, :j + 1]}')
            # print(f'shape of H[:j + 1, :j]: {H[:j + 2, :j+1].shape}')
            y_approx = np.linalg.lstsq(H[:j + 2, :j+1], beta * np.eye(j + 2, 1).ravel(), rcond=None)[0]
            x_approx = x0 + Z[:, :j + 1] @ y_approx
            residuals.append(np.linalg.norm(b - self.A @ x_approx))

            # Check if the residual is below the tolerance
            if np.linalg.norm(b - self.A @ x_approx) <= self.tol:
                break

        # Final solution
        # use y_approx
        y_final = np.linalg.lstsq(H[:j + 1, :j + 1], beta * np.eye(j + 1, 1).ravel(), rcond=None)[0]
        x_m = x0 + Z[:, :j + 1] @ y_final

        return x_m, residuals

class FlexibleGMRES_RL:

    def __init__(self, A, max_iter, tol, M=None):
        self.A = A # use A as linear operator
        self.M = M if M is not None else np.eye(A.shape[0])  # Use identity matrix if no preconditioner is given
        self.max_iter = max_iter
        self.tol = tol
        self.j = 0

    def initialize(self, b, x0=None):

        self.b = b
        if x0 is None:
            x0 = np.zeros_like(self.b)  # Initialize x0 as an array of zeros if not provided

        r0 = self.b - self.A @ x0
        self.beta = np.linalg.norm(r0)
        self.v = np.zeros((len(self.b), self.max_iter + 1))
        self.v[:, 0] = r0 / self.beta
        self.H = np.zeros((self.max_iter + 1, self.max_iter))
        self.Z = np.zeros((self.A.shape[0], self.max_iter))
        self.x0 = x0
        self.j = 0
        self.residuals = []

    def step(self, M=None):

        if self.j >= self.max_iter:
            raise ValueError("Maximum iterations reached. Cannot perform another step.")

        # Step 3: Compute z_j
        if M is None:
            z_j = self.v[:, self.j]
        elif isinstance(M, np.ndarray):
            z_j = np.linalg.solve(M, self.v[:, self.j].reshape(-1, 1)).flatten()
        elif callable(M):
            z_j = M(self.v[:, self.j])
        else:
            raise ValueError("Preconditioner M must be either None, a matrix, or a callable linear operator.")

        self.Z[:, self.j] = z_j

        # Step 4: Compute w
        w = self.A @ z_j

        # Steps 5-8: Arnoldi iteration
        for i in range(0, self.j + 1):
            self.H[i, self.j] = np.dot(w, self.v[:, i])
            w[:] -= self.H[i, self.j] * self.v[:, i]

        # Step 9: Compute h_{j+1,j} and v_{j+1}
        self.H[self.j + 1, self.j] = np.linalg.norm(w)

        self.v[:, self.j + 1] = w / self.H[self.j + 1, self.j]

        # Calculate and store residual    
        y_approx = np.linalg.lstsq(self.H[:self.j + 2, :self.j+1], self.beta * np.eye(self.j + 2, 1).ravel(), rcond=None)[0]
        x_approx = self.x0 + self.Z[:, :self.j + 1] @ y_approx
        residual_vector = self.b - self.A @ x_approx
        residual_norm = np.linalg.norm(residual_vector)
        self.residuals.append(residual_norm)

        self.j += 1

        if self.H[self.j + 1, self.j] < self.tol:
            return False, residual_vector, x_approx, residual_norm, self.residuals # Convergence achieved

        return True, residual_vector, x_approx, residual_norm, self.residuals






# class FlexibleGMRES_RL:
#     def __init__(self, A, M=None, max_iter=100, tol=1e-10):
#         self.A = A  # Linear operator or matrix
#         self.M = M  # Default preconditioner, can be None, a matrix, or a callable
#         self.max_iter = max_iter
#         self.tol = tol
#         self.x0 = None
#         self.b = None
#         self.r0 = None
#         self.beta = None
#         self.v = None  # Basis vectors (columns)
#         self.H = None  # Hessenberg matrix
#         self.Z = None  # Preconditioned vectors (columns)
#         self.j = 0     # Current iteration index
#         self.residuals = []  # List to track residual norms
#         self.x_approx = None  # Current approximate solution

#     def initialize(self, b, x0=None):
#         if x0 is None:
#             x0 = np.zeros_like(b)
#         self.x0 = x0.copy()
#         self.b = b.copy()
#         self.r0 = self.b - self.A @ self.x0
#         self.beta = np.linalg.norm(self.r0)
#         n = len(b)
#         self.v = np.zeros((n, self.max_iter + 1))
#         self.v[:, 0] = self.r0 / self.beta if self.beta != 0 else 0
#         self.H = np.zeros((self.max_iter + 1, self.max_iter))
#         self.Z = np.zeros((n, self.max_iter))
#         self.residuals = []
#         self.j = 0
#         self.x_approx = self.x0.copy()

#     def step(self, M=None):
#         if self.j >= self.max_iter:
#             if len(self.residuals) == 0:
#                 return (False, None, None, None, self.residuals)
#             else:
#                 residual_vector = self.b - self.A @ self.x_approx
#                 residual_norm = np.linalg.norm(residual_vector)
#                 converged = residual_norm <= self.tol
#                 return (converged, residual_vector, self.x_approx, residual_norm, self.residuals)

#         current_j = self.j
#         current_M = M if M is not None else self.M

#         # Step 3: Compute z_j
#         v_j = self.v[:, current_j]
#         if current_M is None:
#             z_j = v_j.copy()
#         elif isinstance(current_M, np.ndarray):
#             z_j = np.linalg.solve(current_M, v_j)
#         elif callable(current_M):
#             z_j = current_M(v_j)
#         else:
#             raise ValueError("M must be None, a numpy array, or a callable function.")
#         self.Z[:, current_j] = z_j

#         # Step 4: Compute w = A @ z_j
#         w = self.A @ z_j

#         # Steps 5-8: Arnoldi process
#         for i in range(current_j + 1):
#             h_ij = np.dot(w, self.v[:, i])
#             self.H[i, current_j] = h_ij
#             w -= h_ij * self.v[:, i]

#         # Step 9: Compute h_{j+1,j} and v_{j+1}
#         h_j1j = np.linalg.norm(w)
#         self.H[current_j + 1, current_j] = h_j1j

#         if h_j1j >= self.tol:
#             self.v[:, current_j + 1] = w / h_j1j

#         # Step 12: Compute y_approx
#         H_sub = self.H[:current_j + 2, :current_j + 1]
#         e_1 = np.zeros(current_j + 2)
#         e_1[0] = self.beta
#         y_approx, _, _, _ = np.linalg.lstsq(H_sub, e_1, rcond=None)

#         # Update approximate solution
#         x_approx = self.x0 + self.Z[:, :current_j + 1] @ y_approx
#         residual_vector = self.b - self.A @ x_approx
#         residual_norm = np.linalg.norm(residual_vector)
#         self.residuals.append(residual_norm)
#         self.x_approx = x_approx.copy()

#         # Check convergence
#         converged = residual_norm <= self.tol

#         # Increment iteration index
#         self.j += 1

#         return converged, residual_vector, x_approx, residual_norm, self.residuals



