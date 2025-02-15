"""
    This is is a class for the calculation of the stress matrix of a universally rigid framework.
"""

import numpy as np
import cvxpy as cp

class Stress:
    def __init__(self, nominal_config):
        """_
            nominal_config: [DIM, NUM_NODES] np.ndarray
        """
        
        self.nominal_config = nominal_config
        self.num_nodes = nominal_config.shape[1]
        self.dim = nominal_config.shape[0]
        
    def sparse_stress(self, incidence_full, hyperparam_dict, sparse_tol=5e-5):
        """
        Our proposed framework to calculate the sparse stress matrix.
        cvxpy is needed for the convex optimization.
        
        hyperparam_dict: dict
            Dictionary containing the hyperparameters for the optimization.
        """

        D, N = self.nominal_config.shape
        M = incidence_full.shape[1]
        
        augmented_config = np.hstack((self.nominal_config.T, np.ones((N, 1))))
        U, _, _ = np.linalg.svd(augmented_config)
        Q = U[:, D+1:]
        
        E = np.vstack([augmented_config.T @ incidence_full @ np.diag(incidence_full.T[:, n]) for n in range(N)])
        Psi = Q.T @ incidence_full
        
        phi = np.diag(Psi.T @ Psi)
    
        
        weights = cp.Variable(M)
        
        obj = cp.Minimize(cp.norm(weights, 1) - hyperparam_dict['alpha'] * phi.T @ weights)
        constraints = [
            cp.norm(incidence_full @ cp.diag(weights) @ incidence_full.T, 2) <= hyperparam_dict['beta'],
            Psi @ cp.diag(weights) @ Psi.T - hyperparam_dict['lambda'] * np.eye(N - D - 1) >> 0,
            E @ weights == 0
        ]
        
        prob = cp.Problem(obj, constraints)
        # prob.solve()
        prob.solve(solver=cp.SCS)

        weights_optimal = weights.value
        # print(weights_optimal)
        # weights_optimal[weights_optimal < sparse_tol] = 0
        # print(weights_optimal)

        stress = incidence_full @ np.diag(weights_optimal) @ incidence_full.T
        
        print(f"There are: {np.count_nonzero(np.abs(weights_optimal) > sparse_tol)} edges")
        print(f"sugeested alpha upper bound: {1/np.max(phi)}")
        
        weights_optimal[np.abs(weights_optimal) < sparse_tol] = 0   # small weights to zero for plotting purpose
        return stress, weights_optimal
    
    import numpy as np

    def yang2019(self):
        
        """
        Compute the stress matrix using the method from Yang 2019.
        
        Parameters:
            nominal_config (numpy.ndarray): A dxn matrix.
        
        Returns:
            numpy.ndarray: The computed stress matrix.
        """
        d, n = self.nominal_config.shape
        nominal_config_aug = np.vstack([self.nominal_config, np.ones((1, n))])
        
        # Validate inputs
        if d > n - 2:
            raise ValueError("n must be at least d+2 for a valid matrix.")
        
        # Define matrix dimensions
        rows = n
        cols = n - d - 1
        
        # Initialize matrix with zeros
        M = np.zeros((rows, cols))
        
        # Populate the matrix with random values where conditions are met
        for j in range(cols):
            for i in range(j, min(j + d + 2, rows)):
                M[i, j] = np.random.rand()
        
        # Vectorize the matrix (convert it to a column vector)
        M_vectorized = M.flatten()
        
        # Get indices of zero elements after vectorization
        zero_idx = np.where(M_vectorized == 0)[0]
        
        # Prepare Q to be solved
        Q_reduced = np.kron(np.eye(n - d - 1), nominal_config_aug)
        Q_reduced = np.delete(Q_reduced, zero_idx, axis=1)
        
        # Solve the zero-free underdetermined system
        U, S, Vt = np.linalg.svd(Q_reduced)
        D_vec_short = Vt[-1, :]
        
        # Put the zero values back
        D_vec_full = np.zeros(n * (n - d - 1))
        nonzero_indices = np.setdiff1d(np.arange(n * (n - d - 1)), zero_idx)
        D_vec_full[nonzero_indices] = D_vec_short
        
        D = D_vec_full.reshape((n, n - d - 1))
        # print(D)
        stress = np.dot(D, D.T)

        stress = stress/np.linalg.norm(stress)
        
        return stress


        
    
    