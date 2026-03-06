import cvxpy as cp
import numpy as np
import scipy

def sparse_id(data, lamb, symmetric=True, zero_indices=None, use_cov=False, solver='MOSEK', verbose=False):
    """
    Solves the sparse identification problem:
    min_A ||A @ data - data||_F^2 + lamb * ||A||_1
    subject to diag(A) = 0 and optionally A[i, j] = 0 for (i, j) in zero_indices.

    Parameters:
    - data: The data matrix (N x M) or the reduced factor B (N x N).
    - lamb: Regularization parameter.
    - symmetric: Whether A should be symmetric.
    - zero_indices: List of tuples (i, j) where A[i, j] should be constrained to 0.
    - solver: Solver to use (default MOSEK).
    - verbose: Solver verbosity.

    Returns:
    - A_est: The estimated adjacency matrix.
    """
    
    N = data.shape[0]
    
    if symmetric:
        A = cp.Variable((N, N), symmetric=True)
    else:
        A = cp.Variable((N, N))

    if use_cov:
        C = data @ data.T / data.shape[1]
        C_sqrt = scipy.linalg.sqrtm(C)
        term1 = cp.sum_squares((A - np.eye(N)) @ C_sqrt)
    else:
        term1 = cp.sum_squares(A @ data - data)
    term2 = lamb * cp.sum(cp.abs(A))
    
    obj = term1 + term2
    
    # Always enforce diagonal zero
    constr = [cp.diag(A) == 0]
    
    # Add optional zero constraints
    if zero_indices:
        for i, j in zero_indices:
            constr.append(cp.abs(A[i, j]) <= 1e-6)

    prob = cp.Problem(cp.Minimize(obj), constr)
    
    # Robust solver strategy
    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        print(f"Solver {solver} failed or unavailable: {e}")

        # REMOVE
        print('RETURNING 0 MARIX')
        return np.zeros((N, N))

        # Fallback solvers
        for fb_solver in [cp.SCS, cp.ECOS]:
            try:
                print(f"Trying fallback solver: {fb_solver}")
                prob.solve(solver=fb_solver, verbose=verbose)
                if prob.status == 'optimal':
                    break
            except Exception:
                continue

    return A.value
