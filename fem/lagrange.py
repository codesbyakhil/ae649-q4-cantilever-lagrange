"""Constraints C d = g imposed by Lagrange multipliers (THEORY.md section 5).

Augmented functional
    Pi*(d, lam) = 1/2 d^T K d - d^T f + lam^T (C d - g)
Stationarity, dPi*/dd = 0 and dPi*/dlam = 0, gives the saddle-point system

    [ K   C^T ] { d   }   { f }
    [ C   0   ] { lam } = { g }

The matrix is symmetric but indefinite (zero diagonal block), so it is solved
by sparse LU with pivoting (scipy.sparse.linalg.spsolve, SuperLU), not by
Cholesky. The columns are reordered by minimum degree on the pattern of A + A^T
(MMD_AT_PLUS_A), which suits a matrix with a symmetric pattern and is about
twice as fast as the default ordering here, with the same accuracy.

Reactions. The first block row is K d + C^T lam = f, so the support forces are
    R = K d - f = -C^T lam.

Scaling. K holds entries of order 1e10 N/m while C holds ones, which makes the
augmented matrix very badly conditioned. Both the constraint rows and the
constraint columns are multiplied by s = max(diag K):

    [ K     s C^T ] { d     }   { f   }
    [ s C   0     ] { lam_s } = { s g }

This is A_s = S A S with S = diag(I, s I), so the solution is S^-1 [d; lam]:
d is unchanged and lam_s = lam / s. The second row s C d = s g is the same
constraint. Reactions: R = -(s C)^T lam_s = -C^T lam.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def constraint_matrix(dofs, n_dof):
    """C (m x n_dof, sparse) for prescribing m DOFs: row i has a 1 in column dofs[i].

    Then C d = d[dofs], so C d = g prescribes d[dofs[i]] = g[i].
    """
    m = len(dofs)
    return sp.csr_matrix((np.ones(m), (np.arange(m), dofs)), shape=(m, n_dof))


def augmented_matrix(K, C, s=1.0):
    """[[K, s C^T], [s C, 0]] as a sparse CSC matrix (s = 1: unscaled)."""
    Cs = s * C
    return sp.bmat([[K, Cs.T], [Cs, None]], format="csc")


def solve_lagrange(K, f, C, g, scale=True):
    """Solve [[K, C^T], [C, 0]] [d; lam] = [f; g] by sparse LU.

    With scale=True (always used in the project) the constraint rows and columns
    are multiplied by s = max(diag K) before the solve (see the module docstring).

    Returns
        d   : displacements (n,)
        lam : Lagrange multipliers in physical units (m,), lam = s * lam_s
        R   : reactions on the structure (n,), R = -(s C)^T lam_s = -C^T lam
    """
    n = K.shape[0]
    s = K.diagonal().max() if scale else 1.0
    A = augmented_matrix(K, C, s)
    rhs = np.concatenate([f, s * g])
    sol = spsolve(A, rhs, permc_spec="MMD_AT_PLUS_A")
    d = sol[:n]
    lam_s = sol[n:]
    R = -((s * C).T @ lam_s)
    lam = s * lam_s
    return d, lam, R
