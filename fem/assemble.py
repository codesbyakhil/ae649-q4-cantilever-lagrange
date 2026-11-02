"""Assembly of the global stiffness matrix.

K = sum over elements of k_e placed at the element's global DOFs. Node n has the
global DOFs 2n (u) and 2n + 1 (v). The 64 entries of every k_e are listed as
(row, column, value) triplets (COO format); converting to CSR adds up the
entries that share a (row, column) position, which is exactly the assembly sum.
"""
import numpy as np
import scipy.sparse as sp

from fem.q4 import element_stiffness


def element_dofs(elems):
    """Global DOFs [2 n1, 2 n1 + 1, ..., 2 n4, 2 n4 + 1] of each element: (n_el, 8),
    or (8,) for the node numbers of a single element."""
    return np.stack([2 * elems, 2 * elems + 1], axis=-1).reshape(np.shape(elems)[:-1] + (8,))


def assemble_stiffness(nodes, elems, D, t, n_gauss=2):
    """Global K (sparse CSR, 2 N x 2 N for N nodes)."""
    n_dof = 2 * len(nodes)
    k = element_stiffness(nodes[elems], D, t, n_gauss)   # (n_el, 8, 8), all elements at once
    dofs = element_dofs(elems)                           # (n_el, 8)
    rows = np.repeat(dofs, 8, axis=1)    # k[e].ravel() lists k[e, a, b] with b fastest:
    cols = np.tile(dofs, 8)              # row dofs[e, a], column dofs[e, b]
    K = sp.coo_matrix((k.ravel(), (rows.ravel(), cols.ravel())), shape=(n_dof, n_dof))
    return K.tocsr()
