"""Post-processing: Gauss-point stresses, tip deflection, force resultants and
convergence measures (observed order, Richardson extrapolation, Cook Section 9.7)."""
import numpy as np

from fem.assemble import element_dofs
from fem.mesh import edge_nodes
from fem.q4 import gauss_rule, shape_functions, strain_displacement


def gauss_point_stresses(nodes, elems, d, D, n_gauss=2):
    """Coordinates xy (n_el, n_gp, 2) and stresses [sigma_x, sigma_y, tau_xy] (n_el, n_gp, 3)
    at the Gauss points, for all elements at once: sigma = D B d_e, (x, y) = sum N_i (x_i, y_i).
    Gauss points are ordered as in element_stiffness (s outer loop, r inner loop)."""
    points, _ = gauss_rule(n_gauss)
    xy_e = nodes[elems]                          # (n_el, 4, 2)
    d_e = d[element_dofs(elems)]                 # (n_el, 8)
    xy, sigma = [], []
    for s in points:
        for r in points:
            B, detJ = strain_displacement(s, r, xy_e)     # (n_el, 3, 8)
            eps = (B @ d_e[:, :, None])[:, :, 0]          # (n_el, 3) strains
            sigma.append(eps @ D.T)                       # each row: sigma = D eps
            xy.append(shape_functions(s, r) @ xy_e)       # (n_el, 2)
    return np.stack(xy, axis=1), np.stack(sigma, axis=1)


def tip_deflection(nodes, d, L):
    """v at (L, 0). Along the end edge the FE field is linear between nodes, so this is
    the mid-height node's v when ny is even and the linear interpolation otherwise
    (the mean of the two end nodes when ny = 1)."""
    right = edge_nodes(nodes, L)
    return np.interp(0.0, nodes[right, 1], d[2 * right + 1])


def force_resultants(nodes, F):
    """Sum of x forces, sum of y forces and moment about the origin, sum (x F_y - y F_x),
    of a nodal force vector F (2 N,)."""
    Fx, Fy = F[0::2], F[1::2]
    return np.sum(Fx), np.sum(Fy), np.sum(nodes[:, 0] * Fy - nodes[:, 1] * Fx)


def observed_order(phi1, phi2, phi3):
    """Order q from three meshes, each with half the element size of the previous one:
    phi = phi_inf + c h^q  =>  (phi1 - phi2) / (phi2 - phi3) = 2^q."""
    return np.log2((phi1 - phi2) / (phi2 - phi3))


def richardson(phi_coarse, phi_fine, q, ratio=2.0):
    """Richardson extrapolation (Cook Eq. 9.7-1) from two meshes with h_coarse / h_fine = ratio:
    phi_inf = (ratio^q phi_fine - phi_coarse) / (ratio^q - 1)."""
    return (ratio**q * phi_fine - phi_coarse) / (ratio**q - 1.0)
