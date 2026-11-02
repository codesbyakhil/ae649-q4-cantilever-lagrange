"""Plane-stress constitutive matrix (Cook Eq. 3.1-3).

3D Hooke's law with sigma_z = tau_yz = tau_zx = 0 leaves
    eps_x = (sigma_x - nu sigma_y) / E
    eps_y = (sigma_y - nu sigma_x) / E
    gamma_xy = tau_xy / G,          G = E / (2 (1 + nu))
Solving for the stresses gives {sigma_x, sigma_y, tau_xy} = D {eps_x, eps_y, gamma_xy} with
    D = E / (1 - nu^2) [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]
gamma_xy is the engineering shear strain, and E / (1 - nu^2) * (1 - nu) / 2 = G.
"""
import numpy as np


def plane_stress_D(E, nu):
    """D (3 x 3) such that [sigma_x, sigma_y, tau_xy] = D [eps_x, eps_y, gamma_xy]."""
    return E / (1.0 - nu**2) * np.array([[1.0, nu, 0.0],
                                         [nu, 1.0, 0.0],
                                         [0.0, 0.0, (1.0 - nu) / 2.0]])
