"""Closed-form counterparts of the FE results V2-V5 (the validation table in README.md).

Each function evaluates a hand-derived formula. Nothing here calls the FE code.
The derivations are written out in the docstrings and re-derived symbolically
in checks/derivations.py.
"""
import numpy as np

# Parent coordinates of the Q4 nodes 1-4 (counter-clockwise from (-1, -1)).
S_NODE = np.array([-1.0, 1.0, 1.0, -1.0])
R_NODE = np.array([-1.0, -1.0, 1.0, 1.0])


# ---------------------------------------------------------------- V2: element stiffness
def q4_rectangle_stiffness(a, b, E, nu, t):
    """Exact stiffness (8 x 8) of a rectangular Q4 element 2a x 2b (a, b = half-lengths).

    With x = a s, y = b r:  N_i,x = s_i (1 + r r_i) / (4a),  N_i,y = r_i (1 + s s_i) / (4b),
    dA = a b ds dr. Integrating exactly over -1 <= s, r <= 1:
        Ixx_ij = int N_i,x N_j,x dA = b / (4a) s_i s_j (1 + r_i r_j / 3)
        Iyy_ij = int N_i,y N_j,y dA = a / (4b) r_i r_j (1 + s_i s_j / 3)
        Ixy_ij = int N_i,x N_j,y dA = s_i r_j / 4
    k = t int B^T D B dA then gives the 2 x 2 block of nodes i, j:
        k_uu = t (D11 Ixx_ij + D33 Iyy_ij)      k_uv = t (D12 Ixy_ij + D33 Ixy_ji)
        k_vu = t (D12 Ixy_ji + D33 Ixy_ij)      k_vv = t (D22 Iyy_ij + D33 Ixx_ij)
    """
    c = E / (1.0 - nu**2)
    D11, D12, D22, D33 = c, c * nu, c, c * (1.0 - nu) / 2.0
    k = np.zeros((8, 8))
    for i in range(4):
        for j in range(4):
            Ixx = b / (4.0 * a) * S_NODE[i] * S_NODE[j] * (1.0 + R_NODE[i] * R_NODE[j] / 3.0)
            Iyy = a / (4.0 * b) * R_NODE[i] * R_NODE[j] * (1.0 + S_NODE[i] * S_NODE[j] / 3.0)
            Ixy_ij = S_NODE[i] * R_NODE[j] / 4.0
            Ixy_ji = S_NODE[j] * R_NODE[i] / 4.0
            k[2 * i, 2 * j] = t * (D11 * Ixx + D33 * Iyy)
            k[2 * i, 2 * j + 1] = t * (D12 * Ixy_ij + D33 * Ixy_ji)
            k[2 * i + 1, 2 * j] = t * (D12 * Ixy_ji + D33 * Ixy_ij)
            k[2 * i + 1, 2 * j + 1] = t * (D22 * Iyy + D33 * Ixx)
    return k


def q4_k11(a, b, E, nu, t):
    """k11 = E t / (1 - nu^2) [b / (3a) + (1 - nu) a / (6b)]  (u1-u1 entry)."""
    return E * t / (1.0 - nu**2) * (b / (3.0 * a) + (1.0 - nu) * a / (6.0 * b))


def q4_k12(E, nu, t):
    """k12 = E t / (8 (1 - nu))  (u1-v1 entry; independent of a and b)."""
    return E * t / (8.0 * (1.0 - nu))


# ---------------------------------------------------------------- V4: Q4 in pure bending
def cook_bending_ratio(a, b, nu):
    """theta_FE / theta_exact = (1 - nu^2) / [1 + (1 - nu)/2 (a/b)^2]   (Cook Eq. 3.6-11, p. 99).

    a, b = element half-length and half-depth. In bending the Q4 element has
    eps_x exact, eps_y = 0 (gives the factor 1 - nu^2) and a parasitic shear strain
    gamma_xy proportional to x (gives the (a/b)^2 term). Equal moments, so the
    rotation ratio is the inverse ratio of the strain energies at equal rotation.
    """
    return (1.0 - nu**2) / (1.0 + (1.0 - nu) / 2.0 * (a / b) ** 2)


# ---------------------------------------------------------------- V5: one-element cantilever
def one_element_cantilever(P, L, depth, E, nu, t):
    """Hand solution of the 1 x 1 mesh clamped at nodes 1 (0, -D/2) and 4 (0, D/2),
    with P/2 on v at nodes 2 (L, -D/2) and 3 (L, D/2).

    Symmetry (u odd in y, v even in y) leaves two unknowns, u3 = -u2 = q and
    v2 = v3 = w. With a = L/2, b = D/2, D11 = E/(1 - nu^2), G = E/(2(1 + nu)):
        u = q r (1 + s)/2,  v = w (1 + s)/2
        eps_x = q r / (2a),  eps_y = 0,  gamma_xy = q (1 + s)/(2b) + w/(2a)
    Minimising Pi(q, w) = U - P w gives
        q = -3 a^2 P / [t (G a^2 + D11 b^2)]
        w = a P (D11 b^2 + 4 G a^2) / [t G b (G a^2 + D11 b^2)]
    which in terms of L and D are
        v_tip = 4 P L (1 + nu) [2 L^2 (1 - nu) + D^2] / {E t D [L^2 (1 - nu) + 2 D^2]}
        u_top = -u_bottom = -6 P L^2 (1 - nu^2) / {E t [L^2 (1 - nu) + 2 D^2]}
    Reactions on the two clamped nodes: sum Fx = 0 and sum M = 0 about the origin
    give Rx = -P L / D at the bottom node and +P L / D at the top node; sum Fy = 0
    gives Ry_bottom + Ry_top = -P, and symmetry splits it equally: -P/2 each.
    """
    v_tip = (4.0 * P * L * (1.0 + nu) * (2.0 * L**2 * (1.0 - nu) + depth**2)
             / (E * t * depth * (L**2 * (1.0 - nu) + 2.0 * depth**2)))
    u_top = -6.0 * P * L**2 * (1.0 - nu**2) / (E * t * (L**2 * (1.0 - nu) + 2.0 * depth**2))
    return {"v_tip": v_tip, "u_top": u_top, "u_bottom": -u_top,
            "Rx_bottom": -P * L / depth, "Rx_top": P * L / depth,
            "Ry_bottom": -P / 2.0, "Ry_top": -P / 2.0}


# ---------------------------------------------------------------- V3: uniform tension
def uniform_tension_displacement(x, y, sigma0, E, nu):
    """Exact plane-stress field for sigma_x = sigma0, sigma_y = tau_xy = 0, with u = 0 on x = 0
    and v = 0 at y = 0:  u = sigma0 x / E,  v = -nu sigma0 y / E."""
    return sigma0 * x / E, -nu * sigma0 * y / E
