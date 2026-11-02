"""Four-node bilinear isoparametric quadrilateral (Q4), plane stress.

Parent coordinates (s, r), s horizontal. Nodes 1(-1,-1), 2(1,-1), 3(1,1), 4(-1,1),
counter-clockwise. Element DOFs d_e = [u1, v1, u2, v2, u3, v3, u4, v4].

Shape functions   N_i = 1/4 (1 + s s_i)(1 + r r_i)
Mapping           x = sum N_i x_i,  y = sum N_i y_i
Jacobian          [J] = [[x_s, y_s], [x_r, y_r]] = [N_s; N_r] [x y]
Chain rule        [N_x; N_y] = inv(J) [N_s; N_r]
Strains           [eps_x, eps_y, gamma_xy] = B d_e
Stiffness         k = t int_-1^1 int_-1^1 B^T D B det(J) ds dr
                    = t sum_i sum_j B^T D B det(J) w_i w_j   (Gauss rule, 2 x 2 by default)

One element or many at once. xy holds the nodal coordinates of one element
(4 x 2) or of a stack of elements (n_el x 4 x 2). The '...' in the indexing
stands for that optional leading element index, and @ multiplies the matrices of
every element in the stack. The formulas are the same in both cases; working on
the whole stack avoids a slow Python loop over elements.
"""
import numpy as np

# Parent coordinates of nodes 1-4.
S_NODE = np.array([-1.0, 1.0, 1.0, -1.0])
R_NODE = np.array([-1.0, -1.0, 1.0, 1.0])


def shape_functions(s, r):
    """N (4,): N_i = 1/4 (1 + s s_i)(1 + r r_i)."""
    return 0.25 * (1.0 + s * S_NODE) * (1.0 + r * R_NODE)


def shape_derivatives(s, r):
    """[N_s; N_r] (2 x 4): N_i,s = 1/4 s_i (1 + r r_i),  N_i,r = 1/4 r_i (1 + s s_i)."""
    dN_ds = 0.25 * S_NODE * (1.0 + r * R_NODE)
    dN_dr = 0.25 * R_NODE * (1.0 + s * S_NODE)
    return np.array([dN_ds, dN_dr])


def jacobian(s, r, xy):
    """[J] = [[x_s, y_s], [x_r, y_r]] = [N_s; N_r] @ xy, shape (..., 2, 2)."""
    return shape_derivatives(s, r) @ xy


def strain_displacement(s, r, xy):
    """B (..., 3, 8) and det J (...) at the parent point (s, r)."""
    J = jacobian(s, r, xy)
    detJ = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    if np.any(detJ <= 0.0):
        raise ValueError("det J <= 0: element nodes are not counter-clockwise or the element is too distorted")
    J_inv = np.empty_like(J)                     # inverse of a 2 x 2 matrix
    J_inv[..., 0, 0] = J[..., 1, 1] / detJ
    J_inv[..., 0, 1] = -J[..., 0, 1] / detJ
    J_inv[..., 1, 0] = -J[..., 1, 0] / detJ
    J_inv[..., 1, 1] = J[..., 0, 0] / detJ
    dN_xy = J_inv @ shape_derivatives(s, r)      # rows: N_i,x and N_i,y
    B = np.zeros(J.shape[:-2] + (3, 8))
    B[..., 0, 0::2] = dN_xy[..., 0, :]           # eps_x    = sum N_i,x u_i
    B[..., 1, 1::2] = dN_xy[..., 1, :]           # eps_y    = sum N_i,y v_i
    B[..., 2, 0::2] = dN_xy[..., 1, :]           # gamma_xy = sum (N_i,y u_i + N_i,x v_i)
    B[..., 2, 1::2] = dN_xy[..., 0, :]
    return B, detJ


def gauss_rule(n):
    """Points and weights of the n-point Gauss-Legendre rule on [-1, 1].

    An n-point rule integrates polynomials up to degree 2n - 1 exactly
    (n = 2: points +-1/sqrt(3), weights 1, exact to degree 3).
    """
    return np.polynomial.legendre.leggauss(n)


def element_stiffness(xy, D, t, n_gauss=2):
    """k (..., 8, 8) = t sum_i sum_j B^T D B det(J) w_i w_j over an n_gauss x n_gauss rule."""
    points, weights = gauss_rule(n_gauss)
    k = np.zeros(np.shape(xy)[:-2] + (8, 8))
    for s, w_s in zip(points, weights):
        for r, w_r in zip(points, weights):
            B, detJ = strain_displacement(s, r, xy)
            BT = np.swapaxes(B, -1, -2)
            k += BT @ D @ B * np.asarray(t * detJ * w_s * w_r)[..., None, None]
    return k
