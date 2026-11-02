"""1D bar with both ends prescribed, solved with Lagrange multipliers (validation item V1).

Governing equation   EA u'' + q = 0   on 0 < x < L
Boundary conditions  u(0) = 0,  u(L) = U
Exact solution       u(x) = q x (L - x) / (2 EA) + U x / L
Axial force          N(x) = EA u'(x) = q (L - 2x) / 2 + EA U / L

Two-node linear elements, constant EA, uniform q and consistent loads. For
this case the FE solution is exact at the nodes (Cook p. 47), so the nodal
values must equal u(x_i) to round-off. The two end conditions are C d = g with
g = [0, U], imposed by the same Lagrange-multiplier solver as the 2D problem
(fem.lagrange.solve_lagrange).

Run from the project root:  python -m bar1d.bar
"""
import numpy as np
import scipy.sparse as sp

from fem.lagrange import constraint_matrix, solve_lagrange

# The V1 bar (SI): steel rod, 1 m long, 1 cm^2 cross-section.
L_BAR = 1.0             # m
EA_BAR = 200e9 * 1e-4   # N    (E = 200 GPa, A = 1e-4 m^2)
Q_BAR = 1.0e3           # N/m  uniform axial load
U_END = 1.0e-5          # m    prescribed displacement at x = L


def element_stiffness(EA, h):
    """k_e = (EA / h) [[1, -1], [-1, 1]]."""
    return EA / h * np.array([[1.0, -1.0], [-1.0, 1.0]])


def element_load(q, h):
    """Consistent load for uniform q: f_e = int_0^h N^T q dx = (q h / 2) [1, 1]."""
    return q * h / 2.0 * np.array([1.0, 1.0])


def assemble(x, EA, q):
    """Global K (sparse CSR) and f for node coordinates x.

    Element e joins nodes e and e + 1; global DOF i is the displacement of node i.
    """
    n = len(x)
    rows, cols, vals = [], [], []
    f = np.zeros(n)
    for e in range(n - 1):
        h = x[e + 1] - x[e]
        dofs = [e, e + 1]
        ke = element_stiffness(EA, h)
        for a in range(2):
            for b in range(2):
                rows.append(dofs[a])
                cols.append(dofs[b])
                vals.append(ke[a, b])
        f[dofs] += element_load(q, h)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()  # duplicates are summed
    return K, f


def exact_u(x, L, EA, q, U):
    """u(x) = q x (L - x) / (2 EA) + U x / L."""
    return q * x * (L - x) / (2.0 * EA) + U * x / L


def exact_support_forces(L, EA, q, U):
    """Forces of the supports on the bar at x = 0 and x = L (positive in +x).

    From N(x) = EA u'(x): the support at x = 0 pushes with -N(0), the one at
    x = L with +N(L). Their sum is -q L (overall equilibrium).
    """
    N0 = q * L / 2.0 + EA * U / L
    NL = -q * L / 2.0 + EA * U / L
    return np.array([-N0, NL])


def solve_bar(x, EA=EA_BAR, q=Q_BAR, U=U_END):
    """Assemble and solve the bar on nodes x with u(0) = 0, u(L) = U by Lagrange multipliers.

    Returns K, f, C, g, d, lam, R.
    """
    K, f = assemble(x, EA, q)
    n = len(x)
    C = constraint_matrix(np.array([0, n - 1]), n)
    g = np.array([0.0, U])
    d, lam, R = solve_lagrange(K, f, C, g)
    return K, f, C, g, d, lam, R


if __name__ == "__main__":
    n_el = 8
    x = np.linspace(0.0, L_BAR, n_el + 1)
    K, f, C, g, d, lam, R = solve_bar(x)
    u_ex = exact_u(x, L_BAR, EA_BAR, Q_BAR, U_END)
    print(f"Bar: L = {L_BAR} m, EA = {EA_BAR:.3e} N, q = {Q_BAR} N/m, U = {U_END:.1e} m, "
          f"{n_el} elements")
    print(f"{'x [m]':>7s} {'exact u [m]':>14s} {'FE d [m]':>14s} {'d - exact [m]':>14s}")
    for i in range(len(x)):
        print(f"{x[i]:7.3f} {u_ex[i]:14.6e} {d[i]:14.6e} {d[i] - u_ex[i]:14.3e}")
    print(f"C d - g          = {C @ d - g}")
    print(f"-lambda          = {-lam} N")
    print(f"exact N-based    = {exact_support_forces(L_BAR, EA_BAR, Q_BAR, U_END)} N")
