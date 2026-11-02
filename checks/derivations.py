"""Symbolic (sympy) derivations of every analytical counterpart in the validation table of README.md.

Each function derives a result from first principles and returns
(derived expression, expression used in checks/ or fem/). tests/test_derivations.py
asserts that each pair simplifies to the same thing, so the closed forms used in
the tests and studies are proven, not typed in.

Run from the project root:  python -m checks.derivations
"""
from functools import lru_cache

import sympy as sp

E, nu, t, a, b, L, Dp, P = sp.symbols("E nu t a b L D P", positive=True)
s, r, x, y = sp.symbols("s r x y", real=True)
S_NODE = [-1, 1, 1, -1]
R_NODE = [-1, -1, 1, 1]


def plane_stress_matrix():
    """Invert 3D Hooke's law with sigma_z = 0: eps = S sigma  =>  D = S^-1."""
    G = E / (2 * (1 + nu))
    S = sp.Matrix([[1 / E, -nu / E, 0], [-nu / E, 1 / E, 0], [0, 0, 1 / G]])
    D_used = E / (1 - nu**2) * sp.Matrix([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    return sp.simplify(S.inv()), D_used


def q4_rectangle_B():
    """B (3 x 8) of a 2a x 2b rectangle: x = a s, y = b r, so d/dx = (1/a) d/ds, d/dy = (1/b) d/dr."""
    N = [(1 + s * si) * (1 + r * ri) / 4 for si, ri in zip(S_NODE, R_NODE)]
    B = sp.zeros(3, 8)
    for i in range(4):
        B[0, 2 * i] = sp.diff(N[i], s) / a
        B[1, 2 * i + 1] = sp.diff(N[i], r) / b
        B[2, 2 * i] = sp.diff(N[i], r) / b
        B[2, 2 * i + 1] = sp.diff(N[i], s) / a
    return B


@lru_cache(maxsize=None)
def q4_rectangle_stiffness_exact():
    """k = t int int B^T D B a b ds dr, integrated exactly (computed once, then cached)."""
    B = q4_rectangle_B()
    D = plane_stress_matrix()[1]
    integrand = sp.expand(B.T * D * B * a * b * t)
    return integrand.applyfunc(lambda e: sp.integrate(e, (s, -1, 1), (r, -1, 1)))


def q4_stiffness_two_by_two_gauss_vs_exact():
    """2 x 2 Gauss (points +-1/sqrt 3, weights 1) applied symbolically vs exact integration."""
    B = q4_rectangle_B()
    D = plane_stress_matrix()[1]
    integrand = B.T * D * B * a * b * t
    g = 1 / sp.sqrt(3)
    k_gauss = sp.zeros(8, 8)
    for sg in (-g, g):
        for rg in (-g, g):
            k_gauss += integrand.subs({s: sg, r: rg})
    return q4_rectangle_stiffness_exact(), k_gauss


def q4_k11_k12():
    """Entries k11 (u1-u1) and k12 (u1-v1) vs the formulas of validation item V2."""
    k = q4_rectangle_stiffness_exact()
    k11 = E * t / (1 - nu**2) * (b / (3 * a) + (1 - nu) * a / (6 * b))
    k12 = E * t / (8 * (1 - nu))
    return (k[0, 0], k11), (k[0, 1], k12)


def cook_bending_ratio():
    """Cook Eq. 3.6-11 from the strain energies of Eqs. 3.6-9 (block) and 3.6-10 (Q4).

    At equal moment M: U = M theta / 2 = k theta^2 / 2, so theta_el / theta_b = k_b / k_el
    = U_b / U_el evaluated at the same theta.
    """
    th = sp.symbols("theta", positive=True)
    D = plane_stress_matrix()[1]
    eps_block = sp.Matrix([-th * y / (2 * a), nu * th * y / (2 * a), 0])
    eps_q4 = sp.Matrix([-th * y / (2 * a), 0, -th * x / (2 * a)])
    U_b = t / 2 * sp.integrate((eps_block.T * D * eps_block)[0], (x, -a, a), (y, -b, b))
    U_el = t / 2 * sp.integrate((eps_q4.T * D * eps_q4)[0], (x, -a, a), (y, -b, b))
    ratio_used = (1 - nu**2) / (1 + (1 - nu) / 2 * (a / b) ** 2)
    return sp.simplify(U_b / U_el), ratio_used


def one_element_cantilever():
    """V5: 1 x 1 mesh, nodes 1 and 4 clamped, P/2 on v at nodes 2 and 3.

    Hand method: the clamp is built into the trial field (u1 = v1 = u4 = v4 = 0) and
    the potential energy Pi = 1/2 d^T k d - f^T d is minimised over the four free
    DOFs u2, v2, u3, v3 (no symmetry assumed). Then a = L/2, b = D/2.
    Returns pairs (derived, used in checks/closed_form.py) for v_tip, u_top, u_bottom
    and the reactions R = k d - f at u1, v1, u4, v4.
    """
    k = q4_rectangle_stiffness_exact().subs({a: L / 2, b: Dp / 2})
    u2, v2, u3, v3 = sp.symbols("u2 v2 u3 v3", real=True)
    d = sp.Matrix([0, 0, u2, v2, u3, v3, 0, 0])
    f = sp.Matrix([0, 0, 0, P / 2, 0, P / 2, 0, 0])
    Pi = (d.T * k * d / 2 - f.T * d)[0]
    sol = sp.solve([sp.diff(Pi, q) for q in (u2, v2, u3, v3)], [u2, v2, u3, v3], dict=True)[0]
    d_sol = d.subs(sol)
    R = sp.simplify(k * d_sol - f)
    v_tip = 4 * P * L * (1 + nu) * (2 * L**2 * (1 - nu) + Dp**2) / (E * t * Dp * (L**2 * (1 - nu) + 2 * Dp**2))
    u_top = -6 * P * L**2 * (1 - nu**2) / (E * t * (L**2 * (1 - nu) + 2 * Dp**2))
    return {"v_tip (node 2)": (sol[v2], v_tip), "v_tip (node 3)": (sol[v3], v_tip),
            "u_top (node 3)": (sol[u3], u_top), "u_bottom (node 2)": (sol[u2], -u_top),
            "Rx_bottom (node 1)": (R[0], -P * L / Dp), "Ry_bottom (node 1)": (R[1], -P / 2),
            "Rx_top (node 4)": (R[6], P * L / Dp), "Ry_top (node 4)": (R[7], -P / 2)}


def beam_theory():
    """Euler-Bernoulli: E I v'' = M = P (L - x), v(0) = v'(0) = 0.
    Timoshenko shear part: v_s' = V / (kappa G A) with V = P, v_s(0) = 0.
    I = int y^2 dA over the t x D rectangle."""
    EI, kGA = sp.symbols("EI kappaGA", positive=True)
    v = sp.Function("v")
    sol = sp.dsolve(sp.Eq(EI * v(x).diff(x, 2), P * (L - x)), v(x),
                    ics={v(0): 0, v(x).diff(x).subs(x, 0): 0})
    v_shear = sp.integrate(P / kGA, (x, 0, L))
    I = sp.integrate(y**2 * t, (y, -Dp / 2, Dp / 2))
    # Shear stress from equilibrium: d(sigma_xx)/dx + d(tau)/dy = 0, tau(+-D/2) = 0.
    Isym = sp.symbols("I", positive=True)
    sigma_xx = -P * (L - x) * y / Isym
    c = sp.symbols("c")
    tau = sp.integrate(-sp.diff(sigma_xx, x), y) + c
    tau = tau.subs(c, sp.solve(tau.subs(y, Dp / 2), c)[0])
    tau_used = P * (Dp**2 / 4 - y**2) / (2 * Isym)
    shear_force = sp.integrate(tau_used * t, (y, -Dp / 2, Dp / 2)).subs(Isym, t * Dp**3 / 12)
    return {"EB tip": (sol.rhs.subs(x, L), P * L**3 / (3 * EI)),
            "shear tip": (v_shear, P * L / kGA),
            "I": (I, t * Dp**3 / 12),
            "tau_xy from equilibrium": (tau, tau_used),
            "tau_xy(-D/2) = 0": (tau_used.subs(y, -Dp / 2), 0),
            "int tau t dy = P": (shear_force, P)}


def bar_solution():
    """V1: u = q x (L - x) / (2 EA) + U x / L solves EA u'' + q = 0, u(0) = 0, u(L) = U.
    Support forces: -N(0) at x = 0 and +N(L) at x = L, N = EA u'."""
    q, EA, U = sp.symbols("q EA U", real=True)
    u = q * x * (L - x) / (2 * EA) + U * x / L
    N = EA * sp.diff(u, x)
    return {"ODE residual": (EA * sp.diff(u, x, 2) + q, 0),
            "u(0)": (u.subs(x, 0), 0), "u(L)": (u.subs(x, L), U),
            "R(0) = -N(0)": (-N.subs(x, 0), -q * L / 2 - EA * U / L),
            "R(L) = +N(L)": (N.subs(x, L), -q * L / 2 + EA * U / L),
            "R(0) + R(L) + q L": (-N.subs(x, 0) + N.subs(x, L) + q * L, 0)}


def uniform_tension():
    """V3: u = sigma0 x / E, v = -nu sigma0 y / E gives sigma = [sigma0, 0, 0] everywhere,
    so equilibrium holds, the top and bottom faces are traction-free, the end x = L
    carries sigma0, and u = 0 on x = 0, v = 0 at y = 0."""
    sig0 = sp.symbols("sigma0", real=True)
    u = sig0 * x / E
    v = -nu * sig0 * y / E
    eps = sp.Matrix([sp.diff(u, x), sp.diff(v, y), sp.diff(u, y) + sp.diff(v, x)])
    sigma = sp.simplify(plane_stress_matrix()[1] * eps)
    return {"sigma_x": (sigma[0], sig0), "sigma_y": (sigma[1], 0), "tau_xy": (sigma[2], 0),
            "u(0, y)": (u.subs(x, 0), 0), "v(x, 0)": (v.subs(y, 0), 0)}


def consistent_edge_load():
    """Uniform traction tau on a 2-node edge of length h (thickness t): f_i = t int N_i tau ds."""
    tau, h, xi = sp.symbols("tau h xi", real=True)
    N1, N2 = (1 - xi) / 2, (1 + xi) / 2              # linear along the edge, xi in [-1, 1]
    f1 = sp.integrate(t * N1 * tau * h / 2, (xi, -1, 1))
    f2 = sp.integrate(t * N2 * tau * h / 2, (xi, -1, 1))
    return {"f1": (f1, tau * t * h / 2), "f2": (f2, tau * t * h / 2)}


def lagrange_equivalence():
    """C = [0 0 1] on a generic symmetric 3 x 3 K: the augmented system gives the
    elimination displacements and lam = f_c - K_cf d_f - K_cc g = -R_c.
    Scaling the constraint row and column by s leaves d unchanged and gives lam_s = lam / s."""
    k11, k12, k13, k22, k23, k33, f1, f2, f3, g, sc = sp.symbols(
        "k11 k12 k13 k22 k23 k33 f1 f2 f3 g s", real=True)
    K = sp.Matrix([[k11, k12, k13], [k12, k22, k23], [k13, k23, k33]])
    f = sp.Matrix([f1, f2, f3])
    C = sp.Matrix([[0, 0, 1]])
    A = K.row_join(C.T).col_join(C.row_join(sp.zeros(1, 1)))
    sol = A.LUsolve(f.col_join(sp.Matrix([g])))
    A_s = K.row_join(sc * C.T).col_join((sc * C).row_join(sp.zeros(1, 1)))
    sol_s = A_s.LUsolve(f.col_join(sp.Matrix([sc * g])))
    # Elimination: d3 = g, K_ff d_f = f_f - K_fc g.
    d_f = K[:2, :2].LUsolve(f[:2, :] - K[:2, 2] * g)
    d = sp.Matrix([d_f[0], d_f[1], g])
    R = K * d - f
    return {"d1": (sol[0], d[0]), "d2": (sol[1], d[1]), "d3": (sol[2], g),
            "lam = -R3": (sol[3], -R[2]),
            "lam = f_c - K_cf d_f - K_cc g": (sol[3], f3 - (K[2, :2] * d_f)[0] - k33 * g),
            "scaled d1": (sol_s[0], sol[0]), "scaled lam_s = lam / s": (sol_s[3], sol[3] / sc)}


def richardson():
    """phi_i = phi_inf + c h_i^q on two meshes => phi_inf (Cook Eq. 9.7-1);
    three meshes with h halved each time => q = log2[(phi1 - phi2) / (phi2 - phi3)]."""
    phi1, phi2, h1, h2, q, phi_inf, c = sp.symbols("phi1 phi2 h1 h2 q phi_inf c", positive=True)
    sol = sp.solve([phi1 - phi_inf - c * h1**q, phi2 - phi_inf - c * h2**q], [phi_inf, c], dict=True)[0]
    used = (phi2 - phi1 * (h2 / h1) ** q) / (1 - (h2 / h1) ** q)
    h = sp.symbols("h", positive=True)
    p1, p2, p3 = (phi_inf + c * (h / 2**i) ** q for i in range(3))
    # (phi1 - phi2) / (phi2 - phi3) = 2^q, so its base-2 logarithm is the order q.
    return {"phi_inf": (sol[phi_inf], used), "(phi1 - phi2) / (phi2 - phi3)": ((p1 - p2) / (p2 - p3), 2**q)}


ALL = {
    "plane-stress D": lambda: {"D": plane_stress_matrix()},
    "Q4 rectangle: 2x2 Gauss = exact": lambda: {"k": q4_stiffness_two_by_two_gauss_vs_exact()},
    "Q4 rectangle: k11, k12": lambda: dict(zip(["k11", "k12"], q4_k11_k12())),
    "Cook Eq. 3.6-11": lambda: {"theta_el / theta_b": cook_bending_ratio()},
    "one-element cantilever (V5)": one_element_cantilever,
    "beam theory": beam_theory,
    "1D bar (V1)": bar_solution,
    "uniform tension (V3)": uniform_tension,
    "consistent edge load": consistent_edge_load,
    "Lagrange = elimination, scaling": lagrange_equivalence,
    "Richardson extrapolation": richardson,
}


def difference_is_zero(derived, used):
    """True if derived - used simplifies to zero (matrices: every entry)."""
    diff = sp.Matrix(derived) - sp.Matrix(used) if isinstance(derived, sp.MatrixBase) else derived - used
    return sp.simplify(diff) == (sp.zeros(*diff.shape) if isinstance(diff, sp.MatrixBase) else 0)


if __name__ == "__main__":
    for name, derive in ALL.items():
        print(f"== {name}")
        for label, (derived, used) in derive().items():
            status = "proved" if difference_is_zero(derived, used) else "MISMATCH"
            shown = "" if isinstance(derived, sp.MatrixBase) else f"  =  {sp.simplify(derived)}"
            print(f"   {label}: {status}{shown}")
