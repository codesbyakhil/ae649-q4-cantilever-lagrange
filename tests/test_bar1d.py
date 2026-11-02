"""V1: 1D bar with both ends prescribed, solved with Lagrange multipliers.

Analytical counterpart: u(x) = q x (L - x) / (2 EA) + U x / L and the support
forces -N(0), +N(L) from N = EA u'. Elimination is written here only, as an
independent check of the Lagrange-multiplier solve.
"""
import numpy as np
import pytest

from bar1d.bar import (EA_BAR, L_BAR, Q_BAR, U_END, assemble, exact_support_forces,
                       exact_u, solve_bar)

# Uniform meshes of 2, 3, 8 and 32 elements, plus one graded mesh
# (nodal exactness does not depend on equal spacing).
MESHES = [np.linspace(0.0, L_BAR, n + 1) for n in (2, 3, 8, 32)]
MESHES.append(L_BAR * np.linspace(0.0, 1.0, 9) ** 2)


def solve_by_elimination(K, f, dofs_p, g):
    """Independent check: d[dofs_p] = g, then K_ff d_f = f_f - K_fp g; R = K d - f."""
    K = K.toarray()
    n = K.shape[0]
    dofs_f = np.setdiff1d(np.arange(n), dofs_p)
    d = np.zeros(n)
    d[dofs_p] = g
    d[dofs_f] = np.linalg.solve(K[np.ix_(dofs_f, dofs_f)],
                                f[dofs_f] - K[np.ix_(dofs_f, dofs_p)] @ g)
    return d, K @ d - f


def test_consistent_load_sums_to_total():
    K, f = assemble(MESHES[2], EA_BAR, Q_BAR)
    assert np.sum(f) == pytest.approx(Q_BAR * L_BAR, rel=1e-14)


def test_unsupported_stiffness_has_one_rigid_body_mode():
    K, f = assemble(MESHES[2], EA_BAR, Q_BAR)
    K = K.toarray()
    np.testing.assert_allclose(K, K.T)
    # A uniform translation produces no force, and it is the only such mode.
    assert np.max(np.abs(K @ np.ones(K.shape[0]))) < 1e-12 * np.max(np.abs(K))
    eig = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(eig) < 1e-10 * np.max(eig)) == 1


@pytest.mark.parametrize("x", MESHES)
def test_nodal_values_are_exact(x):
    K, f, C, g, d, lam, R = solve_bar(x)
    u_ex = exact_u(x, L_BAR, EA_BAR, Q_BAR, U_END)
    np.testing.assert_allclose(d, u_ex, rtol=0, atol=1e-13 * np.max(np.abs(u_ex)))


@pytest.mark.parametrize("x", MESHES)
def test_constraints_are_satisfied(x):
    K, f, C, g, d, lam, R = solve_bar(x)
    np.testing.assert_allclose(C @ d, g, rtol=0, atol=1e-14 * np.max(np.abs(g)))


@pytest.mark.parametrize("x", MESHES)
def test_minus_lambda_equals_exact_support_forces(x):
    # R = -C^T lam and each row of C picks one DOF, so the support force there is -lam.
    K, f, C, g, d, lam, R = solve_bar(x)
    exact = exact_support_forces(L_BAR, EA_BAR, Q_BAR, U_END)
    np.testing.assert_allclose(-lam, exact, rtol=1e-12)
    # With the V1 data: -700 N at x = 0 and -300 N at x = L (sum -q L = -1000 N).
    np.testing.assert_allclose(exact, [-700.0, -300.0], rtol=1e-12)


@pytest.mark.parametrize("x", MESHES)
def test_reactions_equal_K_d_minus_f(x):
    # First block row: K d + C^T lam = f, so K d - f = -C^T lam.
    K, f, C, g, d, lam, R = solve_bar(x)
    np.testing.assert_allclose(K @ d - f, R, rtol=0, atol=1e-9 * np.max(np.abs(R)))


@pytest.mark.parametrize("x", MESHES)
def test_lagrange_matches_elimination(x):
    K, f, C, g, d, lam, R = solve_bar(x)
    d_e, R_e = solve_by_elimination(K, f, np.array([0, len(x) - 1]), g)
    np.testing.assert_allclose(d, d_e, rtol=0, atol=1e-13 * np.max(np.abs(d_e)))
    np.testing.assert_allclose(R, R_e, rtol=0, atol=1e-9 * np.max(np.abs(R_e)))
