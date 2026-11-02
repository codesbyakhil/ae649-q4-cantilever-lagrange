"""V4 (nx x 1 strip in pure bending) and V6 (a)-(c) for the clamped cantilever.

Analytical counterparts: Cook Eq. 3.6-11 for the strip; statics, Clapeyron's
theorem and the minimum-potential-energy argument for the cantilever.
Elimination appears here only, as an independent check.
"""
import numpy as np
import pytest

from checks.beam_theory import pure_bending_end_rotation, second_moment
from checks.closed_form import cook_bending_ratio
from fem.assemble import assemble_stiffness
from fem.cantilever import (DEPTH, E, LENGTH, NU, P, THICKNESS, clamp_dofs,
                            solve_cantilever)
from fem.lagrange import augmented_matrix, constraint_matrix, solve_lagrange
from fem.material import plane_stress_D
from fem.mesh import edge_nodes, structured_mesh
from fem.post import force_resultants, observed_order, richardson

MESHES = [(3, 1), (6, 2), (12, 4), (24, 8)]


@pytest.fixture(scope="module")
def results():
    return {mesh: solve_cantilever(*mesh) for mesh in MESHES}


# ---------------------------------------------------------------- V4: strip in pure bending
@pytest.mark.parametrize("nx", [1, 3, 6, 12])
def test_strip_under_end_couple_matches_cook(nx):
    # nx x 1 strip clamped at x = 0; couple of forces +F (top) and -F (bottom) on u at
    # x = L, moment M = F D. Every element carries the same moment, so each behaves
    # like Cook's single element with a = L / (2 nx), b = D / 2.
    nodes, elems = structured_mesh(LENGTH, DEPTH, nx, 1)
    K = assemble_stiffness(nodes, elems, plane_stress_D(E, NU), THICKNESS)
    bottom, top = edge_nodes(nodes, LENGTH)
    F = 100.0
    f = np.zeros(2 * len(nodes))
    f[2 * top], f[2 * bottom] = F, -F
    clamped = clamp_dofs(nodes)
    d, lam, R = solve_lagrange(K, f, constraint_matrix(clamped, len(f)), np.zeros(len(clamped)))
    theta_fe = (d[2 * top] - d[2 * bottom]) / DEPTH
    theta_exact = pure_bending_end_rotation(F * DEPTH, LENGTH, E, second_moment(THICKNESS, DEPTH))
    a, b = LENGTH / (2 * nx), DEPTH / 2
    assert theta_fe / theta_exact == pytest.approx(cook_bending_ratio(a, b, NU), rel=1e-11)
    if nx == 3:   # validation item V4: 0.1149922
        assert cook_bending_ratio(a, b, NU) == pytest.approx(0.1149922, abs=1e-7)


# ---------------------------------------------------------------- loads and clamp
@pytest.mark.parametrize("mesh", MESHES)
def test_end_load_is_consistent_and_sums_to_P(results, mesh):
    res = results[mesh]
    ny = mesh[1]
    right = edge_nodes(res["nodes"], LENGTH)
    f = res["f"]
    expected = np.full(ny + 1, P / ny)
    expected[[0, -1]] = P / (2 * ny)
    np.testing.assert_allclose(f[2 * right + 1], expected, rtol=1e-14)
    assert np.sum(f) == pytest.approx(P, rel=1e-14)
    assert np.count_nonzero(f) == ny + 1


@pytest.mark.parametrize("mesh", MESHES)
def test_clamp_is_satisfied(results, mesh):
    res = results[mesh]
    assert len(res["clamped"]) == 2 * (mesh[1] + 1)
    assert np.max(np.abs(res["d"][res["clamped"]])) < 1e-12 * np.max(np.abs(res["d"]))


# ---------------------------------------------------------------- V6 (a): statics
@pytest.mark.parametrize("mesh", MESHES)
def test_reactions_balance_the_load(results, mesh):
    res = results[mesh]
    Rx, Ry, MR = force_resultants(res["nodes"], res["R"])
    fx, fy, Mf = force_resultants(res["nodes"], res["f"])
    # Exact in theory; in floating point the error is round-off amplified by the
    # condition number (cond(K_ff) = 9e6 on 24 x 8, so cond * eps = 2e-9).
    assert Ry == pytest.approx(-P, rel=1e-8)            # +10 N
    assert abs(Rx) < 1e-8 * abs(P)
    assert abs(MR + Mf) < 1e-8 * abs(P) * LENGTH        # net moment about the origin
    assert Mf == pytest.approx(P * LENGTH, rel=1e-14)   # load moment about the origin: P L
    assert MR == pytest.approx(-P * LENGTH, rel=1e-8)   # reaction moment: -P L = +20 N m


@pytest.mark.parametrize("mesh", MESHES)
def test_reactions_are_K_d_minus_f_and_minus_lambda(results, mesh):
    res = results[mesh]
    K, d, f, R, lam, clamped = (res[k] for k in ("K", "d", "f", "R", "lam", "clamped"))
    scale = np.max(np.abs(K @ d))                       # size of the internal nodal forces
    np.testing.assert_allclose(K @ d - f, R, rtol=0, atol=1e-11 * scale)
    free = np.setdiff1d(np.arange(len(d)), clamped)
    assert np.max(np.abs(R[free])) == 0.0               # -C^T lam is zero off the clamp
    np.testing.assert_allclose(lam, -R[clamped], rtol=1e-12)


# ---------------------------------------------------------------- V6 (b): Clapeyron
@pytest.mark.parametrize("mesh", MESHES)
def test_clapeyron_dKd_equals_fd(results, mesh):
    # K d + C^T lam = f and C d = 0  =>  d^T K d = f^T d (the reactions do no work).
    res = results[mesh]
    d, K, f = res["d"], res["K"], res["f"]
    assert d @ (K @ d) == pytest.approx(f @ d, rel=1e-9)


# ---------------------------------------------------------------- V6 (c): minimum potential energy
def test_compliance_increases_under_nested_refinement(results):
    # Each mesh subdivides the previous one, so its displacement space contains the
    # previous one. Pi_min = -1/2 f^T d can only decrease, so f^T d can only increase.
    compliance = [results[mesh]["f"] @ results[mesh]["d"] for mesh in MESHES]
    assert all(c > 0 for c in compliance)
    assert all(c2 > c1 for c1, c2 in zip(compliance[:-1], compliance[1:]))


# ---------------------------------------------------------------- the Lagrange system itself
@pytest.mark.parametrize("mesh", [(3, 1), (6, 2), (12, 4)])
def test_lagrange_matches_elimination(results, mesh):
    res = results[mesh]
    K, f, d, R, clamped = (res[k] for k in ("K", "f", "d", "R", "clamped"))
    Kd = K.toarray()
    free = np.setdiff1d(np.arange(len(d)), clamped)
    d_e = np.zeros(len(d))
    d_e[free] = np.linalg.solve(Kd[np.ix_(free, free)], f[free])     # g = 0
    # Two different solution paths agree to round-off times cond(K_ff)
    # (cond = 1.9e6 on 12 x 4, cond * eps = 4e-10).
    np.testing.assert_allclose(d, d_e, rtol=0, atol=1e-9 * np.max(np.abs(d_e)))
    np.testing.assert_allclose(R, Kd @ d_e - f, rtol=0, atol=1e-8 * abs(P))


def test_unsupported_structure_has_three_rigid_body_modes(results):
    eig = np.linalg.eigvalsh(results[(6, 2)]["K"].toarray())
    assert np.sum(np.abs(eig) < 1e-10 * np.max(eig)) == 3


@pytest.mark.parametrize("mesh", [(3, 1), (6, 2), (12, 4)])
def test_augmented_matrix_is_indefinite_and_scaling_fixes_its_conditioning(results, mesh):
    # K is positive definite on the null space of C and C has full row rank, so
    # [[K, C^T], [C, 0]] is nonsingular with exactly n positive and m negative
    # eigenvalues: indefinite, hence no Cholesky. Scaling is the congruence
    # S A S, S = diag(I, s I), which keeps the inertia (Sylvester's law).
    # Unscaled, cond_2 is about 1e22 and the tiny eigenvalues are not even
    # computable in float64 (on 12 x 4 eigvalsh gets their signs wrong), so the
    # inertia is checked on the scaled matrix, whose cond_2 equals cond(K_ff).
    res = results[mesh]
    K, C, clamped = res["K"], res["C"], res["clamped"]
    n, m = K.shape[0], C.shape[0]
    A_s = augmented_matrix(K, C, K.diagonal().max()).toarray()
    eig = np.linalg.eigvalsh(A_s)
    assert np.sum(eig > 0) == n and np.sum(eig < 0) == m
    free = np.setdiff1d(np.arange(n), clamped)
    cond_Kff = np.linalg.cond(K.toarray()[np.ix_(free, free)])
    assert np.linalg.cond(A_s) == pytest.approx(cond_Kff, rel=0.05)
    # float64 SVD agrees with a 50-digit SVD here (studies/exact_field.py).
    assert np.linalg.cond(augmented_matrix(K, C, 1.0).toarray()) > 1e21


# ---------------------------------------------------------------- V6 (e) formulas
def test_observed_order_and_richardson_recover_a_known_sequence():
    phi_inf, c, q = -4.75, 3.2, 2.0
    phi = [phi_inf + c * (0.1 / 2**i) ** q for i in range(3)]
    assert observed_order(*phi) == pytest.approx(q, rel=1e-12)
    assert richardson(phi[1], phi[2], q) == pytest.approx(phi_inf, rel=1e-12)
