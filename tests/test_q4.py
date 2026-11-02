"""V2 (Q4 element stiffness) and V4 for a single element (Q4 in pure bending).

Analytical counterparts: exact integration of k for a rectangle 2a x 2b
(checks/closed_form.py) and Cook Eq. 3.6-11 for the bending error.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from checks.beam_theory import pure_bending_end_rotation, second_moment
from checks.closed_form import cook_bending_ratio, q4_k11, q4_k12, q4_rectangle_stiffness
from fem.lagrange import constraint_matrix, solve_lagrange
from fem.material import plane_stress_D
from fem.q4 import (R_NODE, S_NODE, element_stiffness, gauss_rule, jacobian,
                    shape_derivatives, shape_functions, strain_displacement)

E, NU, T = 200e9, 0.3, 0.1
D = plane_stress_D(E, NU)


def rectangle(a, b, x0=0.0, y0=0.0):
    """Nodal coordinates (4 x 2) of a 2a x 2b rectangle with corner node 1 at (x0, y0)."""
    return np.array([[x0, y0], [x0 + 2 * a, y0], [x0 + 2 * a, y0 + 2 * b], [x0, y0 + 2 * b]])


# A general (non-parallelogram) convex quadrilateral, counter-clockwise.
DISTORTED = np.array([[0.0, 0.0], [1.3, 0.2], [1.1, 0.9], [0.2, 1.2]])


def rigid_body_modes(xy):
    """Columns: x translation, y translation, small rotation (u = -y, v = x)."""
    modes = np.zeros((8, 3))
    modes[0::2, 0] = 1.0
    modes[1::2, 1] = 1.0
    modes[0::2, 2] = -xy[:, 1]
    modes[1::2, 2] = xy[:, 0]
    return modes


# ---------------------------------------------------------------- material
def test_plane_stress_D_gives_uniaxial_stress_and_shear_modulus():
    # Uniaxial stress sigma0: eps = [sigma0/E, -nu sigma0/E, 0] must give [sigma0, 0, 0].
    sigma0 = 1.0e6
    np.testing.assert_allclose(D @ np.array([sigma0 / E, -NU * sigma0 / E, 0.0]),
                               [sigma0, 0.0, 0.0], atol=1e-9 * sigma0)
    assert D[2, 2] == pytest.approx(E / (2.0 * (1.0 + NU)), rel=1e-15)
    np.testing.assert_array_equal(D, D.T)


# ---------------------------------------------------------------- shape functions and mapping
def test_shape_functions_partition_of_unity_and_kronecker_delta():
    for s, r in [(-0.3, 0.7), (0.0, 0.0), (0.9, -0.4)]:
        assert np.sum(shape_functions(s, r)) == pytest.approx(1.0, abs=1e-15)
        assert np.sum(shape_derivatives(s, r), axis=1) == pytest.approx([0.0, 0.0], abs=1e-15)
    for j in range(4):
        np.testing.assert_allclose(shape_functions(S_NODE[j], R_NODE[j]), np.eye(4)[j], atol=1e-15)


def test_shape_derivatives_match_finite_differences():
    s, r, h = 0.37, -0.61, 1e-6
    fd_s = (shape_functions(s + h, r) - shape_functions(s - h, r)) / (2 * h)
    fd_r = (shape_functions(s, r + h) - shape_functions(s, r - h)) / (2 * h)
    np.testing.assert_allclose(shape_derivatives(s, r), np.array([fd_s, fd_r]), atol=1e-9)


@pytest.mark.parametrize("hx, hy", [(1.0, 1.0), (2.0 / 3.0, 0.15), (0.0625, 0.0234375)])
def test_rectangle_jacobian_is_diagonal_with_det_area_over_four(hx, hy):
    xy = rectangle(hx / 2, hy / 2, x0=0.3, y0=-0.2)
    points, _ = gauss_rule(2)
    for s in points:
        for r in points:
            np.testing.assert_allclose(jacobian(s, r, xy), np.diag([hx / 2, hy / 2]), atol=1e-15)
            B, detJ = strain_displacement(s, r, xy)
            assert detJ == pytest.approx(hx * hy / 4, rel=1e-14)


def test_linear_field_gives_constant_strain_on_a_distorted_element():
    # u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y  =>  eps = [a1, b2, a2 + b1] exactly.
    # Basis of the patch test (V3): Q4 contains every linear field, on any shape.
    a0, a1, a2, b0, b1, b2 = 1e-3, 2e-4, -3e-4, -5e-4, 4e-4, 1e-4
    d = np.zeros(8)
    d[0::2] = a0 + a1 * DISTORTED[:, 0] + a2 * DISTORTED[:, 1]
    d[1::2] = b0 + b1 * DISTORTED[:, 0] + b2 * DISTORTED[:, 1]
    for s, r in [(-0.577, 0.577), (0.2, -0.9), (0.95, 0.95)]:
        B, detJ = strain_displacement(s, r, DISTORTED)
        np.testing.assert_allclose(B @ d, [a1, b2, a2 + b1], rtol=1e-12, atol=1e-18)


def test_inverted_element_is_rejected():
    with pytest.raises(ValueError):
        strain_displacement(0.0, 0.0, rectangle(0.5, 0.5)[[0, 3, 2, 1]])  # clockwise order


# ---------------------------------------------------------------- V2: stiffness
def test_stiffness_matches_exact_integration_unit_square():
    a = b = 0.5
    k = element_stiffness(rectangle(a, b), D, T)
    assert k[0, 0] == pytest.approx(q4_k11(a, b, E, NU, T), rel=1e-14)
    assert k[0, 1] == pytest.approx(q4_k12(E, NU, T), rel=1e-14)
    # Value quoted in validation item V2: k11 = 9.8901098901e9 N/m.
    assert q4_k11(a, b, E, NU, T) == pytest.approx(9.8901098901e9, rel=1e-10)


@pytest.mark.parametrize("a, b", [(0.5, 0.5), (1.0, 0.075), (1.0 / 3.0, 0.075), (0.2, 0.7)])
def test_every_stiffness_entry_matches_exact_integration(a, b):
    k = element_stiffness(rectangle(a, b), D, T)
    k_exact = q4_rectangle_stiffness(a, b, E, NU, T)
    np.testing.assert_allclose(k, k_exact, rtol=0, atol=1e-14 * np.max(np.abs(k_exact)))
    assert k[0, 0] == pytest.approx(q4_k11(a, b, E, NU, T), rel=1e-14)
    assert k[0, 1] == pytest.approx(q4_k12(E, NU, T), rel=1e-14)


def test_two_by_two_gauss_is_exact_for_rectangles_but_not_for_general_quads():
    # For a rectangle the integrand is at most quadratic in s and in r, so 2 x 2
    # (exact to degree 3) and 3 x 3 Gauss agree. For a general quadrilateral
    # det J is not constant, the integrand is rational, and the rules differ.
    xy = rectangle(0.3, 0.1)
    k2, k3 = element_stiffness(xy, D, T, 2), element_stiffness(xy, D, T, 3)
    np.testing.assert_allclose(k2, k3, rtol=0, atol=1e-13 * np.max(np.abs(k3)))
    k2, k3 = element_stiffness(DISTORTED, D, T, 2), element_stiffness(DISTORTED, D, T, 3)
    assert np.max(np.abs(k2 - k3)) > 1e-6 * np.max(np.abs(k3))


@pytest.mark.parametrize("xy", [rectangle(0.5, 0.5), rectangle(1.0, 0.075), DISTORTED])
def test_stiffness_is_symmetric_with_exactly_three_rigid_body_modes(xy):
    k = element_stiffness(xy, D, T)
    np.testing.assert_allclose(k, k.T, rtol=0, atol=1e-15 * np.max(np.abs(k)))
    # Rigid-body motions produce no strain, hence no nodal force.
    assert np.max(np.abs(k @ rigid_body_modes(xy))) < 1e-12 * np.max(np.abs(k))
    eig = np.linalg.eigvalsh(k)
    assert np.sum(np.abs(eig) < 1e-10 * np.max(eig)) == 3
    assert np.min(eig) > -1e-10 * np.max(eig)          # positive semi-definite


def test_stack_of_elements_gives_the_same_matrices_as_one_at_a_time():
    # fem/q4.py works on one element (4 x 2) or a stack (n_el x 4 x 2) with the same formulas.
    stack = np.array([rectangle(0.5, 0.5), rectangle(1.0, 0.075), DISTORTED])
    k_stack = element_stiffness(stack, D, T)
    for e in range(len(stack)):
        np.testing.assert_array_equal(k_stack[e], element_stiffness(stack[e], D, T))
    B_stack, detJ_stack = strain_displacement(0.3, -0.2, stack)
    for e in range(len(stack)):
        B, detJ = strain_displacement(0.3, -0.2, stack[e])
        np.testing.assert_array_equal(B_stack[e], B)
        assert detJ_stack[e] == detJ


# ---------------------------------------------------------------- V4: one element in pure bending
@pytest.mark.parametrize("a_over_b", [1.0, 2.0, 40.0 / 9.0])
def test_bending_mode_energy_ratio_equals_cook(a_over_b):
    # Cook's argument: in the bending mode (Fig. 3.6-2b, Eq. 3.6-10) each node moves
    # horizontally by -theta b s_i r_i / 2. The exact block (Eq. 3.6-9) stores
    # U_b = E t theta^2 b^3 / (6a). At equal moment theta_el / theta_b = U_b / U_el.
    b = 0.075
    a = a_over_b * b
    k = element_stiffness(rectangle(a, b, x0=-a, y0=-b), D, T)
    theta = 1e-4
    d = np.zeros(8)
    d[0::2] = -theta * b * S_NODE * R_NODE / 2.0
    U_el = 0.5 * d @ k @ d
    U_b = E * T * theta**2 * b**3 / (6.0 * a)
    assert U_b / U_el == pytest.approx(cook_bending_ratio(a, b, NU), rel=1e-12)


@pytest.mark.parametrize("a_over_b", [1.0, 2.0, 40.0 / 9.0])
def test_clamped_element_under_end_couple_matches_cook(a_over_b):
    # One element, nodes 1 and 4 clamped by Lagrange multipliers, couple of forces
    # -F (node 2, bottom) and +F (node 3, top) on u: M = 2 F b.
    b = 0.075
    a = a_over_b * b
    k = element_stiffness(rectangle(a, b, y0=-b), D, T)
    C = constraint_matrix(np.array([0, 1, 6, 7]), 8)          # u1, v1, u4, v4
    F = 100.0
    f = np.zeros(8)
    f[2], f[4] = -F, F
    d, lam, R = solve_lagrange(sp.csr_matrix(k), f, C, np.zeros(4))
    theta_fe = (d[4] - d[2]) / (2 * b)                       # rotation of the loaded edge
    theta_exact = pure_bending_end_rotation(2 * F * b, 2 * a, E, second_moment(T, 2 * b))
    assert theta_fe / theta_exact == pytest.approx(cook_bending_ratio(a, b, NU), rel=1e-12)
    # validation item V4 quotes 0.6740741 for a/b = 1.
    if a_over_b == 1.0:
        assert cook_bending_ratio(a, b, NU) == pytest.approx(0.6740741, abs=1e-7)
