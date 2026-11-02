"""V3: uniform tension, an exact linear elasticity field (patch test).

Beam of README.md, traction sigma0 in x on the end x = L, u = 0 at every
left-edge node and v = 0 at the left mid-height node, all by Lagrange multipliers.
Exact solution: u = sigma0 x / E, v = -nu sigma0 y / E (checks/closed_form.py).
It is linear and Q4 contains every linear field, so the FE solution must be exact,
also on a distorted mesh. "Exact" means round-off: on 12 x 4 the scaled system has
cond_2 = 1.9e6, so errors up to cond * eps = 4e-10 are round-off; the tolerance is 1e-10.
"""
import numpy as np
import pytest

from checks.closed_form import uniform_tension_displacement
from fem.assemble import assemble_stiffness
from fem.cantilever import DEPTH, E, LENGTH, NU, THICKNESS
from fem.lagrange import constraint_matrix, solve_lagrange
from fem.loads import edge_traction_load
from fem.material import plane_stress_D
from fem.mesh import distort_interior, edge_nodes, structured_mesh
from fem.post import force_resultants, gauss_point_stresses

SIGMA0 = 1.0e6   # Pa


def solve_tension(nx, ny, distort, rigid=(0.0, 0.0, 0.0)):
    """Solve uniform tension. rigid = (u0, v0, theta) adds a rigid motion to the
    prescribed values: u = u0 - theta y on the left edge and v = v0 at (0, 0),
    so g is not zero (and not constant along the edge)."""
    nodes, elems = structured_mesh(LENGTH, DEPTH, nx, ny)
    if distort:
        nodes = distort_interior(nodes, LENGTH, DEPTH, nx, ny, fraction=0.2, seed=1)
    D = plane_stress_D(E, NU)
    K = assemble_stiffness(nodes, elems, D, THICKNESS)
    f = edge_traction_load(nodes, edge_nodes(nodes, LENGTH), [SIGMA0, 0.0], THICKNESS)
    left = edge_nodes(nodes, 0.0)
    mid = left[np.argmin(np.abs(nodes[left, 1]))]
    assert nodes[mid, 1] == 0.0
    dofs = np.concatenate([2 * left, [2 * mid + 1]])
    u0, v0, theta = rigid
    g = np.concatenate([u0 - theta * nodes[left, 1], [v0]])
    d, lam, R = solve_lagrange(K, f, constraint_matrix(dofs, 2 * len(nodes)), g)
    return nodes, elems, D, f, left, mid, dofs, d, lam, R


def exact_nodal(nodes, rigid=(0.0, 0.0, 0.0)):
    u, v = uniform_tension_displacement(nodes[:, 0], nodes[:, 1], SIGMA0, E, NU)
    u0, v0, theta = rigid
    d = np.zeros(2 * len(nodes))
    d[0::2] = u + u0 - theta * nodes[:, 1]
    d[1::2] = v + v0 + theta * nodes[:, 0]
    return d


@pytest.mark.parametrize("distort", [False, True])
def test_nodal_displacements_are_exact(distort):
    nodes, elems, D, f, left, mid, dofs, d, lam, R = solve_tension(12, 4, distort)
    d_ex = exact_nodal(nodes)
    assert np.max(np.abs(d - d_ex)) / np.max(np.abs(d_ex)) < 1e-10


@pytest.mark.parametrize("distort", [False, True])
def test_gauss_point_stresses_are_uniform(distort):
    nodes, elems, D, f, left, mid, dofs, d, lam, R = solve_tension(12, 4, distort)
    xy, sigma = gauss_point_stresses(nodes, elems, d, D)
    np.testing.assert_allclose(sigma[:, :, 0], SIGMA0, rtol=1e-10)
    assert np.max(np.abs(sigma[:, :, 1:])) < 1e-10 * SIGMA0


def test_distorted_mesh_is_really_distorted():
    nodes, elems = structured_mesh(LENGTH, DEPTH, 12, 4)
    moved = distort_interior(nodes, LENGTH, DEPTH, 12, 4, fraction=0.2, seed=1)
    shift = np.abs(moved - nodes) / np.array([LENGTH / 12, DEPTH / 4])
    assert np.max(shift) > 0.15                                   # interior nodes moved
    boundary = np.isin(np.arange(len(nodes)),
                       np.concatenate([edge_nodes(nodes, 0.0), edge_nodes(nodes, LENGTH)]))
    boundary |= np.isclose(np.abs(nodes[:, 1]), DEPTH / 2)
    assert np.max(shift[boundary]) == 0.0                         # boundary nodes did not


@pytest.mark.parametrize("distort", [False, True])
def test_reactions_equal_consistent_loads_with_opposite_sign(distort):
    nodes, elems, D, f, left, mid, dofs, d, lam, R = solve_tension(12, 4, distort)
    # The support carries sigma0 on the left face: consistent loads -sigma0 t h / 2 at
    # the corners and -sigma0 t h at interior nodes, in x. No force in y at the mid node.
    f_left = edge_traction_load(nodes, left, [-SIGMA0, 0.0], THICKNESS)
    np.testing.assert_allclose(R[2 * left], f_left[2 * left], rtol=1e-10)
    assert abs(R[2 * mid + 1]) < 1e-10 * SIGMA0 * THICKNESS * DEPTH
    np.testing.assert_allclose(-lam, R[dofs], rtol=1e-12, atol=1e-12 * SIGMA0 * THICKNESS * DEPTH)
    Fx, Fy, M = force_resultants(nodes, R + f)                    # overall equilibrium
    assert max(abs(Fx), abs(Fy), abs(M)) < 1e-9 * SIGMA0 * THICKNESS * DEPTH


@pytest.mark.parametrize("distort", [False, True])
def test_nonzero_prescribed_values(distort):
    # C d = g with g != 0: the exact field plus a rigid motion is reproduced exactly,
    # and the reactions do not change (a rigid motion creates no stress).
    rigid = (2e-6, -1e-6, 3e-6)
    nodes, elems, D, f, left, mid, dofs, d, lam, R = solve_tension(12, 4, distort, rigid)
    d_ex = exact_nodal(nodes, rigid)
    assert np.max(np.abs(d - d_ex)) / np.max(np.abs(d_ex)) < 1e-10
    R0 = solve_tension(12, 4, distort)[-1]
    np.testing.assert_allclose(R, R0, rtol=0, atol=1e-9 * SIGMA0 * THICKNESS * DEPTH)
