"""The cantilever of README.md: problem data (SI units) and the solve.

x in [0, LENGTH], y in [-DEPTH/2, DEPTH/2]. Clamped at x = 0 (u = v = 0 at every
left-edge node, by Lagrange multipliers); total load P on v, spread over the end
face x = LENGTH as a uniform shear traction P / (t D).
"""
import numpy as np

from fem.assemble import assemble_stiffness
from fem.lagrange import constraint_matrix, solve_lagrange
from fem.loads import edge_traction_load
from fem.material import plane_stress_D
from fem.mesh import edge_nodes, structured_mesh

LENGTH = 2.0        # m    L
DEPTH = 0.15        # m    D in THEORY.md (here D is kept for the material matrix)
THICKNESS = 0.10    # m    t
E = 200e9           # Pa
NU = 0.30
P = -10.0           # N    total end load, negative = downward


def clamp_dofs(nodes, x_value=0.0):
    """u and v DOFs of every node on the line x = x_value: [2 n, 2 n + 1, ...]."""
    left = edge_nodes(nodes, x_value)
    return np.column_stack([2 * left, 2 * left + 1]).ravel()


def solve_cantilever(nx, ny):
    """Mesh, assemble, load and clamp the cantilever on an nx x ny mesh and solve
    [[K, C^T], [C, 0]] [d; lam] = [f; 0]. Returns a dict with nodes, elems, D, K, f,
    C, g, clamped (constrained DOFs, the row order of C), d, lam and R."""
    nodes, elems = structured_mesh(LENGTH, DEPTH, nx, ny)
    D = plane_stress_D(E, NU)
    K = assemble_stiffness(nodes, elems, D, THICKNESS)
    right = edge_nodes(nodes, LENGTH)
    f = edge_traction_load(nodes, right, [0.0, P / (THICKNESS * DEPTH)], THICKNESS)
    clamped = clamp_dofs(nodes)
    C = constraint_matrix(clamped, 2 * len(nodes))
    g = np.zeros(len(clamped))
    d, lam, R = solve_lagrange(K, f, C, g)
    return {"nodes": nodes, "elems": elems, "D": D, "K": K, "f": f, "C": C, "g": g,
            "clamped": clamped, "d": d, "lam": lam, "R": R}
