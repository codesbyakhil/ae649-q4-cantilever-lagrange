"""Structured nx x ny mesh of Q4 elements on [0, L] x [-D/2, D/2].

Nodes are numbered up each vertical grid line, left to right:
    node(i, j) = i (ny + 1) + j,    i = 0..nx along x,  j = 0..ny along y
Element (i, j), numbered e = i ny + j, has the corners
    [node(i, j), node(i+1, j), node(i+1, j+1), node(i, j+1)]
i.e. counter-clockwise from the bottom-left, matching the parent nodes
1(-1,-1), 2(1,-1), 3(1,1), 4(-1,1). Element column i spans x_i <= x <= x_(i+1).
"""
import numpy as np


def structured_mesh(L, depth, nx, ny):
    """Return nodes ((nx+1)(ny+1) x 2 coordinates) and elems (nx ny x 4 node numbers)."""
    x = np.linspace(0.0, L, nx + 1)
    y = np.linspace(-depth / 2.0, depth / 2.0, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")          # X[i, j] = x_i, Y[i, j] = y_j
    nodes = np.column_stack([X.ravel(), Y.ravel()])  # row i (ny + 1) + j = node(i, j)
    elems = np.zeros((nx * ny, 4), dtype=int)
    for i in range(nx):
        for j in range(ny):
            n1 = i * (ny + 1) + j
            elems[i * ny + j] = [n1, n1 + ny + 1, n1 + ny + 2, n1 + 1]
    return nodes, elems


def edge_nodes(nodes, x_value):
    """Node numbers on the vertical line x = x_value, sorted by y (bottom to top)."""
    tol = 1e-9 * (np.max(np.abs(nodes[:, 0])) + 1.0)
    ids = np.flatnonzero(np.abs(nodes[:, 0] - x_value) < tol)
    return ids[np.argsort(nodes[ids, 1])]


def distort_interior(nodes, L, depth, nx, ny, fraction=0.2, seed=0):
    """Copy of nodes with every interior node moved by a random offset of up to
    +-fraction of the grid spacing in x and in y. Boundary nodes do not move, so
    edges, loads and supports are unchanged.

    fraction <= 0.2 keeps every element convex (det J > 0): in grid-spacing units a
    corner is 1/sqrt(2) = 0.71 from the diagonal through its neighbours, and the
    corner and that diagonal can each move at most 0.2 sqrt(2) = 0.28 towards each other.
    """
    rng = np.random.default_rng(seed)
    hx, hy = L / nx, depth / ny
    moved = nodes.copy()
    for i in range(1, nx):
        for j in range(1, ny):
            n = i * (ny + 1) + j
            moved[n, 0] += fraction * hx * rng.uniform(-1.0, 1.0)
            moved[n, 1] += fraction * hy * rng.uniform(-1.0, 1.0)
    return moved
