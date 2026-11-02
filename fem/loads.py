"""Consistent nodal loads of a uniform edge traction.

Along a straight element edge of length h the Q4 shape functions are linear,
N1 = (1 - xi)/2 and N2 = (1 + xi)/2, so for a uniform traction tbar (Pa)
    f_i = t int N_i tbar ds = t h tbar / 2   at each end node of the segment.
Summed over the segments of a uniformly divided edge: t h tbar at every interior
node and t h tbar / 2 at the two end nodes. For the cantilever's end load
tbar_y = P / (t D) on ny segments of length D / ny this is P / ny at interior
nodes and P / (2 ny) at the corners, adding up to P.
"""
import numpy as np


def edge_traction_load(nodes, edge, traction, t):
    """Global load vector (2 N,) of a uniform traction [tx, ty] on the edge through
    the node numbers `edge` (in order along the edge)."""
    f = np.zeros(2 * len(nodes))
    for n1, n2 in zip(edge[:-1], edge[1:]):
        h = np.linalg.norm(nodes[n2] - nodes[n1])
        for n in (n1, n2):
            f[2 * n] += t * h / 2.0 * traction[0]
            f[2 * n + 1] += t * h / 2.0 * traction[1]
    return f
