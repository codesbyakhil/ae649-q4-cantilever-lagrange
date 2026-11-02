"""Figure 3 (V3) and the scaling of the Lagrange system (THEORY.md section 5.5).

Uniform tension sigma0 on the beam, u = 0 on the left edge and v = 0 at the left
mid-height node (Lagrange multipliers). Exact field u = sigma0 x / E, v = -nu sigma0 y / E.
Nodal errors on a regular and a distorted 12 x 4 mesh, with and without scaling the
constraint rows by s = max diag K, and the condition numbers of the augmented matrix.
Run from the project root:  python -m studies.exact_field
"""
import mpmath
import numpy as np

from checks.closed_form import uniform_tension_displacement
from fem.assemble import assemble_stiffness
from fem.cantilever import DEPTH, E, LENGTH, NU, THICKNESS, solve_cantilever
from fem.lagrange import augmented_matrix, constraint_matrix, solve_lagrange
from fem.loads import edge_traction_load
from fem.material import plane_stress_D
from fem.mesh import distort_interior, edge_nodes, structured_mesh
from fem.post import gauss_point_stresses
from studies.common import BLUE, INK2, MUTED, ORANGE, plt, save_figure, setup_style, write_table

SIGMA0 = 1.0e6    # Pa
NX, NY = 12, 4


def tension_problem(distort):
    nodes, elems = structured_mesh(LENGTH, DEPTH, NX, NY)
    if distort:
        nodes = distort_interior(nodes, LENGTH, DEPTH, NX, NY, fraction=0.2, seed=1)
    D = plane_stress_D(E, NU)
    K = assemble_stiffness(nodes, elems, D, THICKNESS)
    f = edge_traction_load(nodes, edge_nodes(nodes, LENGTH), [SIGMA0, 0.0], THICKNESS)
    left = edge_nodes(nodes, 0.0)
    mid = left[np.argmin(np.abs(nodes[left, 1]))]
    dofs = np.concatenate([2 * left, [2 * mid + 1]])
    C = constraint_matrix(dofs, 2 * len(nodes))
    return nodes, elems, D, K, f, C, np.zeros(len(dofs)), left


def exact_nodal(nodes):
    u, v = uniform_tension_displacement(nodes[:, 0], nodes[:, 1], SIGMA0, E, NU)
    d = np.zeros(2 * len(nodes))
    d[0::2], d[1::2] = u, v
    return d


def main():
    setup_style()
    lines = [f"V3  uniform tension sigma0 = {SIGMA0:.0e} Pa, {NX} x {NY} mesh; exact u = sigma0 x/E, v = -nu sigma0 y/E",
             "",
             "mesh      | constraint rows | max nodal error / max|u| | max |sigma - [sigma0,0,0]| / sigma0 | "
             "max |R + consistent left-edge load| (N)",
             "-" * 128]
    errors = {}
    for distort in (False, True):
        nodes, elems, D, K, f, C, g, left = tension_problem(distort)
        d_ex = exact_nodal(nodes)
        f_left = edge_traction_load(nodes, left, [SIGMA0, 0.0], THICKNESS)
        for scale in (True, False):
            d, lam, R = solve_lagrange(K, f, C, g, scale=scale)
            err = np.max(np.abs(d - d_ex)) / np.max(np.abs(d_ex))
            xy, sigma = gauss_point_stresses(nodes, elems, d, D)
            s_err = np.max(np.abs(sigma - np.array([SIGMA0, 0.0, 0.0]))) / SIGMA0
            r_err = np.max(np.abs(R[2 * left] + f_left[2 * left]))
            errors[(distort, scale)] = err
            lines.append(f"{'distorted' if distort else 'regular':9s} | {'scaled by s' if scale else 'unscaled':15s} | "
                         f"{err:24.1e} | {s_err:36.1e} | {r_err:.1e}")
    lines += ["", "Reaction check: R at the u-DOFs of the left edge equals minus the consistent nodal loads",
              "of sigma0 on that edge (-sigma0 t h/2 at the corners, -sigma0 t h inside).", "",
              "Condition number cond_2 of [[K, s C^T], [s C, 0]] for the cantilever clamp",
              "(float64 SVD; checked with a 50-digit SVD (mpmath) on the two smallest meshes)",
              "",
              "mesh  | unscaled (s = 1)  50-digit check | scaled (s = max diag K)  50-digit check | cond_2(K_ff), eliminated system",
              "-" * 118]
    for nx, ny in [(3, 1), (6, 2), (12, 4), (24, 8)]:
        res = solve_cantilever(nx, ny)
        K, C, clamped = res["K"], res["C"], res["clamped"]
        row = f"{nx:2d}x{ny:<2d} |"
        for s in (1.0, K.diagonal().max()):
            A = augmented_matrix(K, C, s).toarray()
            c64 = np.linalg.cond(A)
            if len(A) <= 48:
                with mpmath.workdps(50):
                    sv = sorted(abs(v) for v in mpmath.svd_r(mpmath.matrix(A.tolist()), compute_uv=False))
                    c50 = f"{float(sv[-1] / sv[0]):.3e}"
            else:
                c50 = "-"
            row += f" {c64:16.3e} {c50:>14s} |"
        free = np.setdiff1d(np.arange(K.shape[0]), clamped)
        row += f" {np.linalg.cond(K.toarray()[np.ix_(free, free)]):.3e}"
        lines.append(row)
    lines += ["", "Inertia (numbers of positive / negative eigenvalues from float64 eigvalsh); theory: n / m",
              "mesh  |   n    m | unscaled  | scaled"]
    for nx, ny in [(3, 1), (6, 2), (12, 4)]:
        res = solve_cantilever(nx, ny)
        K, C = res["K"], res["C"]
        counts = []
        for s in (1.0, K.diagonal().max()):
            eig = np.linalg.eigvalsh(augmented_matrix(K, C, s).toarray())
            counts.append(f"{np.sum(eig > 0):4d} / {np.sum(eig < 0):<3d}")
        lines.append(f"{nx:2d}x{ny:<2d} | {K.shape[0]:4d} {C.shape[0]:4d} | {counts[0]} | {counts[1]}")
    lines += ["", "Unscaled, cond_2 ~ 1e22: K is ~1e10 N/m while C holds ones. With s = max diag K the",
              "augmented matrix is as well conditioned as the eliminated system K_ff (within 3%), and",
              "the exact field is recovered to round-off; unscaled, sparse LU loses about four digits.",
              "With cond_2 beyond 1/eps the smallest eigenvalues are no longer reliably computable in",
              "float64: on 12 x 4 even their signs come out wrong (inertia above), although the exact",
              "matrix has n positive and m negative eigenvalues (and the scaled one shows exactly that)."]
    write_table("exact_field", lines)

    # Figure: distorted mesh, FE deformed shape vs exact nodal positions; errors scaled/unscaled.
    nodes, elems, D, K, f, C, g, left = tension_problem(True)
    d, lam, R = solve_lagrange(K, f, C, g)
    d_ex = exact_nodal(nodes)
    mag = 0.15 * LENGTH / np.max(np.abs(d_ex))
    fig, ax1 = plt.subplots(figsize=(10.0, 2.9))
    for conn in elems:
        loop = np.append(conn, conn[0])
        ax1.plot(nodes[loop, 0], nodes[loop, 1], color=MUTED, linewidth=0.6)
        moved = nodes[loop] + mag * np.column_stack([d[2 * loop], d[2 * loop + 1]])
        ax1.plot(moved[:, 0], moved[:, 1], color=BLUE, linewidth=1.0)
    exact_pos = nodes + mag * np.column_stack([d_ex[0::2], d_ex[1::2]])
    ax1.plot(exact_pos[:, 0], exact_pos[:, 1], "o", color=ORANGE, markersize=5, markerfacecolor="none",
             label="exact field at the nodes")
    ax1.plot([], [], color=BLUE, linewidth=1.0, label=f"FE deformed (x{mag:.0f})")
    ax1.plot([], [], color=MUTED, linewidth=0.6, label="distorted mesh")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("V3: uniform tension on a distorted 12 x 4 mesh: FE nodes land on the exact field")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.85), ncol=3)
    save_figure(fig, "exact_field")

    fig, ax2 = plt.subplots(figsize=(5.6, 4.0))
    labels = ["regular mesh", "distorted mesh"]
    xpos = np.arange(2)
    ax2.bar(xpos - 0.18, [errors[(False, True)], errors[(True, True)]], width=0.34, color=BLUE, label="scaled by s")
    ax2.bar(xpos + 0.18, [errors[(False, False)], errors[(True, False)]], width=0.34, color=ORANGE, label="unscaled")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-17, 1e-2)
    ax2.set_xticks(xpos, labels)
    ax2.set_ylabel("max nodal error / max |u|")
    ax2.set_title("V3 exact field (12 x 4): scaling the constraint rows")
    ax2.legend(loc="upper left")
    ax2.grid(axis="x", visible=False)
    for i, key in enumerate([(False, True), (True, True)]):
        ax2.text(i - 0.18, errors[key] * 3, f"{errors[key]:.0e}", ha="center", color=INK2, fontsize=9)
    for i, key in enumerate([(False, False), (True, False)]):
        ax2.text(i + 0.18, errors[key] * 3, f"{errors[key]:.0e}", ha="center", color=INK2, fontsize=9)
    save_figure(fig, "exact_field_scaling")


if __name__ == "__main__":
    main()
