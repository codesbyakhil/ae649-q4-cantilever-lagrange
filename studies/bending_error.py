"""Figure 4 (V4): Q4 in pure bending against Cook Eq. 3.6-11.

theta_FE / theta_exact = (1 - nu^2) / [1 + (1 - nu)/2 (a/b)^2]  (line), against FE
results (markers) for single elements and for nx x 1 strips of the cantilever,
each clamped at x = 0 by Lagrange multipliers and loaded by an end couple.
Run from the project root:  python -m studies.bending_error
"""
import numpy as np
import scipy.sparse as sp

from checks.beam_theory import pure_bending_end_rotation, second_moment
from checks.closed_form import cook_bending_ratio
from fem.assemble import assemble_stiffness
from fem.cantilever import DEPTH, E, LENGTH, NU, THICKNESS, clamp_dofs
from fem.lagrange import constraint_matrix, solve_lagrange
from fem.material import plane_stress_D
from fem.mesh import edge_nodes, structured_mesh
from studies.common import BLUE, INK2, ORANGE, plt, save_figure, setup_style, write_table

F = 100.0   # N, couple of forces +F (top) and -F (bottom) on u at the loaded end


def end_couple_ratio(length, depth, nx):
    """theta_FE / theta_exact for an nx x 1 strip (length x depth), clamped at x = 0,
    couple +F / -F at the top / bottom node of x = length (moment M = F depth)."""
    nodes, elems = structured_mesh(length, depth, nx, 1)
    K = assemble_stiffness(nodes, elems, plane_stress_D(E, NU), THICKNESS)
    bottom, top = edge_nodes(nodes, length)
    f = np.zeros(2 * len(nodes))
    f[2 * top], f[2 * bottom] = F, -F
    clamped = clamp_dofs(nodes)
    d, lam, R = solve_lagrange(sp.csr_matrix(K), f, constraint_matrix(clamped, len(f)),
                               np.zeros(len(clamped)))
    theta_fe = (d[2 * top] - d[2 * bottom]) / depth
    theta_exact = pure_bending_end_rotation(F * depth, length, E, second_moment(THICKNESS, depth))
    return theta_fe / theta_exact


def main():
    setup_style()
    b = DEPTH / 2.0
    single = [0.5, 1.0, 2.0, 3.0, 40.0 / 9.0, 6.0]        # a/b of one element (length 2a)
    strips = [1, 2, 3, 6, 12, 24]                         # nx of the nx x 1 cantilever strip
    lines = ["V4  Q4 in pure bending: theta_FE / theta_exact vs Cook Eq. 3.6-11 (nu = 0.3)",
             "    a, b = element half-length and half-depth; one element (length 2a, depth D) or an",
             "    nx x 1 strip of the L = 2 m cantilever (a = L/(2 nx), b = D/2), clamped by Lagrange multipliers",
             "",
             "model          |    a/b  | FE ratio      | Cook Eq. 3.6-11 | FE - Cook",
             "-" * 74]
    fe_single, fe_strip = [], []
    for ab in single:
        r = end_couple_ratio(2 * ab * b, DEPTH, 1)
        fe_single.append(r)
        lines.append(f"one element    | {ab:7.4f} | {r:.10f} | {cook_bending_ratio(ab * b, b, NU):.10f}    | "
                     f"{r - cook_bending_ratio(ab * b, b, NU): .1e}")
    for nx in strips:
        a = LENGTH / (2 * nx)
        r = end_couple_ratio(LENGTH, DEPTH, nx)
        fe_strip.append(r)
        lines.append(f"{nx:2d} x 1 strip   | {a / b:7.4f} | {r:.10f} | {cook_bending_ratio(a, b, NU):.10f}    | "
                     f"{r - cook_bending_ratio(a, b, NU): .1e}")
    lines += ["", "Every element of a strip carries the same moment, so the strip behaves exactly like one",
              "element: the ratio depends only on a/b. In pure bending the 3 x 1 strip (a/b = 4.44)",
              "rotates only 11.5% as much as the exact solution (parasitic-shear locking), which is why",
              "the coarse meshes of the cantilever are so stiff (studies/tip_deflection.py)."]
    write_table("bending_error", lines)

    ab_line = np.linspace(0.0, 14.0, 400)
    fig, ax = plt.subplots()
    ax.plot(ab_line, cook_bending_ratio(ab_line, 1.0, NU), color=BLUE,
            label=r"Cook Eq. 3.6-11: $(1-\nu^2)\,/\,[1 + \frac{1-\nu}{2}(a/b)^2]$")
    ax.plot(single, fe_single, "o", color=ORANGE, markersize=8, label="FE: one element")
    strip_ab = [LENGTH / (2 * nx) / b for nx in strips]
    ax.plot(strip_ab, fe_strip, "s", color=INK2, markersize=7, markerfacecolor="none",
            markeredgewidth=1.6, label=r"FE: $n_x \times 1$ strip of the cantilever")
    for nx, x, y in zip(strips, strip_ab, fe_strip):
        if nx in (1, 3, 12):
            offset = (-62, 10) if nx == 1 else (10, -14)     # keep the 1 x 1 label inside the axes
            ax.annotate(f"{nx} x 1 strip", (x, y), textcoords="offset points", xytext=offset,
                        color=INK2, fontsize=9)
    ax.set_xlabel("element aspect ratio a/b")
    ax.set_ylabel(r"$\theta_{FE} / \theta_{exact}$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("V4: Q4 in pure bending (parasitic shear)")
    ax.legend(loc="upper right")
    save_figure(fig, "bending_error")


if __name__ == "__main__":
    main()
