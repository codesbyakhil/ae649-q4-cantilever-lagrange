"""Figure 5 (V5): the cantilever on a 1 x 1 mesh against the hand solution.

Hand solution (checks/closed_form.py, derived in checks/derivations.py):
    v_tip = 4 P L (1 + nu) [2 L^2 (1 - nu) + D^2] / {E t D [L^2 (1 - nu) + 2 D^2]}
    u_top = -u_bottom = -6 P L^2 (1 - nu^2) / {E t [L^2 (1 - nu) + 2 D^2]}
    Rx = -P L / D (bottom), +P L / D (top), Ry = -P/2 each  (statics + symmetry)
Run from the project root:  python -m studies.one_element
"""
import numpy as np

from checks.beam_theory import second_moment, tip_deflection_euler_bernoulli
from checks.closed_form import one_element_cantilever
from fem.cantilever import DEPTH, E, LENGTH, NU, P, THICKNESS, solve_cantilever
from studies.common import BLUE, INK, INK2, MUTED, ORANGE, plt, save_figure, setup_style, write_table


def main():
    setup_style()
    res = solve_cantilever(1, 1)
    d, R, lam, clamped = res["d"], res["R"], res["lam"], res["clamped"]
    hand = one_element_cantilever(P, LENGTH, DEPTH, E, NU, THICKNESS)
    # Mesh nodes: 0 = (0, -D/2), 1 = (0, D/2), 2 = (L, -D/2), 3 = (L, D/2).
    rows = [("v at (L, -D/2) (m)", d[5], hand["v_tip"]),
            ("v at (L, +D/2) (m)", d[7], hand["v_tip"]),
            ("u at (L, +D/2) (m)", d[6], hand["u_top"]),
            ("u at (L, -D/2) (m)", d[4], hand["u_bottom"]),
            ("Rx at (0, -D/2) (N)", R[0], hand["Rx_bottom"]),
            ("Ry at (0, -D/2) (N)", R[1], hand["Ry_bottom"]),
            ("Rx at (0, +D/2) (N)", R[2], hand["Rx_top"]),
            ("Ry at (0, +D/2) (N)", R[3], hand["Ry_top"])]
    lines = ["V5  one-element cantilever (1 x 1 mesh), clamped by Lagrange multipliers, P/2 at each end node",
             "",
             "quantity              | code              | hand solution     | relative difference",
             "-" * 82]
    for name, code, exact in rows:
        lines.append(f"{name:21s} | {code: .10e} | {exact: .10e} | {abs(code - exact) / abs(exact):.1e}")
    lines += ["", "Multipliers (one per clamped DOF, rows of C in the order u0, v0, u1, v1):",
              "  lambda      = " + "  ".join(f"{v: .6f}" for v in lam) + "  N",
              "  -R(clamped) = " + "  ".join(f"{v: .6f}" for v in -R[clamped]) + "  N",
              ""]
    v_eb = tip_deflection_euler_bernoulli(P, LENGTH, E, second_moment(THICKNESS, DEPTH))
    lines += [f"v_tip / Euler-Bernoulli = {hand['v_tip'] / v_eb:.5f}  (one element captures 1.4% of the deflection:",
              "Q4 shear locking, a/b = 13.3; compare Cook Eq. 3.6-11 in studies/bending_error.py)"]
    write_table("one_element", lines)

    nodes = res["nodes"]
    loop = [0, 2, 3, 1, 0]
    mag = 0.08 / abs(hand["v_tip"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(nodes[loop, 0], nodes[loop, 1], color=MUTED, linewidth=1.2, label="undeformed")
    moved = nodes + mag * np.column_stack([d[0::2], d[1::2]])
    ax.plot(moved[loop, 0], moved[loop, 1], color=BLUE, linewidth=2.0, label=f"deformed (x{mag:.1e})")
    ax.fill(nodes[[0, 1], 0] - 0.05, nodes[[0, 1], 1], color=MUTED)
    for n in (2, 3):     # end load P/2 at each loaded node (downward)
        ax.annotate("", xy=(moved[n, 0], moved[n, 1] - 0.005), xytext=(moved[n, 0], moved[n, 1] + 0.05),
                    arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.8})
    ax.text(moved[3, 0] + 0.03, moved[3, 1] + 0.03, "P/2 = -5 N\neach", color=INK2)
    for n, ry, rx in ((0, R[1], R[0]), (1, R[3], R[2])):
        ax.text(nodes[n, 0] + 0.04, nodes[n, 1] + (0.015 if n == 1 else -0.035),
                f"Rx = {rx:+.3f} N, Ry = {ry:+.3f} N", color=INK, fontsize=10)
    ax.text(1.0, 0.12, f"v_tip: code {hand['v_tip']:.6e} m = hand solution", ha="center", color=INK2)
    ax.set_xlim(-0.15, 2.45)
    ax.set_ylim(-0.2, 0.17)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("V5: one-element cantilever, reactions = -lambda")
    ax.legend(loc="lower left")
    save_figure(fig, "one_element")


if __name__ == "__main__":
    main()
