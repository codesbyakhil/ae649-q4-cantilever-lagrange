"""Figure 1 (V1): 1D bar, both ends prescribed, Lagrange multipliers.

FE nodal values on the exact parabola u(x) = q x (L - x) / (2 EA) + U x / L, and
the support forces -N(0), +N(L) against -lambda.
Run from the project root:  python -m studies.bar1d
"""
import numpy as np

from bar1d.bar import EA_BAR, L_BAR, Q_BAR, U_END, exact_support_forces, exact_u, solve_bar
from studies.common import BLUE, INK2, ORANGE, plt, save_figure, setup_style, write_table


def main():
    setup_style()
    exact_R = exact_support_forces(L_BAR, EA_BAR, Q_BAR, U_END)
    lines = ["V1  1D bar: EA u'' + q = 0, u(0) = 0, u(L) = U, solved with Lagrange multipliers",
             f"    L = {L_BAR} m, EA = {EA_BAR:.3e} N, q = {Q_BAR:.1f} N/m, U = {U_END:.1e} m",
             "",
             "elements | max |d - u_exact| / max|u| | -lambda_1 (N)  exact -N(0) | -lambda_2 (N)  exact N(L)",
             "-" * 96]
    for n_el in (2, 3, 8, 32):
        x = np.linspace(0.0, L_BAR, n_el + 1)
        K, f, C, g, d, lam, R = solve_bar(x)
        u = exact_u(x, L_BAR, EA_BAR, Q_BAR, U_END)
        err = np.max(np.abs(d - u)) / np.max(np.abs(u))
        lines.append(f"{n_el:8d} | {err:26.1e} | {-lam[0]:13.6f} {exact_R[0]:12.6f} | "
                     f"{-lam[1]:13.6f} {exact_R[1]:11.6f}")
    lines += ["", "Nodal values exact to round-off (linear elements, constant EA, consistent loads:",
              "Cook p. 47); -lambda equals the support forces from N = EA u'."]
    write_table("bar1d", lines)

    x = np.linspace(0.0, L_BAR, 9)
    K, f, C, g, d, lam, R = solve_bar(x)
    xf = np.linspace(0.0, L_BAR, 200)
    fig, ax = plt.subplots()
    ax.plot(xf, 1e6 * exact_u(xf, L_BAR, EA_BAR, Q_BAR, U_END), color=BLUE,
            label=r"exact  $u = qx(L-x)/2EA + Ux/L$")
    ax.plot(x, 1e6 * d, "o", color=ORANGE, markersize=8, label="FE nodes, 8 elements (Lagrange)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel(r"u ($\mu$m)")
    ax.set_title("V1: 1D bar, both ends prescribed by Lagrange multipliers")
    ax.text(0.04, 0.2, f"support force $-\\lambda_1$ = {-lam[0]:.1f} N", color=INK2)
    ax.text(0.98, 8.6, f"support force $-\\lambda_2$ = {-lam[1]:.1f} N", color=INK2, ha="right")
    ax.legend(loc="center right")
    save_figure(fig, "bar1d")


if __name__ == "__main__":
    main()
