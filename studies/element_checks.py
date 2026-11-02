"""Figure 2 (V2): Q4 element stiffness by 2 x 2 Gauss vs exact integration.

Table: k11 and k12 from the code vs the closed forms, the largest difference over
all 64 entries, 2 x 2 vs 3 x 3 Gauss, and the eigenvalues of k (3 zero).
Run from the project root:  python -m studies.element_checks
"""
import numpy as np

from checks.closed_form import q4_k11, q4_k12, q4_rectangle_stiffness
from fem.material import plane_stress_D
from fem.q4 import element_stiffness
from studies.common import BLUE, INK2, ORANGE, plt, save_figure, setup_style, write_table

E, NU, T = 200e9, 0.3, 0.1
CASES = [(0.5, 0.5, "unit square (validation item V2)"),
         (1.0 / 3.0, 0.075, "3 x 1 mesh element"),
         (2.0 / 3.0 / 2 ** 4 / 2, 0.075 / 2 ** 4, "48 x 16 mesh element"),
         (0.2, 0.7, "tall element")]
DISTORTED = np.array([[0.0, 0.0], [1.3, 0.2], [1.1, 0.9], [0.2, 1.2]])


def rectangle(a, b):
    return np.array([[0.0, 0.0], [2 * a, 0.0], [2 * a, 2 * b], [0.0, 2 * b]])


def main():
    setup_style()
    D = plane_stress_D(E, NU)
    lines = ["V2  Q4 stiffness, 2 x 2 Gauss vs exact integration (E = 200 GPa, nu = 0.3, t = 0.1 m)",
             "    k11 = E t/(1 - nu^2) [b/(3a) + (1 - nu) a/(6b)],  k12 = E t/(8(1 - nu))",
             "",
             "element (a, b in m)              | k11 code (N/m)    k11 formula      | k12 code (N/m)    k12 formula      "
             "| max|k - k_exact|/max|k| | max|k_2x2 - k_3x3|/max|k|",
             "-" * 160]
    for a, b, name in CASES:
        k = element_stiffness(rectangle(a, b), D, T)
        k_ex = q4_rectangle_stiffness(a, b, E, NU, T)
        k3 = element_stiffness(rectangle(a, b), D, T, n_gauss=3)
        lines.append(f"{name:21s} {a:.4f} {b:.4f} | {k[0, 0]:.10e} {q4_k11(a, b, E, NU, T):.10e} | "
                     f"{k[0, 1]:.10e} {q4_k12(E, NU, T):.10e} | {np.max(np.abs(k - k_ex)) / np.max(np.abs(k)):23.1e} | "
                     f"{np.max(np.abs(k - k3)) / np.max(np.abs(k)):.1e}")
    k2d = element_stiffness(DISTORTED, D, T)
    k3d = element_stiffness(DISTORTED, D, T, n_gauss=3)
    lines.append(f"{'general quadrilateral':31s} | (no closed form)                    | "
                 f"{'':35s}| {'':23s} | {np.max(np.abs(k2d - k3d)) / np.max(np.abs(k2d)):.1e}")
    lines += ["", "2 x 2 Gauss is exact for a rectangle (integrand at most quadratic in s and r, rule exact to",
              "degree 3), so 3 x 3 changes nothing; for a general quadrilateral det J varies and it does.",
              "A rectangle's k depends only on a/b (k11 above), so every mesh of the sequence, all with",
              "a/b = 4.44, has the same element matrix as the 3 x 1 mesh.", ""]

    k = element_stiffness(rectangle(0.5, 0.5), D, T)
    eig = np.linalg.eigvalsh(k)
    lines.append("Eigenvalues of k, unit square (N/m):")
    lines += [f"  {i + 1}: {lam: .6e}   lambda/lambda_max = {lam / eig[-1]: .1e}" for i, lam in enumerate(eig)]
    n_zero = int(np.sum(np.abs(eig) < 1e-10 * eig[-1]))
    lines.append(f"Zero eigenvalues: {n_zero} (2 translations + 1 rotation: zero strain, zero energy)")
    write_table("element_checks", lines)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ratio = np.maximum(np.abs(eig) / eig[-1], 1e-17)
    colors = [ORANGE if r < 1e-10 else BLUE for r in ratio]
    ax.bar(np.arange(1, 9), ratio, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e3)
    ax.set_xticks(np.arange(1, 9))
    ax.set_xlabel("eigenvalue number")
    ax.set_ylabel(r"$|\lambda_i| / \lambda_{max}$")
    ax.set_title("V2: eigenvalues of the Q4 stiffness (2 x 2 Gauss, unit square)")
    ax.text(2.0, 1e-13, "3 rigid-body modes\n(zero to round-off)", ha="center", color=INK2)
    ax.text(6.0, 4.0, "5 deformation modes", ha="center", color=INK2)
    save_figure(fig, "element_checks")


if __name__ == "__main__":
    main()
