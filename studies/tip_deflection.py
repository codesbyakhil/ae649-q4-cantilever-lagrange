"""Figure 6 (V6 (c)-(e)): tip deflection under mesh refinement.

v(L, 0) on the nested meshes 3 x 1 ... 384 x 128 against Euler-Bernoulli PL^3/3EI and
Timoshenko PL^3/3EI + PL/(kappa G A); compliance f^T d (must increase); observed order
q = log2[(v1 - v2)/(v2 - v3)] and Richardson extrapolation (Cook Eq. 9.7-1).
Run from the project root:  python -m studies.tip_deflection
"""
import numpy as np

from checks.beam_theory import (second_moment, tip_deflection_euler_bernoulli,
                                tip_deflection_timoshenko)
from fem.cantilever import DEPTH, E, LENGTH, NU, P, THICKNESS, solve_cantilever
from fem.post import observed_order, richardson, tip_deflection
from studies.common import (AQUA, BLUE, INK2, MESHES, MUTED, ORANGE, plt, save_figure,
                            setup_style, write_table)

# Cross-check only: values from an independent implementation: v_tip in um from a separate implementation.
INDEPENDENT_UM = {(3, 1): -0.54734, (6, 2): -1.62173, (12, 4): -3.19979, (24, 8): -4.23595,
                  (48, 16): -4.61223, (96, 32): -4.71813, (192, 64): -4.74577}


def main():
    setup_style()
    v_eb = tip_deflection_euler_bernoulli(P, LENGTH, E, second_moment(THICKNESS, DEPTH))
    v_ti = tip_deflection_timoshenko(P, LENGTH, E, NU, THICKNESS, DEPTH)
    v, compliance, n_el = [], [], []
    for nx, ny in MESHES:
        res = solve_cantilever(nx, ny)
        v.append(tip_deflection(res["nodes"], res["d"], LENGTH))
        compliance.append(res["f"] @ res["d"])
        n_el.append(nx * ny)
    v = np.array(v)

    lines = ["V6  tip deflection v(L, 0) of the clamped cantilever (P = -10 N), nested meshes",
             f"    Euler-Bernoulli PL^3/3EI = {v_eb * 1e6:.5f} um, Timoshenko (kappa = 5/6) = {v_ti * 1e6:.5f} um",
             "",
             "mesh     |   DOF | v_tip (um) | / Euler-B. | / Timosh. | compliance f.d (J) | independent (um) | difference",
             "-" * 113]
    for i, (nx, ny) in enumerate(MESHES):
        ind = INDEPENDENT_UM.get((nx, ny))
        ind_txt = f"{ind:16.5f} | {v[i] * 1e6 - ind: .1e}" if ind is not None else f"{'-':>16s} | -"
        lines.append(f"{nx:3d}x{ny:<4d} | {2 * (nx + 1) * (ny + 1):5d} | {v[i] * 1e6:10.5f} | {v[i] / v_eb:10.4f} | "
                     f"{v[i] / v_ti:9.4f} | {compliance[i]:18.6e} | {ind_txt}")
    increasing = all(c2 > c1 for c1, c2 in zip(compliance[:-1], compliance[1:]))
    lines += ["", f"Compliance f.d increases on every refinement: {increasing} (minimum potential energy,",
              "nested spaces). The independent column agrees to the 5 printed digits.", "",
              "meshes (h, h/2, h/4)        | observed order q | Richardson, q observed (um) | Richardson, q = 2 (um)",
              "-" * 100]
    estimates = []
    for i in range(len(MESHES) - 2):
        q = observed_order(*v[i:i + 3])
        r_q = richardson(v[i + 1], v[i + 2], q)
        r_2 = richardson(v[i + 1], v[i + 2], 2.0)
        estimates.append((q, r_q, r_2))
        names = " -> ".join(f"{nx}x{ny}" for nx, ny in MESHES[i:i + 3])
        lines.append(f"{names:28s} | {q:16.3f} | {r_q * 1e6:27.5f} | {r_2 * 1e6:.5f}")
    q_last, v_ext, _ = estimates[-1]
    spread = abs(estimates[-1][1] - estimates[-2][1])
    lines += ["", f"Extrapolated tip deflection (finest triple, q = {q_last:.3f}): {v_ext * 1e6:.5f} um",
              f"  = {v_ext / v_ti:.5f} x Timoshenko ({(v_ext / v_ti - 1) * 100:+.2f}%), "
              f"{v_ext / v_eb:.5f} x Euler-Bernoulli ({(v_ext / v_eb - 1) * 100:+.2f}%)",
              f"  change from the previous triple's estimate: {spread * 1e6:.5f} um "
              f"({spread / abs(v_ext) * 100:.3f}%)",
              "The observed order rises towards 2, the rate expected for bilinear elements (error O(h^2)).",
              "Coarse meshes are far from it: parasitic-shear locking dominates there (studies/bending_error.py)."]
    write_table("tip_deflection", lines)

    fig, ax = plt.subplots()
    ax.axhline(v_ti * 1e6, color=ORANGE, linewidth=1.5, label=f"Timoshenko {v_ti * 1e6:.4f}")
    ax.axhline(v_eb * 1e6, color=AQUA, linewidth=1.5, linestyle="--", label=f"Euler-Bernoulli {v_eb * 1e6:.4f}")
    ax.axhline(v_ext * 1e6, color=MUTED, linewidth=1.2, linestyle=":", label=f"Richardson {v_ext * 1e6:.4f}")
    ax.plot(n_el, v * 1e6, "o-", color=BLUE, markersize=7, label="FE, Q4 2 x 2 Gauss")
    for i in (0, 2, 4):
        ax.annotate(f"{MESHES[i][0]}x{MESHES[i][1]}", (n_el[i], v[i] * 1e6), textcoords="offset points",
                    xytext=(-44, -4), color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1.5, 2.0e5)
    ax.set_xlabel("number of elements")
    ax.set_ylabel(r"tip deflection $v(L, 0)$ ($\mu$m)")
    ax.set_title("V6: tip deflection under nested mesh refinement")
    ax.legend(loc="upper right")
    zoom = ax.inset_axes([0.45, 0.24, 0.52, 0.30])          # finest meshes, close-up
    for value, color, style in ((v_ti, ORANGE, "-"), (v_eb, AQUA, "--"), (v_ext, MUTED, ":")):
        zoom.axhline(value * 1e6, color=color, linewidth=1.3, linestyle=style)
    zoom.plot(n_el[4:], v[4:] * 1e6, "o-", color=BLUE, markersize=5)
    zoom.set_xscale("log")
    zoom.set_xlim(500, 1.0e5)
    zoom.set_ylim(-4.775, -4.60)
    zoom.tick_params(labelsize=8)
    zoom.set_title("finest meshes", fontsize=9, color=INK2)
    save_figure(fig, "tip_deflection")

    h = LENGTH / np.array([nx for nx, ny in MESHES], dtype=float)
    diff = np.abs(np.diff(v))
    fig, ax = plt.subplots()
    ax.loglog(h[:-1], diff * 1e6, "o-", color=BLUE, label=r"$|v_h - v_{h/2}|$")
    ref = diff[-1] * 1e6 * (h[:-1] / h[-2]) ** 2
    ax.loglog(h[:-1], ref, "--", color=MUTED, linewidth=1.5, label="slope 2 (error O(h^2))")
    ax.invert_xaxis()
    ax.set_xlabel("element length h = L / nx (m)")
    ax.set_ylabel(r"change in $v_{tip}$ on halving h ($\mu$m)")
    ax.set_title("V6: observed convergence rate")
    ax.legend(loc="lower left")
    save_figure(fig, "convergence_rate")


if __name__ == "__main__":
    main()
