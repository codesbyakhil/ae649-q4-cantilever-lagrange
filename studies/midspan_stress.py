"""Figure 7 (V6 (d)): bending stress at mid-span against beam theory.

sigma_xx at the Gauss points of the element column at mid-span, against
sigma_xx = -P (L - x) y / I evaluated at the same points. Mid-span is far from both
ends (6.7 depths), so end effects have decayed (Saint-Venant) and beam theory is
the right reference there; the clamped corners carry a stress concentration instead.

Column: the one containing x = L/2, i.e. column (nx - 1) // 2 (when L/2 is a grid line,
the column just left of it). Error measure: max over its Gauss points of
|sigma_FE - sigma_beam| / sigma_peak(x_g), with sigma_peak(x) = |P| (L - x) (D/2) / I the beam
stress at the surface at the same x (the measure used for the independent table).
Run from the project root:  python -m studies.midspan_stress
"""
import numpy as np

from checks.beam_theory import bending_stress, second_moment
from fem.cantilever import DEPTH, LENGTH, P, THICKNESS, solve_cantilever
from fem.post import gauss_point_stresses
from studies.common import AQUA, BLUE, INK2, MESHES, ORANGE, plt, save_figure, setup_style, write_table

I = second_moment(THICKNESS, DEPTH)
# Cross-check only: values from an independent implementation.
INDEPENDENT = {(3, 1): 0.52, (6, 2): 0.53, (12, 4): 0.31, (24, 8): 0.12, (48, 16): 0.041,
               (96, 32): 0.015, (192, 64): 0.0059}


def midspan_column(nx, ny, res):
    """x, y, FE sigma_xx and beam sigma_xx at the Gauss points of the mid-span column."""
    xy, sigma = gauss_point_stresses(res["nodes"], res["elems"], res["d"], res["D"])
    col = (nx - 1) // 2
    elements = np.arange(col * ny, (col + 1) * ny)           # element e = i ny + j
    x = xy[elements, :, 0].ravel()
    y = xy[elements, :, 1].ravel()
    s_fe = sigma[elements, :, 0].ravel()
    return x, y, s_fe, bending_stress(P, LENGTH, I, x, y)


def main():
    setup_style()
    peak_mid = abs(bending_stress(P, LENGTH, I, LENGTH / 2, DEPTH / 2))
    lines = ["V6  bending stress in the mid-span element column vs sigma_xx = -P (L - x) y / I",
             f"    beam-theory surface stress at x = L/2: +-{peak_mid / 1e3:.4f} kPa",
             "",
             "mesh     | column x-range (m) | max error / local peak | reduction | independent | max error / peak at L/2",
             "-" * 108]
    data = {}
    errors = []
    for nx, ny in MESHES:
        res = solve_cantilever(nx, ny)
        x, y, s_fe, s_beam = midspan_column(nx, ny, res)
        local_peak = np.abs(bending_stress(P, LENGTH, I, x, DEPTH / 2))
        err_local = np.max(np.abs(s_fe - s_beam) / local_peak)
        err_mid = np.max(np.abs(s_fe - s_beam)) / peak_mid
        reduction = f"{errors[-1] / err_local:9.2f}" if errors else f"{'-':>9s}"
        errors.append(err_local)
        col = (nx - 1) // 2
        ind = INDEPENDENT.get((nx, ny))
        lines.append(f"{nx:3d}x{ny:<4d} | {col * LENGTH / nx:.4f} - {(col + 1) * LENGTH / nx:.4f}    | "
                     f"{err_local:22.4f} | {reduction} | {ind if ind is not None else '-':>11} | {err_mid:.4f}")
        data[(nx, ny)] = (x, y, s_fe, s_beam, local_peak)
    falls_from = next(i for i in range(len(errors)) if all(e2 < e1 for e1, e2 in zip(errors[i:-1], errors[i + 1:])))
    lines += ["", "reduction = error on the previous mesh / error on this mesh.",
              f"The error falls on every refinement from {MESHES[falls_from][0]} x {MESHES[falls_from][1]} on"
              + (f" (it does not fall from {MESHES[0][0]} x {MESHES[0][1]} to {MESHES[1][0]} x {MESHES[1][1]}:"
                 " both are dominated by locking)." if falls_from > 0 else ".")]
    write_table("midspan_stress", lines)

    fig, ax = plt.subplots()
    yy = np.linspace(-1.0, 1.0, 3)
    ax.plot(yy, yy, color=INK2, linewidth=1.5, label=r"beam theory  $-P(L-x)y/I$")
    for (mesh, color, marker) in (((6, 2), ORANGE, "s"), ((24, 8), AQUA, "^"), ((96, 32), BLUE, "o")):
        x, y, s_fe, s_beam, local_peak = data[mesh]
        ax.plot(y / (DEPTH / 2), s_fe / local_peak, marker, color=color, markersize=6 if mesh != (96, 32) else 4,
                linestyle="none", label=f"FE Gauss points, {mesh[0]} x {mesh[1]}")
    ax.set_xlabel(r"$y / (D/2)$")
    ax.set_ylabel(r"$\sigma_{xx} / [\,|P|(L-x)(D/2)/I\,]$")
    ax.set_title("V6: bending stress at mid-span, FE vs beam theory")
    ax.legend(loc="upper left")
    save_figure(fig, "midspan_stress")


if __name__ == "__main__":
    main()
