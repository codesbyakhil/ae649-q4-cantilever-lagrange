"""Figure 8 (V6 (a), (b)): support reactions R = -C^T lambda along the clamp.

Table, every mesh: sum Rx = 0, sum Ry = -P, net moment of reactions and loads = 0,
and Clapeyron d^T K d = f^T d. Figure: nodal reactions divided by their tributary
area (t h, t h/2 at the corners) against the beam-theory tractions of the clamp on the
beam at x = 0:  t_x = -sigma_xx(0, y) = P L y / I,  t_y = -tau_xy(0, y) = -P (D^2/4 - y^2) / (2 I).
Run from the project root:  python -m studies.reactions
"""
import numpy as np

from checks.beam_theory import bending_stress, second_moment, shear_stress
from fem.cantilever import DEPTH, LENGTH, P, THICKNESS, solve_cantilever
from fem.mesh import edge_nodes
from fem.post import force_resultants
from studies.common import BLUE, INK2, MESHES, ORANGE, plt, save_figure, setup_style, write_table

I = second_moment(THICKNESS, DEPTH)


def clamp_tractions(res, ny):
    """y of the clamped nodes and R / (t h_trib) in x and y (traction estimates, Pa)."""
    left = edge_nodes(res["nodes"], 0.0)
    h_trib = np.full(ny + 1, DEPTH / ny)
    h_trib[[0, -1]] = DEPTH / (2 * ny)
    R = res["R"]
    return res["nodes"][left, 1], R[2 * left] / (THICKNESS * h_trib), R[2 * left + 1] / (THICKNESS * h_trib)


def main():
    setup_style()
    lines = ["V6  reactions R = -C^T lambda on the clamp; statics and Clapeyron on every mesh (P = -10 N)",
             f"    expected: sum Rx = 0, sum Ry = -P = {-P:.1f} N, reaction moment about (0, 0) = -P L = {-P * LENGTH:.1f} N m",
             "",
             "mesh     | sum Rx (N) | sum Ry (N)       | moment of R (N m) | net moment R + f (N m) | "
             "(d.K.d - f.d) / f.d | max|K d - f + C^T lambda| / max|K d|",
             "-" * 138]
    profiles = {}
    for nx, ny in MESHES:
        res = solve_cantilever(nx, ny)
        nodes, R, f, d, K = res["nodes"], res["R"], res["f"], res["d"], res["K"]
        Rx, Ry, MR = force_resultants(nodes, R)
        fx, fy, Mf = force_resultants(nodes, f)
        clap = (d @ (K @ d) - f @ d) / (f @ d)
        Kd = K @ d
        row1 = np.max(np.abs(Kd - f + res["C"].T @ res["lam"])) / np.max(np.abs(Kd))   # K d + C^T lam = f
        lines.append(f"{nx:3d}x{ny:<4d} | {Rx: .1e}   | {Ry:.12f}  | {MR:.12f}   | {MR + Mf: .1e}               | "
                     f"{clap: .1e}             | {row1:.1e}")
        if (nx, ny) in ((24, 8), (96, 32), (384, 128)):
            profiles[(nx, ny)] = clamp_tractions(res, ny)
    lines += ["", "All exact up to round-off: element stiffnesses annihilate rigid-body motions, so the",
              "nodal forces K d are self-equilibrated, and with C d = 0 the reactions do no work.",
              "The last column is the residual of the first block row, K d + C^T lambda = f, relative to",
              "the largest nodal force, so R = K d - f = -C^T lambda holds to round-off."]
    y96, tx96, ty96 = profiles[(96, 32)]
    tx_beam = -bending_stress(P, LENGTH, I, 0.0, y96)
    ty_beam = -shear_stress(P, I, DEPTH, y96)
    inner = slice(1, -1)
    lines += ["", "Distribution along the clamp (R / tributary area t h, t h/2 at the corners):",
              f"  x (bending), 96 x 32, interior nodes: max |t_x(FE) - PLy/I| / max|PLy/I| = "
              f"{np.max(np.abs(tx96[inner] - tx_beam[inner])) / np.max(np.abs(tx_beam)):.3f}",
              "  corner nodes, t(FE) / beam value at the corner, and the corner traction itself:",
              "  mesh     | t_x / (PLy/I) | t_x corner (kPa) | t_y corner (kPa) | Ry corner node (N)"]
    for mesh in ((24, 8), (96, 32), (384, 128)):
        y, tx, ty = profiles[mesh]
        ry_corner = ty[-1] * THICKNESS * DEPTH / (2 * mesh[1])
        lines.append(f"  {mesh[0]:3d}x{mesh[1]:<4d} | {tx[-1] / -bending_stress(P, LENGTH, I, 0.0, y[-1]):13.3f} | "
                     f"{abs(tx[-1]) / 1e3:16.1f} | {ty[-1] / 1e3:16.1f} | {ry_corner:.4f}")
    lines += ["  The corner tractions keep growing as the mesh is refined: the clamped corners are",
              "  singular points of the 2D elasticity solution (the clamp also stops the Poisson",
              "  contraction there). In y the corners carry a large share of the shear force, so the",
              "  parabolic beam shear does not describe the traction AT a fully clamped 2D edge; beam",
              "  theory only fixes the resultants there (Saint-Venant), and those are exact above."]
    write_table("reactions", lines)

    yy = np.linspace(-DEPTH / 2, DEPTH / 2, 101)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    ax1.plot(-bending_stress(P, LENGTH, I, 0.0, yy) / 1e3, yy * 1e3, color=INK2, linewidth=1.5,
             label=r"beam: $PLy/I$")
    ax2.plot(-shear_stress(P, I, DEPTH, yy) / 1e3, yy * 1e3, color=INK2, linewidth=1.5,
             label=r"beam: $-P(D^2/4-y^2)/2I$" + "\n(resultant only at a 2D clamp)")
    for mesh, color, marker in (((24, 8), ORANGE, "s"), ((96, 32), BLUE, "o")):
        y, tx, ty = profiles[mesh]
        size = 6 if mesh == (24, 8) else 4
        ax1.plot(tx / 1e3, y * 1e3, marker, color=color, markersize=size, label=f"FE {mesh[0]} x {mesh[1]}")
        ax2.plot(ty / 1e3, y * 1e3, marker, color=color, markersize=size, label=f"FE {mesh[0]} x {mesh[1]}")
    ax1.set_xlabel(r"$R_x / (t\,h_{trib})$ (kPa)")
    ax2.set_xlabel(r"$R_y / (t\,h_{trib})$ (kPa)")
    ax1.set_ylabel("y (mm)")
    ax1.set_title("x: linear, as beam theory (except corners)")
    ax2.set_title("y: sum = -P, concentrated at the corners")
    ax1.legend(loc="lower left")
    ax2.legend(loc="center right")
    fig.suptitle(r"V6: reactions $R = -C^T\lambda$ along the clamp vs beam theory", color=INK2)
    save_figure(fig, "reactions")


if __name__ == "__main__":
    main()
