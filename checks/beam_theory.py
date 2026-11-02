"""Beam-theory reference values for the cantilever (validation item V6(d); independent values: studies/tip_deflection.py).

Cantilever clamped at x = 0, transverse end load P at x = L (P > 0 upward).
Bending moment (sagging positive)   M(x) = P (L - x)
Euler-Bernoulli                     E I v'' = M,  v(0) = v'(0) = 0
                                    => v(L) = P L^3 / (3 E I)
Bending stress                      sigma_xx = -M y / I = -P (L - x) y / I
Timoshenko shear term               shear force V = P, shear strain V / (kappa G A)
                                    => extra tip deflection P L / (kappa G A), kappa = 5/6
                                       for a rectangle (Cook p. 535)
Pure bending (end couple M)         curvature M / (E I) => end rotation M L / (E I)

Run from the project root:  python -m checks.beam_theory
"""
from fem.cantilever import DEPTH, E, LENGTH, NU, P, THICKNESS

KAPPA_RECTANGLE = 5.0 / 6.0


def second_moment(t, depth):
    """I = t D^3 / 12 for a rectangle of width t and depth D."""
    return t * depth**3 / 12.0


def shear_modulus(E, nu):
    """G = E / (2 (1 + nu))."""
    return E / (2.0 * (1.0 + nu))


def tip_deflection_euler_bernoulli(P, L, E, I):
    """v(L) = P L^3 / (3 E I)."""
    return P * L**3 / (3.0 * E * I)


def tip_deflection_shear(P, L, G, A, kappa=KAPPA_RECTANGLE):
    """Shear part of the Timoshenko tip deflection: P L / (kappa G A)."""
    return P * L / (kappa * G * A)


def tip_deflection_timoshenko(P, L, E, nu, t, depth, kappa=KAPPA_RECTANGLE):
    """P L^3 / (3 E I) + P L / (kappa G A) with I = t D^3 / 12, A = t D."""
    I = second_moment(t, depth)
    return (tip_deflection_euler_bernoulli(P, L, E, I)
            + tip_deflection_shear(P, L, shear_modulus(E, nu), t * depth, kappa))


def bending_stress(P, L, I, x, y):
    """sigma_xx(x, y) = -P (L - x) y / I."""
    return -P * (L - x) * y / I


def shear_stress(P, I, depth, y):
    """tau_xy(y) = P (D^2/4 - y^2) / (2 I).

    From equilibrium d(sigma_xx)/dx + d(tau_xy)/dy = 0 with d(sigma_xx)/dx = P y / I and
    tau_xy = 0 on y = +-D/2 (the same as V Q / (I t) with shear force V = P).
    """
    return P * (depth**2 / 4.0 - y**2) / (2.0 * I)


def pure_bending_end_rotation(M, L, E, I):
    """End rotation of a cantilever under an end couple M: theta = M L / (E I)."""
    return M * L / (E * I)


if __name__ == "__main__":
    I = second_moment(THICKNESS, DEPTH)
    G = shear_modulus(E, NU)
    v_eb = tip_deflection_euler_bernoulli(P, LENGTH, E, I)
    v_sh = tip_deflection_shear(P, LENGTH, G, THICKNESS * DEPTH)
    v_ti = tip_deflection_timoshenko(P, LENGTH, E, NU, THICKNESS, DEPTH)
    s_mid = bending_stress(P, LENGTH, I, LENGTH / 2.0, DEPTH / 2.0)
    print(f"I = t D^3 / 12             = {I:.6e} m^4")
    print(f"G = E / 2(1 + nu)          = {G:.6e} Pa")
    print(f"Euler-Bernoulli tip v      = {v_eb * 1e6:.5f} um")
    print(f"Shear term (kappa = 5/6)   = {v_sh * 1e6:.5f} um")
    print(f"Timoshenko tip v           = {v_ti * 1e6:.5f} um")
    print(f"Shear term / Euler-Bernoulli = {v_sh / v_eb * 100:.2f} %  (= E/(4 kappa G) (D/L)^2)")
    print(f"sigma_xx at x = L/2, y = D/2 = {s_mid / 1e3:.4f} kPa (and {-s_mid / 1e3:.4f} kPa at y = -D/2)")
