"""V5: the cantilever on a 1 x 1 mesh against the hand solution (checks/closed_form.py)."""
import numpy as np
import pytest

from checks.closed_form import one_element_cantilever
from fem.cantilever import DEPTH, E, LENGTH, NU, P, THICKNESS, solve_cantilever
from fem.post import tip_deflection

# Nodes of the 1 x 1 mesh (fem/mesh.py numbering): 0 = (0, -D/2), 1 = (0, D/2),
# 2 = (L, -D/2), 3 = (L, D/2), i.e. parent nodes 1, 4, 2, 3.
BOTTOM_LEFT, TOP_LEFT, BOTTOM_RIGHT, TOP_RIGHT = 0, 1, 2, 3


@pytest.fixture(scope="module")
def solved():
    return solve_cantilever(1, 1), one_element_cantilever(P, LENGTH, DEPTH, E, NU, THICKNESS)


def test_mesh_node_positions(solved):
    res, hand = solved
    np.testing.assert_allclose(res["nodes"], [[0, -DEPTH / 2], [0, DEPTH / 2],
                                              [LENGTH, -DEPTH / 2], [LENGTH, DEPTH / 2]])


def test_tip_displacements_equal_hand_solution(solved):
    res, hand = solved
    d = res["d"]
    for n in (BOTTOM_RIGHT, TOP_RIGHT):
        assert d[2 * n + 1] == pytest.approx(hand["v_tip"], rel=1e-12)
    assert d[2 * TOP_RIGHT] == pytest.approx(hand["u_top"], rel=1e-12)
    assert d[2 * BOTTOM_RIGHT] == pytest.approx(hand["u_bottom"], rel=1e-12)
    assert tip_deflection(res["nodes"], d, LENGTH) == pytest.approx(hand["v_tip"], rel=1e-12)


def test_values_quoted_in_plan(solved):
    res, hand = solved
    assert hand["v_tip"] == pytest.approx(-6.851084e-8, rel=1e-6)
    assert hand["u_top"] == pytest.approx(3.838313e-9, rel=1e-6)


def test_reactions_and_multipliers_equal_statics(solved):
    res, hand = solved
    R, lam, clamped = res["R"], res["lam"], res["clamped"]
    expected = {2 * BOTTOM_LEFT: hand["Rx_bottom"], 2 * BOTTOM_LEFT + 1: hand["Ry_bottom"],
                2 * TOP_LEFT: hand["Rx_top"], 2 * TOP_LEFT + 1: hand["Ry_top"]}
    for dof, value in expected.items():
        assert R[dof] == pytest.approx(value, rel=1e-12)
    # Each row of C picks one clamped DOF, so lam_i = -R at that DOF.
    np.testing.assert_allclose(lam, -R[clamped], rtol=1e-12)
    # validation item V5: Rx = +133.333 N (bottom), -133.333 N (top), Ry = +5 N each.
    assert hand["Rx_bottom"] == pytest.approx(400.0 / 3.0, rel=1e-12)
    assert hand["Ry_top"] == pytest.approx(5.0, rel=1e-12)


def test_clamp_is_satisfied(solved):
    res, hand = solved
    assert np.max(np.abs(res["d"][res["clamped"]])) < 1e-12 * abs(hand["v_tip"])
