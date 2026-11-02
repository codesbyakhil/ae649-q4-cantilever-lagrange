"""Every closed form used as an analytical counterpart is re-derived symbolically
(checks/derivations.py) and must agree with the expression the project uses."""
import pytest

sp = pytest.importorskip("sympy")

from checks.derivations import ALL, difference_is_zero  # noqa: E402


@pytest.mark.parametrize("name", list(ALL))
def test_derivation_agrees_with_formula_used(name):
    for label, (derived, used) in ALL[name]().items():
        assert difference_is_zero(derived, used), f"{name}: {label}"
