import math

import numpy as np
import pytest

from saddle.basis import BasisError, BasisSet, Shell, load_basis


def self_overlap(shell: Shell) -> float:
    """<x^L g | x^L g> for the normalized contraction (axis Cartesian component)."""
    a = np.asarray(shell.exponents)
    c = np.asarray(shell.normalized_coefficients())
    L = shell.angular_momentum
    dfact = math.prod(range(1, 2 * L, 2)) if L else 1
    s = 0.0
    for ai, ci in zip(a, c, strict=True):
        for aj, cj in zip(a, c, strict=True):
            p = ai + aj
            s += ci * cj * dfact / (2 * p) ** L * (math.pi / p) ** 1.5
    return s


def test_def2_svp_function_counts() -> None:
    b = load_basis("def2-SVP", ["O", "H", "H", "C"])
    assert b.n_functions(["O", "H", "H"]) == 24
    assert b.n_functions(["C"]) == 14
    assert b.pure


@pytest.mark.parametrize("element", ["H", "C", "Cl"])
def test_contractions_are_normalized(element: str) -> None:
    for shell in load_basis("def2-SVP", [element]).shells[element]:
        assert self_overlap(shell) == pytest.approx(1.0, rel=1e-12)


def test_sp_shells_are_split() -> None:
    b = load_basis("6-31G", ["C"])
    ls = [s.angular_momentum for s in b.shells["C"]]
    assert ls.count(1) == 2 and ls.count(0) == 3


def test_jkfit_auxiliary_basis_loads() -> None:
    b = load_basis("def2-universal-jkfit", ["C", "H"])
    assert max(s.angular_momentum for s in b.shells["C"]) >= 3


def test_unknown_element_or_basis() -> None:
    with pytest.raises(BasisError):
        load_basis("def2-SVP", ["Og"])
    with pytest.raises(BasisError):
        load_basis("no-such-basis", ["H"])


def test_validation_rejects_bad_shells() -> None:
    with pytest.raises(BasisError, match="exponent"):
        Shell(0, (1.0, -0.5), (0.5, 0.5), True)
    with pytest.raises(BasisError, match="length"):
        Shell(0, (1.0,), (0.5, 0.5), True)
    good = Shell(0, (1.0,), (1.0,), True)
    with pytest.raises(BasisError, match="mixes"):
        BasisSet("x", {"H": (good, Shell(1, (1.0,), (1.0,), False))})
    with pytest.raises(BasisError, match="contiguous"):
        BasisSet("x", {"H": (good, Shell(2, (1.0,), (1.0,), True))})


def test_function_count_per_shell() -> None:
    assert Shell(2, (1.0,), (1.0,), True).n_functions == 5
    assert Shell(2, (1.0,), (1.0,), False).n_functions == 6
