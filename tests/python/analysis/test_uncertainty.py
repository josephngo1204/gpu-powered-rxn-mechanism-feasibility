import math

import pytest

from saddle.analysis.uncertainty import COMPONENTS, Term, UncertaintyBudget


def full(**override: Term) -> UncertaintyBudget:
    terms = {
        "numerical": Term("numerical", 0.1, "measured"),
        "grid": Term("grid", 0.2, "measured"),
        "basis": Term("basis", 0.3, "measured"),
        "method": Term("method", 1.0, "benchmark"),
        "thermo": Term("thermo", 0.4, "estimate"),
        "solvent": Term("solvent", 0.0, "measured", note="gas phase"),
    }
    terms.update(override)
    return UncertaintyBudget(tuple(terms.values()))


def test_components_match_spec() -> None:
    assert COMPONENTS == ("numerical", "grid", "basis", "method", "thermo", "solvent")


def test_quadrature_sum() -> None:
    b = full()
    assert b.sigma_total == pytest.approx(math.sqrt(0.01 + 0.04 + 0.09 + 1.0 + 0.16))
    assert b.confident
    assert b.missing == ()


def test_qualitative_term_blocks_confidence() -> None:
    b = full(solvent=Term("solvent", None, "qualitative", note="implicit only"))
    assert not b.confident
    assert b.qualitative == ("solvent",)
    assert b.sigma_total == pytest.approx(math.sqrt(0.01 + 0.04 + 0.09 + 1.0 + 0.16))


def test_missing_component_blocks_confidence() -> None:
    b = UncertaintyBudget((Term("numerical", 0.1, "measured"),))
    assert not b.confident
    assert set(b.missing) == set(COMPONENTS) - {"numerical"}


def test_breakdown_is_reported_per_component() -> None:
    rows = full().breakdown()
    assert [r[0] for r in rows] == list(COMPONENTS)
    assert rows[3] == ("method", 1.0, "benchmark", "")


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf])
def test_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError):
        Term("grid", value, "measured")


def test_qualitative_must_not_carry_a_number_and_vice_versa() -> None:
    with pytest.raises(ValueError):
        Term("solvent", 1.0, "qualitative")
    with pytest.raises(ValueError):
        Term("solvent", None, "measured")


def test_rejects_unknown_and_duplicate_components() -> None:
    with pytest.raises(ValueError, match="unknown"):
        Term("vibes", 0.1, "estimate")
    with pytest.raises(ValueError, match="duplicate"):
        UncertaintyBudget((Term("grid", 0.1, "measured"), Term("grid", 0.2, "measured")))
