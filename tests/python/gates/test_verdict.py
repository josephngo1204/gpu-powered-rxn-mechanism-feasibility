import pytest

from saddle.analysis.uncertainty import Term, UncertaintyBudget
from saddle.gates.gates import GateResult
from saddle.gates.verdict import (
    PathwayOutcome,
    Verdict,
    decide,
    rank_pair,
)

LIMIT = 23.0


def sigma(total: float, qualitative: bool = False) -> UncertaintyBudget:
    terms = [Term(c, 0.0, "measured") for c in ("numerical", "grid", "basis", "thermo")]
    terms.append(Term("method", total, "benchmark"))
    terms.append(
        Term("solvent", None, "qualitative") if qualitative else Term("solvent", 0.0, "measured")
    )
    return UncertaintyBudget(tuple(terms))


def gates(failing: str | None = None) -> list[GateResult]:
    return [
        GateResult(g, g, g != failing, "", None if g != failing else "x")
        for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
    ]


def outcome(name: str, barrier: float, s: float = 1.0, **kw: object) -> PathwayOutcome:
    defaults: dict[str, object] = dict(
        gates=gates(), barrier_kcal_mol=barrier, uncertainty=sigma(s)
    )
    defaults.update(kw)
    return PathwayOutcome(name=name, **defaults)  # type: ignore[arg-type]


def test_supported_when_clearly_below_limit() -> None:
    v = decide([outcome("a", 18.0)], LIMIT)["a"]
    assert v.verdict is Verdict.SUPPORTED


def test_infeasible_when_clearly_above_limit() -> None:
    assert decide([outcome("a", 30.0)], LIMIT)["a"].verdict is Verdict.INFEASIBLE


def test_overlap_with_limit_is_inconclusive() -> None:
    v = decide([outcome("a", 22.0)], LIMIT)["a"]
    assert v.verdict is Verdict.INCONCLUSIVE
    assert "limit" in v.reason


def test_outcompeted_when_another_route_is_lower_by_more_than_2_sigma() -> None:
    res = decide([outcome("a", 18.0, 0.5), outcome("b", 12.0, 0.5)], LIMIT)
    assert res["a"].verdict is Verdict.OUTCOMPETED
    assert "b" in res["a"].reason
    assert res["b"].verdict is Verdict.SUPPORTED


def test_close_competitor_does_not_outcompete() -> None:
    res = decide([outcome("a", 18.0, 0.5), outcome("b", 17.5, 0.5)], LIMIT)
    assert res["a"].verdict is Verdict.SUPPORTED
    assert res["b"].verdict is Verdict.SUPPORTED


def test_invalid_competitor_cannot_outcompete() -> None:
    res = decide([outcome("a", 18.0, 0.5), outcome("b", 5.0, 0.5, gates=gates("G6"))], LIMIT)
    assert res["a"].verdict is Verdict.SUPPORTED
    assert res["b"].verdict is Verdict.INCONCLUSIVE
    assert "G6" in res["b"].reason


def test_failed_gate_blocks_any_confident_verdict() -> None:
    v = decide([outcome("a", 10.0, gates=gates("G4"))], LIMIT)["a"]
    assert v.verdict is Verdict.INCONCLUSIVE
    assert v.failed_gates == ("G4",)


def test_missing_gates_block_verdict() -> None:
    v = decide([outcome("a", 10.0, gates=None)], LIMIT)["a"]
    assert v.verdict is Verdict.INCONCLUSIVE


def test_incomplete_competition_is_inconclusive_and_flagged() -> None:
    v = decide([outcome("a", 10.0, gates=gates("G7"))], LIMIT)["a"]
    assert v.verdict is Verdict.INCONCLUSIVE
    assert v.incomplete


def test_qualitative_uncertainty_blocks_confident_verdict() -> None:
    v = decide([outcome("a", 10.0, uncertainty=sigma(1.0, qualitative=True))], LIMIT)["a"]
    assert v.verdict is Verdict.INCONCLUSIVE
    assert "qualitative" in v.reason


def test_no_saddle_found_and_barrierless() -> None:
    res = decide(
        [
            PathwayOutcome("x", saddle_found=False, search_exhausted=True),
            PathwayOutcome("y", saddle_found=False, barrierless_scan=True),
            PathwayOutcome("z", saddle_found=False),
        ],
        LIMIT,
    )
    assert res["x"].verdict is Verdict.NO_SADDLE_FOUND
    assert res["y"].verdict is Verdict.BARRIERLESS_CANDIDATE
    assert res["z"].verdict is Verdict.INCONCLUSIVE


def test_verdict_carries_numbers_for_the_report() -> None:
    v = decide([outcome("a", 18.0, 1.0)], LIMIT)["a"]
    assert v.barrier_kcal_mol == 18.0
    assert v.sigma_kcal_mol == pytest.approx(1.0)
    assert v.limit_kcal_mol == LIMIT


def test_rank_pair_resolution() -> None:
    # OTs vs OMs: 0.15 kcal/mol gap must not be resolved (spec §3 C8).
    r = rank_pair(outcome("ots", 20.15, 0.4), outcome("oms", 20.0, 0.4))
    assert not r.resolved
    # OTf vs OTs: 5.7 kcal/mol gap is a strong signal.
    r = rank_pair(outcome("otf", 14.45, 1.0), outcome("ots", 20.15, 1.0))
    assert r.resolved and r.lower == "otf"
    assert r.gap_kcal_mol == pytest.approx(5.7)


def test_rank_pair_needs_valid_pathways() -> None:
    r = rank_pair(outcome("a", 10.0, gates=gates("G1")), outcome("b", 20.0))
    assert not r.resolved
    assert "a" in r.reason


def test_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError):
        decide([outcome("a", 1.0), outcome("a", 2.0)], LIMIT)
