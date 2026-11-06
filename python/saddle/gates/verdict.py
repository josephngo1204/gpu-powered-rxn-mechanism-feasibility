"""Verdict logic.

SUPPORTED            gates pass, dG‡ + 2σ below the limit, not outcompeted
OUTCOMPETED          feasible, but another valid route is lower by more than 2σ
INFEASIBLE           dG‡ - 2σ above the limit
NO_SADDLE_FOUND      search budget exhausted without a valid saddle
BARRIERLESS_CANDIDATE tight scans are monotonic, no saddle
INCONCLUSIVE         anything else

Confident verdicts (SUPPORTED, OUTCOMPETED, INFEASIBLE) need every gate to
pass and a confident uncertainty budget. For a comparison between two
barriers, σ_diff = sqrt(σ_a² + σ_b²): conservative, it ignores the partial
cancellation of correlated method error.
"""

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass

from saddle.analysis.uncertainty import UncertaintyBudget
from saddle.gates.gates import GateResult

N_SIGMA = 2.0


class Verdict(enum.Enum):
    SUPPORTED = "SUPPORTED"
    OUTCOMPETED = "OUTCOMPETED"
    INFEASIBLE = "INFEASIBLE"
    NO_SADDLE_FOUND = "NO_SADDLE_FOUND"
    BARRIERLESS_CANDIDATE = "BARRIERLESS_CANDIDATE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class PathwayOutcome:
    name: str
    gates: Sequence[GateResult] | None = None
    barrier_kcal_mol: float | None = None
    uncertainty: UncertaintyBudget | None = None
    saddle_found: bool = True
    search_exhausted: bool = False
    barrierless_scan: bool = False

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(g.gate for g in self.gates or () if not g.passed)

    @property
    def valid(self) -> bool:
        """Gates all present and passing, barrier and confident σ available."""
        return (
            self.saddle_found
            and self.gates is not None
            and len(self.gates) > 0
            and not self.failed_gates
            and self.barrier_kcal_mol is not None
            and self.uncertainty is not None
            and self.uncertainty.confident
        )


@dataclass(frozen=True)
class VerdictResult:
    pathway: str
    verdict: Verdict
    reason: str
    barrier_kcal_mol: float | None
    sigma_kcal_mol: float | None
    limit_kcal_mol: float
    failed_gates: tuple[str, ...] = ()
    incomplete: bool = False


def _sigma(p: PathwayOutcome) -> float:
    assert p.uncertainty is not None
    return p.uncertainty.sigma_total


def _blocked(p: PathwayOutcome) -> str | None:
    """Why a found saddle cannot support a confident verdict, or None."""
    if p.gates is None or len(p.gates) == 0:
        return "gates not evaluated"
    if p.failed_gates:
        return "failed gates: " + ", ".join(p.failed_gates)
    if p.barrier_kcal_mol is None:
        return "no barrier"
    if p.uncertainty is None:
        return "no uncertainty budget"
    if p.uncertainty.qualitative:
        return "qualitative uncertainty terms: " + ", ".join(p.uncertainty.qualitative)
    if p.uncertainty.missing:
        return "missing uncertainty terms: " + ", ".join(p.uncertainty.missing)
    return None


def decide(pathways: Sequence[PathwayOutcome], limit_kcal_mol: float) -> dict[str, VerdictResult]:
    names = [p.name for p in pathways]
    if len(set(names)) != len(names):
        raise ValueError("pathway names must be unique")
    out: dict[str, VerdictResult] = {}
    for p in pathways:
        sig = _sigma(p) if p.uncertainty is not None else None

        def result(
            v: Verdict, reason: str, p: PathwayOutcome = p, sig: float | None = sig
        ) -> VerdictResult:
            return VerdictResult(
                p.name,
                v,
                reason,
                p.barrier_kcal_mol,
                sig,
                limit_kcal_mol,
                p.failed_gates,
                incomplete="G7" in p.failed_gates,
            )

        if not p.saddle_found:
            if p.barrierless_scan:
                out[p.name] = result(
                    Verdict.BARRIERLESS_CANDIDATE, "tight scans monotonic, no saddle"
                )
            elif p.search_exhausted:
                out[p.name] = result(Verdict.NO_SADDLE_FOUND, "search budget exhausted")
            else:
                out[p.name] = result(Verdict.INCONCLUSIVE, "no saddle yet and search not exhausted")
            continue
        blocked = _blocked(p)
        if blocked is not None:
            out[p.name] = result(Verdict.INCONCLUSIVE, blocked)
            continue
        assert p.barrier_kcal_mol is not None and sig is not None
        g = p.barrier_kcal_mol
        if g - N_SIGMA * sig > limit_kcal_mol:
            out[p.name] = result(
                Verdict.INFEASIBLE, f"{g:.2f} - 2σ above limit {limit_kcal_mol:.2f}"
            )
            continue
        if not g + N_SIGMA * sig < limit_kcal_mol:
            out[p.name] = result(
                Verdict.INCONCLUSIVE, f"{g:.2f} ± 2σ({sig:.2f}) overlaps limit {limit_kcal_mol:.2f}"
            )
            continue
        winners = []
        for other in pathways:
            if other.name == p.name or not other.valid:
                continue
            assert other.barrier_kcal_mol is not None
            diff_sigma = math.hypot(sig, _sigma(other))
            if g - other.barrier_kcal_mol > N_SIGMA * diff_sigma:
                winners.append(other.name)
        if winners:
            out[p.name] = result(
                Verdict.OUTCOMPETED, "lower by more than 2σ: " + ", ".join(winners)
            )
        else:
            out[p.name] = result(
                Verdict.SUPPORTED, f"{g:.2f} + 2σ below limit {limit_kcal_mol:.2f}"
            )
    return out


@dataclass(frozen=True)
class Ranking:
    resolved: bool
    lower: str | None
    gap_kcal_mol: float | None
    sigma_kcal_mol: float | None
    reason: str


def rank_pair(a: PathwayOutcome, b: PathwayOutcome) -> Ranking:
    """Order two pathways only when the gap exceeds 2σ_diff; otherwise 'not resolved'."""
    for p in (a, b):
        if not p.valid:
            return Ranking(False, None, None, None, f"{p.name}: {_blocked(p) or 'invalid'}")
    assert a.barrier_kcal_mol is not None and b.barrier_kcal_mol is not None
    gap = abs(a.barrier_kcal_mol - b.barrier_kcal_mol)
    s = math.hypot(_sigma(a), _sigma(b))
    lower = a.name if a.barrier_kcal_mol < b.barrier_kcal_mol else b.name
    if gap > N_SIGMA * s:
        return Ranking(True, lower, gap, s, f"{lower} lower by {gap:.2f} > 2σ ({2 * s:.2f})")
    return Ranking(False, None, gap, s, f"not resolved: gap {gap:.2f} <= 2σ ({2 * s:.2f})")
