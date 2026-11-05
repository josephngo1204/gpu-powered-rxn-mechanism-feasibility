"""TS validation gates G1-G8 (spec §5).

Each gate is a pure function of an evidence bundle and returns pass/fail, a
reason and the spec's failure action. Missing evidence always fails: a gate
never passes because something was not checked (spec §10, §12).

Thresholds are spec values except where marked PROPOSED; changing any of
them is an "Ask first" item (spec §12).
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from saddle.analysis.hessian_quality import QualityReport
from saddle.graphs import Graph

FloatArray = npt.NDArray[np.float64]
Bond = tuple[int, int]

G1_RMS_GRADIENT = 1e-5  # Eh/bohr
G1_MAX_GRADIENT = 1.5e-5
G4_SMALL_MODE_FACTOR = 3.0  # second imaginary mode within 3x the noise floor
# PROPOSED (not in spec): minimum share of the TS mode's bond-stretch
# amplitude that must fall on the declared breaking/forming bonds.
G5_MIN_REACTIVE_SHARE = 0.5
G5_SIGN_IGNORE_FRACTION = 1e-3
G8_REQUIRED_CHECKS = ("grid", "precision", "functional")


@dataclass(frozen=True)
class GateResult:
    gate: str
    name: str
    passed: bool
    reason: str
    failure_action: str | None


@dataclass(frozen=True)
class Evidence:
    """Everything the gates may inspect. Fields default to None (not collected)."""

    gradient: FloatArray | None = None  # internal (TR-projected) gradient
    raw_gradient: FloatArray | None = None
    scf_converged: bool | None = None
    orbitals_stable: bool | None = None
    uses_pcm: bool | None = None
    pcm_converged: bool | None = None
    hessian_quality: QualityReport | None = None
    frequencies_cm: FloatArray | None = None
    noise_floor_cm: float | None = None
    second_mode_persists: bool | None = None
    ts_mode_cartesian: FloatArray | None = None
    coords_bohr: FloatArray | None = None
    atom_map: Sequence[int] | None = None
    breaking: Sequence[Bond] = ()
    forming: Sequence[Bond] = ()
    reactant_graph: Graph | None = None
    expected_product_graph: Graph | None = None
    irc_forward_graph: Graph | None = None
    irc_reverse_graph: Graph | None = None
    alternative_product_graphs: Mapping[str, Graph] = field(default_factory=dict)
    declared_competitors: Sequence[str] = ()
    competitors_evaluated: Mapping[str, bool] | None = None
    baseline_verdict: str | None = None
    robustness_verdicts: Mapping[str, str] | None = None

    def replace(self, **changes: Any) -> "Evidence":
        return dataclasses.replace(self, **changes)


def _missing(gate: str, name: str, what: str, action: str) -> GateResult:
    return GateResult(gate, name, False, f"missing evidence: {what}", action)


def g1_stationarity(e: Evidence) -> GateResult:
    gate, name, action = "G1", "stationarity", "refine_again"
    if e.gradient is None:
        return _missing(gate, name, "gradient", action)
    g = np.asarray(e.gradient, dtype=np.float64)
    rms = float(np.sqrt(np.mean(g**2)))
    gmax = float(np.max(np.abs(g)))
    ok = rms <= G1_RMS_GRADIENT and gmax <= G1_MAX_GRADIENT
    reason = (
        f"rms {rms:.2e}, max {gmax:.2e} Eh/bohr (limits {G1_RMS_GRADIENT:g}, {G1_MAX_GRADIENT:g})"
    )
    # G1 is judged on the internal (TR-projected) gradient; a spurious DFT-grid
    # torque is reported but does not count (user decision, tasks/plan.md).
    if e.raw_gradient is not None:
        spurious = float(np.max(np.abs(np.asarray(e.raw_gradient, dtype=np.float64) - g)))
        reason += f"; translation/rotation part removed (max {spurious:.1e})"
    return GateResult(gate, name, ok, reason, None if ok else action)


def g2_electronic(e: Evidence) -> GateResult:
    gate, name, action = "G2", "electronic quality", "change_guess_or_reject"
    if e.scf_converged is None or e.orbitals_stable is None or e.uses_pcm is None:
        return _missing(gate, name, "SCF convergence, orbital stability or PCM usage", action)
    if e.uses_pcm and e.pcm_converged is None:
        return _missing(gate, name, "PCM convergence flag", action)
    problems = []
    if not e.scf_converged:
        problems.append("SCF not converged")
    if not e.orbitals_stable:
        problems.append("orbital solution unstable")
    if e.uses_pcm and not e.pcm_converged:
        problems.append("PCM not converged")
    ok = not problems
    return GateResult(
        gate, name, ok, "; ".join(problems) or "converged and stable", None if ok else action
    )


def g3_hessian(e: Evidence) -> GateResult:
    gate, name, action = "G3", "Hessian quality", "tighten_and_recompute"
    if e.hessian_quality is None:
        return _missing(gate, name, "Hessian quality report", action)
    q = e.hessian_quality
    failing = [c.name for c in q.checks if c.required and not (c.passed and c.evaluated)]
    failing += [c.name for c in q.checks if not c.required and c.evaluated and not c.passed]
    ok = q.passed
    reason = "all quality targets met" if ok else "failed: " + ", ".join(failing)
    if q.flagged_columns:
        reason += f"; spiky columns {list(q.flagged_columns)}"
    return GateResult(gate, name, ok, reason, None if ok else action)


def g4_saddle_order(e: Evidence) -> GateResult:
    gate, name, action = "G4", "saddle order", "displace_and_reoptimize"
    if e.frequencies_cm is None or e.noise_floor_cm is None:
        return _missing(gate, name, "frequencies or noise floor", action)
    f = np.sort(np.asarray(e.frequencies_cm, dtype=np.float64))
    floor = float(e.noise_floor_cm)
    imaginary = f[f < -floor]
    if imaginary.size == 0:
        return GateResult(
            gate, name, False, "no imaginary mode above the noise floor: a minimum", action
        )
    if imaginary.size == 1:
        return GateResult(gate, name, True, f"one imaginary mode {imaginary[0]:.1f} cm^-1", None)
    second = imaginary[1:]
    if imaginary.size == 2 and abs(second[0]) <= G4_SMALL_MODE_FACTOR * floor:
        if e.second_mode_persists is False:
            return GateResult(
                gate,
                name,
                True,
                f"second mode {second[0]:.1f} cm^-1 vanished at tighter settings",
                None,
            )
        if e.second_mode_persists is None:
            return GateResult(
                gate,
                name,
                False,
                f"second imaginary mode {second[0]:.1f} cm^-1 within 3x noise floor; "
                "displace both ways and recheck at tighter settings",
                action,
            )
    return GateResult(
        gate,
        name,
        False,
        f"{imaginary.size} imaginary modes: second-order or higher saddle",
        action,
    )


def _bond_changes(
    coords_bohr: FloatArray, mode: FloatArray, atom_map: Sequence[int], bonds: Sequence[Bond]
) -> dict[Bond, float]:
    """First-order bond-length change along the mode for each bond."""
    x = np.asarray(coords_bohr, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(mode, dtype=np.float64).reshape(-1, 3)
    index = {a: i for i, a in enumerate(atom_map)}
    out: dict[Bond, float] = {}
    for a, b in bonds:
        i, j = index[a], index[b]
        d = x[j] - x[i]
        out[(a, b)] = float(np.dot(d, m[j] - m[i]) / np.linalg.norm(d))
    return out


def g5_mode_character(e: Evidence) -> GateResult:
    gate, name, action = "G5", "mode character", "reject_wrong_saddle"
    if e.ts_mode_cartesian is None or e.coords_bohr is None or e.atom_map is None:
        return _missing(gate, name, "TS mode, coordinates or atom map", action)
    if e.reactant_graph is None or e.expected_product_graph is None:
        return _missing(gate, name, "reactant and product graphs", action)
    reactive = [tuple(sorted(b)) for b in (*e.breaking, *e.forming)]
    if not reactive:
        return _missing(gate, name, "declared bond changes", action)
    reference = {tuple(sorted(b)) for b in e.reactant_graph | e.expected_product_graph}
    reference |= set(reactive)
    changes = _bond_changes(
        e.coords_bohr, e.ts_mode_cartesian, e.atom_map, [(int(a), int(b)) for a, b in reference]
    )
    total = sum(v * v for v in changes.values())
    if total <= 0.0:
        return GateResult(
            gate, name, False, "mode changes no bond lengths (e.g. a torsion)", action
        )
    share = sum(changes[(int(a), int(b))] ** 2 for a, b in reactive) / total
    if share < G5_MIN_REACTIVE_SHARE:
        return GateResult(
            gate, name, False, f"reactive bonds carry {share:.0%} of bond-stretch amplitude", action
        )
    scale = max(abs(v) for v in changes.values())

    def signs(bonds: Sequence[Bond]) -> set[float]:
        values = [changes[(min(a, b), max(a, b))] for a, b in bonds]
        return {float(np.sign(v)) for v in values if abs(v) > G5_SIGN_IGNORE_FRACTION * scale}

    s_break, s_form = signs(e.breaking), signs(e.forming)
    if len(s_break) > 1 or len(s_form) > 1 or (s_break and s_form and s_break == s_form):
        return GateResult(
            gate, name, False, "bond-change sign pattern does not match breaking vs forming", action
        )
    return GateResult(gate, name, True, f"reactive bonds carry {share:.0%} of the mode", None)


def g6_connectivity(e: Evidence) -> GateResult:
    gate, name, action = "G6", "connectivity", "relabel_or_reject"
    if e.irc_forward_graph is None or e.irc_reverse_graph is None:
        return _missing(gate, name, "IRC endpoint graphs", action)
    if e.reactant_graph is None or e.expected_product_graph is None:
        return _missing(gate, name, "reactant and expected product graphs", action)
    ends = (e.irc_forward_graph, e.irc_reverse_graph)
    r, p = e.reactant_graph, e.expected_product_graph
    if (ends[0] == r and ends[1] == p) or (ends[0] == p and ends[1] == r):
        return GateResult(gate, name, True, "IRC connects reactants and intended products", None)
    others = [n for n, g in e.alternative_product_graphs.items() if any(end == g for end in ends)]
    if others:
        return GateResult(
            gate, name, False, f"IRC endpoint matches declared pathway {', '.join(others)}", action
        )
    return GateResult(gate, name, False, "IRC endpoints do not match the intended graphs", action)


def g7_competition(e: Evidence) -> GateResult:
    gate, name, action = "G7", "competition", "mark_incomplete"
    if e.competitors_evaluated is None:
        return _missing(gate, name, "competitor status", action)
    pending = [c for c in e.declared_competitors if not e.competitors_evaluated.get(c, False)]
    ok = not pending
    reason = "all declared competitors evaluated" if ok else f"not evaluated: {', '.join(pending)}"
    return GateResult(gate, name, ok, reason, None if ok else action)


def g8_robustness(e: Evidence) -> GateResult:
    gate, name, action = "G8", "robustness", "mark_inconclusive"
    if e.baseline_verdict is None or e.robustness_verdicts is None:
        return _missing(gate, name, "baseline or robustness verdicts", action)
    absent = [c for c in G8_REQUIRED_CHECKS if c not in e.robustness_verdicts]
    if absent:
        return _missing(gate, name, f"robustness checks {', '.join(absent)}", action)
    changed = [k for k, v in e.robustness_verdicts.items() if v != e.baseline_verdict]
    ok = not changed
    reason = "verdict stable" if ok else f"verdict changes under: {', '.join(changed)}"
    return GateResult(gate, name, ok, reason, None if ok else action)


ALL_GATES: tuple[Callable[[Evidence], GateResult], ...] = (
    g1_stationarity,
    g2_electronic,
    g3_hessian,
    g4_saddle_order,
    g5_mode_character,
    g6_connectivity,
    g7_competition,
    g8_robustness,
)


# G1-G7. G8 (robustness) compares verdicts, so it runs only after every
# variant has its own preliminary verdict from these seven.
GATES_BEFORE_ROBUSTNESS = ALL_GATES[:-1]


def evaluate_all(e: Evidence) -> list[GateResult]:
    return [gate(e) for gate in ALL_GATES]
