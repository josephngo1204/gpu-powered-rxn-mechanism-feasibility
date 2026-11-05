"""Plain-language wording: what each gate asks, what a verdict means, how fast is fast."""

from __future__ import annotations

import math

from saddle.analysis.kinetics import half_life_s
from saddle.dashboard.views import Pathway

GATE_QUESTIONS = {
    "G1": "Are the forces on every atom essentially zero (a true stationary point)?",
    "G2": "Did the electronic-structure calculation converge to a stable solution?",
    "G3": "Is the curvature matrix (Hessian) numerically trustworthy?",
    "G4": "Is there exactly one downhill direction, as a mountain pass has?",
    "G5": "Does that downhill direction break and form the intended bonds?",
    "G6": "Does sliding downhill both ways reach the intended reactants and products?",
    "G7": "Were all declared competing pathways evaluated?",
    "G8": "Does the verdict survive changing grid, precision, functional and orientation?",
}

VERDICT_MEANING = {
    "SUPPORTED": "fast enough, even at the slow end of the error bar",
    "INFEASIBLE": "too slow, even at the fast end of the error bar",
    "INCONCLUSIVE": "undecided: the error bar straddles the limit, or a check failed",
    "OUTCOMPETED": "fast enough, but a competing pathway is clearly faster",
    "NO_SADDLE_FOUND": "no valid transition state was found",
    "BARRIERLESS_CANDIDATE": "no barrier was found; the step may happen without one",
}

# (seconds, singular unit name), smallest first.
_UNITS = (
    (1e-12, "picosecond"),
    (1e-9, "nanosecond"),
    (1e-6, "microsecond"),
    (1e-3, "millisecond"),
    (1.0, "second"),
    (60.0, "minute"),
    (3600.0, "hour"),
    (86400.0, "day"),
    (3.15576e7, "year"),
)


def safe_half_life_s(barrier_kcal_mol: float, temperature_k: float) -> float:
    """Eyring half-life, or infinity when the rate underflows (huge barriers)."""
    try:
        return half_life_s(barrier_kcal_mol, temperature_k)
    except (OverflowError, ZeroDivisionError):
        return math.inf


def duration(seconds: float) -> str:
    """A human-readable duration: '2.3 microseconds', '5 hours', '1e+12 years'."""
    if math.isinf(seconds):
        return "practically forever"
    if seconds < _UNITS[0][0]:
        return "under a picosecond"
    size, unit = next((u for u in reversed(_UNITS) if seconds >= u[0]), _UNITS[0])
    value = seconds / size
    if unit == "year" and value >= 1e4:
        return f"{value:.0e} years"
    shown = f"{value:.2g}" if value < 10 else f"{value:.0f}"
    return f"{shown} {unit}{'' if shown == '1' else 's'}"


def headline(p: Pathway, temperature_k: float | None, target_s: float | None) -> str:
    """One sentence a non-specialist can read: the verdict, in time units."""
    meaning = VERDICT_MEANING.get(p.verdict, "no verdict")
    if p.barrier is None or temperature_k is None:
        return f"{p.verdict.replace('_', ' ').capitalize()}: {meaning}."
    t_half = safe_half_life_s(p.barrier, temperature_k)
    celsius = temperature_k - 273.15
    sentence = f"Half-life about {duration(t_half)} at {celsius:.0f} °C"
    if target_s is not None:
        sentence += f" (target: under {duration(target_s)})"
    return f"{sentence}, so {meaning}."


# Verdict -> (status color role, icon). Status always ships with icon + word.
_VERDICT_STATUS = {
    "SUPPORTED": ("good", "✓"),
    "OUTCOMPETED": ("warning", "↓"),
    "INFEASIBLE": ("critical", "×"),
    "NO_SADDLE_FOUND": ("critical", "×"),
    "BARRIERLESS_CANDIDATE": ("warning", "?"),
    "INCONCLUSIVE": ("warning", "?"),
}


def verdict_status(verdict: str) -> tuple[str, str]:
    return _VERDICT_STATUS.get(verdict, ("warning", "?"))
