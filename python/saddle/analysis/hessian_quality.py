"""Hessian quality report against the spec §6 targets (feeds gate G3).

Targets come from spec §6. Changing them is an "Ask first" item (spec §12).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import numpy.typing as npt

from saddle import _core

FloatArray = npt.NDArray[np.float64]

# Frequencies above this are compared in absolute cm^-1 (spec §6).
STABLE_MODE_FLOOR_CM = 100.0
# A column whose asymmetry share exceeds this multiple of the median is a spike.
SPIKE_SHARE_RATIO = 4.0


@dataclass(frozen=True)
class HessianTargets:
    max_asymmetry: float = 1e-4
    max_external_residual_cm: float = 5.0
    max_frequency_change_cm: float = 1.0
    max_imaginary_change_fraction: float = 0.02
    max_delta_g_change_kcal_mol: float = 0.1
    imaginary_threshold_cm: float = 0.0


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    target: float
    passed: bool
    evaluated: bool = True
    required: bool = True
    detail: str = ""


@dataclass(frozen=True)
class QualityReport:
    checks: tuple[Check, ...]
    flagged_columns: tuple[int, ...]
    frequencies_cm: Mapping[float, FloatArray] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed and c.evaluated for c in self.checks if c.required) and all(
            c.passed for c in self.checks if c.evaluated
        )

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            if not c.evaluated:
                status = "NOT EVALUATED" + ("" if c.required else " (optional)")
            lines.append(f"{c.name}: {status} value={c.value:.3g} target={c.target:.3g} {c.detail}")
        return "\n".join(lines)


def _spiky_columns(raw: FloatArray) -> list[int]:
    asym = raw - raw.T
    col_sq = np.sum(asym**2, axis=0)
    total = float(np.sum(col_sq))
    if total == 0.0:
        return []
    share = col_sq / (total / 2.0)
    median = float(np.median(share))
    if median == 0.0:
        return [int(i) for i in np.flatnonzero(share > 0.0)]
    return [int(i) for i in np.flatnonzero(share > SPIKE_SHARE_RATIO * median)]


def assess_hessians(
    raw_by_step: Mapping[float, FloatArray],
    coords_bohr: FloatArray,
    masses_amu: FloatArray,
    *,
    delta_g_by_precision: Mapping[str, float] | None = None,
    targets: HessianTargets | None = None,
) -> QualityReport:
    """Assess unsymmetrized FD Hessians computed at one or more accepted steps."""
    if not raw_by_step:
        raise ValueError("assess_hessians needs at least one Hessian")
    t = targets or HessianTargets()
    coords = np.asarray(coords_bohr, dtype=np.float64)
    masses = np.asarray(masses_amu, dtype=np.float64)

    asymmetry = 0.0
    external = 0.0
    flagged: set[int] = set()
    freqs: dict[float, FloatArray] = {}
    for step, raw in raw_by_step.items():
        raw = np.asarray(raw, dtype=np.float64)
        norm = float(np.linalg.norm(raw))
        a = float(np.linalg.norm(raw - raw.T) / norm) if norm > 0 else 0.0
        asymmetry = max(asymmetry, a)
        if a > t.max_asymmetry:
            flagged.update(_spiky_columns(raw))
        vib = _core.analyze_vibrations(0.5 * (raw + raw.T), coords, masses)
        external = max(external, float(vib.max_external_residual_cm))
        freqs[step] = np.asarray(vib.frequencies_cm)

    checks = [
        Check("asymmetry", asymmetry, t.max_asymmetry, asymmetry <= t.max_asymmetry),
        Check(
            "column_spikes",
            float(len(flagged)),
            0.0,
            not flagged,
            detail=f"columns {sorted(flagged)}" if flagged else "",
        ),
        Check(
            "external_residual",
            external,
            t.max_external_residual_cm,
            external <= t.max_external_residual_cm,
        ),
    ]

    pairs = list(combinations(sorted(freqs), 2))
    freq_change = 0.0
    imag_change = 0.0
    for h1, h2 in pairs:
        f1, f2 = freqs[h1], freqs[h2]
        stable = (f1 > STABLE_MODE_FLOOR_CM) & (f2 > STABLE_MODE_FLOOR_CM)
        if np.any(stable):
            freq_change = max(freq_change, float(np.max(np.abs(f1[stable] - f2[stable]))))
        imag = (f1 < -t.imaginary_threshold_cm) & (f2 < -t.imaginary_threshold_cm)
        if np.any(imag):
            rel = np.abs(f1[imag] - f2[imag]) / np.abs(f1[imag])
            imag_change = max(imag_change, float(np.max(rel)))
    evaluated = bool(pairs)
    note = "" if evaluated else "needs Hessians at two or more accepted steps"
    checks.append(
        Check(
            "frequency_stability",
            freq_change,
            t.max_frequency_change_cm,
            evaluated and freq_change <= t.max_frequency_change_cm,
            evaluated=evaluated,
            detail=note,
        )
    )
    checks.append(
        Check(
            "imaginary_stability",
            imag_change,
            t.max_imaginary_change_fraction,
            evaluated and imag_change <= t.max_imaginary_change_fraction,
            evaluated=evaluated,
            detail=note,
        )
    )

    if delta_g_by_precision:
        values = list(delta_g_by_precision.values())
        spread = max(values) - min(values)
        checks.append(
            Check(
                "precision_delta_g",
                spread,
                t.max_delta_g_change_kcal_mol,
                spread <= t.max_delta_g_change_kcal_mol,
                required=False,
                detail=f"modes {sorted(delta_g_by_precision)}",
            )
        )
    else:
        checks.append(
            Check(
                "precision_delta_g",
                0.0,
                t.max_delta_g_change_kcal_mol,
                False,
                evaluated=False,
                required=False,
                detail="no precision ladder supplied",
            )
        )

    return QualityReport(tuple(checks), tuple(sorted(flagged)), freqs)
