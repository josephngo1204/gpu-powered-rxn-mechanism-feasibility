"""Hessian quality report against the spec §6 targets (feeds gate G3).

Targets come from spec §6. Changing them is an "Ask first" item (spec §12).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import numpy.typing as npt

from saddle import _core, units

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
    # Imaginary modes smaller than this are numerical zeros, not compared.
    # PROPOSED fallback (spec gives none); pass the measured noise floor instead.
    imaginary_threshold_cm: float = 10.0


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    target: float
    passed: bool
    evaluated: bool = True
    required: bool = True
    detail: str = ""
    # Reported but never decides pass/fail (rotational residual of a DFT grid).
    informational: bool = False


@dataclass(frozen=True)
class QualityReport:
    checks: tuple[Check, ...]
    flagged_columns: tuple[int, ...]
    frequencies_cm: Mapping[float, FloatArray] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        decisive = [c for c in self.checks if not c.informational]
        return all(c.passed and c.evaluated for c in decisive if c.required) and all(
            c.passed for c in decisive if c.evaluated
        )

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            if not c.evaluated:
                status = "NOT EVALUATED" + ("" if c.required else " (optional)")
            elif c.informational:
                status = "INFO"
            lines.append(f"{c.name}: {status} value={c.value:.3g} target={c.target:.3g} {c.detail}")
        return "\n".join(lines)


def external_residuals(
    hessian: FloatArray, coords: FloatArray, masses: FloatArray
) -> tuple[float, float]:
    """Largest translational and rotational frequencies (cm^-1) of the unprojected Hessian.

    Translations are exactly invariant for cuEST models (the grid moves with
    the atoms); rotations are not for DFT, because integration grids are not
    rotationally invariant (spec §6 "DFT grid: orientation dependence").
    """
    m = np.asarray(masses, dtype=np.float64)
    x = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    sq = np.repeat(np.sqrt(m), 3)
    hmw = np.asarray(hessian) / np.outer(sq, sq)
    r = x - (m[:, None] * x).sum(axis=0) / m.sum()
    n = len(m)
    t = np.zeros((3 * n, 3))
    rot = np.zeros((3 * n, 3))
    for i in range(n):
        for a in range(3):
            t[3 * i + a, a] = np.sqrt(m[i])
            rot[3 * i : 3 * i + 3, a] = np.sqrt(m[i]) * np.cross(np.eye(3)[a], r[i])

    def top(basis: FloatArray) -> float:
        u, sv, _ = np.linalg.svd(basis, full_matrices=False)
        keep = u[:, sv > 1e-8 * sv.max()] if sv.max() > 0 else u[:, :0]
        if keep.shape[1] == 0:
            return 0.0
        w = np.linalg.eigvalsh(keep.T @ hmw @ keep)
        return float(
            np.sqrt(np.abs(w).max() / units.AMU_TO_ELECTRON_MASS) * units.HARTREE_TO_WAVENUMBER
        )

    return top(t), top(rot)


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


@dataclass(frozen=True)
class _StepMetrics:
    """Worst-case single-Hessian metrics over all steps, and each step's frequencies."""

    asymmetry: float
    translation_cm: float
    rotation_cm: float
    flagged_columns: frozenset[int]
    frequencies_cm: dict[float, FloatArray]


def _step_metrics(
    raw_by_step: Mapping[float, FloatArray],
    coords: FloatArray,
    masses: FloatArray,
    t: HessianTargets,
) -> _StepMetrics:
    asymmetry = translation = rotation = 0.0
    flagged: set[int] = set()
    freqs: dict[float, FloatArray] = {}
    for step, raw_in in raw_by_step.items():
        raw = np.asarray(raw_in, dtype=np.float64)
        norm = float(np.linalg.norm(raw))
        a = float(np.linalg.norm(raw - raw.T) / norm) if norm > 0 else 0.0
        asymmetry = max(asymmetry, a)
        if a > t.max_asymmetry:
            flagged.update(_spiky_columns(raw))
        sym = 0.5 * (raw + raw.T)
        t_res, r_res = external_residuals(sym, coords, masses)
        translation = max(translation, t_res)
        rotation = max(rotation, r_res)
        freqs[step] = np.asarray(_core.analyze_vibrations(sym, coords, masses).frequencies_cm)
    return _StepMetrics(asymmetry, translation, rotation, frozenset(flagged), freqs)


def _single_hessian_checks(m: _StepMetrics, t: HessianTargets) -> list[Check]:
    flagged = sorted(m.flagged_columns)
    return [
        Check("asymmetry", m.asymmetry, t.max_asymmetry, m.asymmetry <= t.max_asymmetry),
        Check(
            "column_spikes",
            float(len(flagged)),
            0.0,
            not flagged,
            detail=f"columns {flagged}" if flagged else "",
        ),
        Check(
            "translation_residual",
            m.translation_cm,
            t.max_external_residual_cm,
            m.translation_cm <= t.max_external_residual_cm,
        ),
        Check(
            "rotation_residual",
            m.rotation_cm,
            t.max_external_residual_cm,
            m.rotation_cm <= t.max_external_residual_cm,
            informational=True,
            detail="grid orientation error; judged by the G8 rotation test (user decision)",
        ),
    ]


def _step_stability_checks(
    freqs: Mapping[float, FloatArray], t: HessianTargets, imag_floor: float
) -> list[Check]:
    """Frequencies must not move between FD steps: stiff modes in cm^-1, imaginary in fraction."""
    pairs = list(combinations(sorted(freqs), 2))
    freq_change = imag_change = 0.0
    for h1, h2 in pairs:
        f1, f2 = freqs[h1], freqs[h2]
        stable = (f1 > STABLE_MODE_FLOOR_CM) & (f2 > STABLE_MODE_FLOOR_CM)
        if np.any(stable):
            freq_change = max(freq_change, float(np.max(np.abs(f1[stable] - f2[stable]))))
        imag = (f1 < -imag_floor) & (f2 < -imag_floor)
        if np.any(imag):
            rel = np.abs(f1[imag] - f2[imag]) / np.abs(f1[imag])
            imag_change = max(imag_change, float(np.max(rel)))
    evaluated = bool(pairs)
    note = "" if evaluated else "needs Hessians at two or more accepted steps"
    return [
        Check(
            "frequency_stability",
            freq_change,
            t.max_frequency_change_cm,
            evaluated and freq_change <= t.max_frequency_change_cm,
            evaluated=evaluated,
            detail=note,
        ),
        Check(
            "imaginary_stability",
            imag_change,
            t.max_imaginary_change_fraction,
            evaluated and imag_change <= t.max_imaginary_change_fraction,
            evaluated=evaluated,
            detail=note,
        ),
    ]


def _precision_check(delta_g_by_precision: Mapping[str, float] | None, t: HessianTargets) -> Check:
    """Optional: the barrier must not move across math modes (precision ladder)."""
    if not delta_g_by_precision:
        return Check(
            "precision_delta_g",
            0.0,
            t.max_delta_g_change_kcal_mol,
            False,
            evaluated=False,
            required=False,
            detail="no precision ladder supplied",
        )
    values = list(delta_g_by_precision.values())
    spread = max(values) - min(values)
    return Check(
        "precision_delta_g",
        spread,
        t.max_delta_g_change_kcal_mol,
        spread <= t.max_delta_g_change_kcal_mol,
        required=False,
        detail=f"modes {sorted(delta_g_by_precision)}",
    )


def assess_hessians(
    raw_by_step: Mapping[float, FloatArray],
    coords_bohr: FloatArray,
    masses_amu: FloatArray,
    *,
    delta_g_by_precision: Mapping[str, float] | None = None,
    targets: HessianTargets | None = None,
    noise_floor_cm: float | None = None,
) -> QualityReport:
    """Assess unsymmetrized FD Hessians computed at one or more accepted steps.

    `noise_floor_cm` (from the noise probe) sets which imaginary modes are
    real enough to compare; without it the targets' fallback is used.
    """
    if not raw_by_step:
        raise ValueError("assess_hessians needs at least one Hessian")
    t = targets or HessianTargets()
    imag_floor = t.imaginary_threshold_cm if noise_floor_cm is None else noise_floor_cm
    coords = np.asarray(coords_bohr, dtype=np.float64)
    masses = np.asarray(masses_amu, dtype=np.float64)
    metrics = _step_metrics(raw_by_step, coords, masses, t)
    checks = [
        *_single_hessian_checks(metrics, t),
        *_step_stability_checks(metrics.frequencies_cm, t, imag_floor),
        _precision_check(delta_g_by_precision, t),
    ]
    return QualityReport(
        tuple(checks), tuple(sorted(metrics.flagged_columns)), metrics.frequencies_cm
    )
