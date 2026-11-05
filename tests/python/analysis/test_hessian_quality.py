import numpy as np
import numpy.typing as npt
import pytest

from saddle import _core
from saddle.analysis.hessian_quality import HessianTargets, assess_hessians

FloatArray = npt.NDArray[np.float64]

COORDS = np.array([0.0, 0.0, 0.0, 1.43, 1.11, 0.0, -1.43, 1.11, 0.0])
MASSES = np.array([15.99491461957, 1.00782503223, 1.00782503223])


def _dist(i: int, j: int) -> float:
    return float(np.linalg.norm(COORDS[3 * i : 3 * i + 3] - COORDS[3 * j : 3 * j + 3]))


def _model(k_hh: float = 0.1) -> _core.Springs:
    return _core.Springs(
        [(0, 1, 0.5, _dist(0, 1)), (0, 2, 0.5, _dist(0, 2)), (1, 2, k_hh, _dist(1, 2))]
    )


def _raw(step: float, k_hh: float = 0.1) -> FloatArray:
    model = _model(k_hh)
    points = _core.central_stencil(COORDS, step)
    grads = [model.evaluate(c)[1] for _, _, c in points]
    raw, _, _ = _core.assemble_central(grads, step)
    return np.asarray(raw)


def _check(report, name):  # type: ignore[no-untyped-def]
    return next(c for c in report.checks if c.name == name)


def test_clean_hessians_pass_all_targets() -> None:
    report = assess_hessians({1e-3: _raw(1e-3), 2e-3: _raw(2e-3)}, COORDS, MASSES)
    assert report.passed, report.summary()
    assert report.flagged_columns == ()


def test_targets_match_spec_section_6() -> None:
    t = HessianTargets()
    assert t.max_asymmetry == 1e-4
    assert t.max_external_residual_cm == 5.0
    assert t.max_frequency_change_cm == 1.0
    assert t.max_imaginary_change_fraction == 0.02
    assert t.max_delta_g_change_kcal_mol == 0.1


def test_injected_noise_fails_asymmetry() -> None:
    rng = np.random.default_rng(0)
    raw = _raw(1e-3) + 1e-3 * rng.standard_normal((9, 9))
    report = assess_hessians({1e-3: raw}, COORDS, MASSES)
    assert not report.passed
    assert not _check(report, "asymmetry").passed


def test_column_spike_is_flagged() -> None:
    raw = _raw(1e-3)
    raw[:, 4] += 0.05
    report = assess_hessians({1e-3: raw}, COORDS, MASSES)
    assert not report.passed
    assert 4 in report.flagged_columns
    assert not _check(report, "column_spikes").passed


def test_broken_translation_invariance_fails_translation_residual() -> None:
    raw = _raw(1e-3) + 1e-3 * np.eye(9)
    report = assess_hessians({1e-3: raw}, COORDS, MASSES)
    assert not _check(report, "translation_residual").passed
    assert not report.passed


def test_rotational_noninvariance_is_reported_but_does_not_fail() -> None:
    """A DFT grid breaks rotational invariance only (user decision: split TR check)."""
    sq = np.repeat(np.sqrt(MASSES), 3)
    x = COORDS.reshape(-1, 3)
    r = x - (MASSES[:, None] * x).sum(0) / MASSES.sum()
    rot = np.zeros((9, 3))
    for i in range(3):
        for a in range(3):
            rot[3 * i : 3 * i + 3, a] = np.sqrt(MASSES[i]) * np.cross(np.eye(3)[a], r[i])
    q, _ = np.linalg.qr(rot)
    bump = 2e-6 * (q @ q.T)  # mass-weighted rotational curvature only
    perturbed = {h: _raw(h) + np.outer(sq, sq) * bump for h in (1e-3, 2e-3)}
    report = assess_hessians(perturbed, COORDS, MASSES)
    assert _check(report, "translation_residual").passed
    rot_check = _check(report, "rotation_residual")
    assert rot_check.informational and rot_check.value > 5.0
    assert report.passed, report.summary()


def test_step_dependence_fails_frequency_stability() -> None:
    shifted = _raw(2e-3, k_hh=0.103)
    report = assess_hessians({1e-3: _raw(1e-3), 2e-3: shifted}, COORDS, MASSES)
    assert not _check(report, "frequency_stability").passed


def test_imaginary_frequency_stability() -> None:
    a, b = _raw(1e-3, -0.05), _raw(2e-3, -0.0525)
    report = assess_hessians({1e-3: a, 2e-3: b}, COORDS, MASSES)
    check = _check(report, "imaginary_stability")
    fa = _core.analyze_vibrations(0.5 * (a + a.T), COORDS, MASSES).frequencies_cm[0]
    fb = _core.analyze_vibrations(0.5 * (b + b.T), COORDS, MASSES).frequencies_cm[0]
    assert fa < 0 and fb < 0
    assert check.value == pytest.approx(abs(fa - fb) / abs(fa), rel=1e-9)
    assert check.value > 0.02
    assert not check.passed


def test_precision_sensitivity_of_barrier() -> None:
    report = assess_hessians(
        {1e-3: _raw(1e-3)},
        COORDS,
        MASSES,
        delta_g_by_precision={"native-fp64": 20.00, "default": 20.15},
    )
    assert not _check(report, "precision_delta_g").passed


def test_single_step_marks_stability_as_not_evaluated() -> None:
    report = assess_hessians({1e-3: _raw(1e-3)}, COORDS, MASSES)
    check = _check(report, "frequency_stability")
    assert check.evaluated is False
    assert report.passed is False, "a Hessian with unevaluated checks cannot pass"


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        assess_hessians({}, COORDS, MASSES)


def test_near_zero_imaginary_modes_are_not_compared() -> None:
    # Linear A-H-C saddle: bends sit at about -0.5 and -1.0 cm^-1 for the two steps.
    coords = np.array([0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 5.0, 0.0, 0.0])
    masses = np.array([12.0, 1.00782503223, 12.0])
    model = _core.Springs([(0, 1, -0.06, 2.5), (1, 2, -0.06, 2.5), (0, 2, 0.4, 5.0)])

    def raw(step: float) -> FloatArray:
        pts = _core.central_stencil(coords, step)
        r, _, _ = _core.assemble_central([model.evaluate(c)[1] for _, _, c in pts], step)
        return np.asarray(r)

    raws = {1e-3: raw(1e-3), 2e-3: raw(2e-3)}
    assert assess_hessians(raws, coords, masses, noise_floor_cm=5.0).passed
    assert assess_hessians(raws, coords, masses).passed
    strict = assess_hessians(raws, coords, masses, noise_floor_cm=0.1)
    assert not _check(strict, "imaginary_stability").passed
