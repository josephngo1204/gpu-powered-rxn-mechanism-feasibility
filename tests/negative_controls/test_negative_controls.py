"""Spec §11 negative controls on fake surfaces.

Each control is a convincing wrong answer that must be rejected for the
stated reason. Controls that need real electronic structure (PCM
non-convergence, rotated structure on a coarse grid) use evidence-level
stand-ins here and are repeated with cuEST in Phase F.
"""

import dataclasses

import numpy as np
import numpy.typing as npt

from saddle import _core
from saddle.analysis.hessian_quality import assess_hessians
from saddle.analysis.uncertainty import Term, UncertaintyBudget
from saddle.gates import gates as G
from saddle.gates.verdict import PathwayOutcome, Verdict, decide, rank_pair

FloatArray = npt.NDArray[np.float64]

# Collinear A(0)-H(1)-C(2) hydrogen transfer at the spring rest geometry (bohr).
# The antisymmetric H motion is the only negative direction when the A...C
# spring holds the symmetric stretch: k_AH + k_HC + 4 k_AC > 0.
COORDS = np.array([0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 5.0, 0.0, 0.0])
MASSES = np.array([12.0, 1.00782503223, 12.0])
MAP = (1, 2, 3)
NOISE_FLOOR_CM = 5.0


def _r(i: int, j: int) -> float:
    return float(np.linalg.norm(COORDS[3 * i : 3 * i + 3] - COORDS[3 * j : 3 * j + 3]))


def springs(k01: float, k12: float, k02: float) -> _core.Springs:
    return _core.Springs([(0, 1, k01, _r(0, 1)), (1, 2, k12, _r(1, 2)), (0, 2, k02, _r(0, 2))])


def raw_hessian(model: _core.Springs, coords: FloatArray, step: float) -> FloatArray:
    points = _core.central_stencil(coords, step)
    raw, _, _ = _core.assemble_central([model.evaluate(c)[1] for _, _, c in points], step)
    return np.asarray(raw)


def evidence_for(model: _core.Springs, coords: FloatArray = COORDS) -> G.Evidence:
    """Run the numerical pipeline and collect evidence for a candidate TS."""
    _, gradient = model.evaluate(coords)
    raws = {1e-3: raw_hessian(model, coords, 1e-3), 2e-3: raw_hessian(model, coords, 2e-3)}
    quality = assess_hessians(raws, coords, MASSES, noise_floor_cm=NOISE_FLOOR_CM)
    vib = _core.analyze_vibrations(0.5 * (raws[1e-3] + raws[1e-3].T), coords, MASSES)
    reactant = {frozenset((1, 2)), frozenset((1, 3))}
    product = {frozenset((2, 3)), frozenset((1, 3))}
    return G.Evidence(
        gradient=np.asarray(gradient),
        scf_converged=True,
        orbitals_stable=True,
        uses_pcm=False,
        hessian_quality=quality,
        frequencies_cm=np.asarray(vib.frequencies_cm),
        noise_floor_cm=NOISE_FLOOR_CM,
        ts_mode_cartesian=np.asarray(vib.modes_cartesian)[:, 0],
        coords_bohr=coords,
        atom_map=MAP,
        breaking=((1, 2),),
        forming=((2, 3),),
        reactant_graph=reactant,
        expected_product_graph=product,
        irc_forward_graph=product,
        irc_reverse_graph=reactant,
        declared_competitors=(),
        competitors_evaluated={},
        baseline_verdict="SUPPORTED",
        robustness_verdicts={
            "grid": "SUPPORTED",
            "precision": "SUPPORTED",
            "functional": "SUPPORTED",
        },
    )


# H transfer from A to C: A-H breaks, H-C forms, A...C holds the frame.
TRUE_TS = springs(-0.06, -0.06, 0.4)


def failed(e: G.Evidence) -> list[str]:
    return [r.gate for r in G.evaluate_all(e) if not r.passed]


def test_positive_control_true_ts_passes_every_gate() -> None:
    e = evidence_for(TRUE_TS)
    results = G.evaluate_all(e)
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_second_order_saddle_fails_g4() -> None:
    # A repulsive A...C term makes the symmetric stretch a second negative direction.
    e = evidence_for(springs(-0.06, -0.06, -0.2))
    assert "G4" in failed(e)
    assert "second-order" in G.g4_saddle_order(e).reason


def test_near_stationary_geometry_with_residual_force_fails_g1() -> None:
    displaced = COORDS.copy()
    displaced[3] += 2e-3  # H off the saddle point along the transfer coordinate
    e = evidence_for(TRUE_TS, displaced)
    assert failed(e)[0] == "G1"


def test_wrong_products_fail_g6() -> None:
    e = evidence_for(TRUE_TS)
    e = e.replace(irc_forward_graph=e.reactant_graph)  # IRC returns to reactants both ways
    assert failed(e) == ["G6"]


def test_torsion_like_saddle_passes_g4_but_fails_g5_and_g6() -> None:
    e = evidence_for(TRUE_TS)
    coords = e.coords_bohr.reshape(3, 3)  # type: ignore[union-attr]
    torsion = np.concatenate([np.cross([0.0, 0.0, 1.0], c - coords.mean(axis=0)) for c in coords])
    rotamer = e.reactant_graph
    e = e.replace(
        ts_mode_cartesian=torsion / np.linalg.norm(torsion),
        irc_forward_graph=rotamer,
        irc_reverse_graph=rotamer,
    )
    assert G.g4_saddle_order(e).passed
    assert {"G5", "G6"} <= set(failed(e))


def test_injected_noise_fails_g3() -> None:
    e = evidence_for(TRUE_TS)
    rng = np.random.default_rng(7)
    raws = {
        h: raw_hessian(TRUE_TS, COORDS, h) + 5e-4 * rng.standard_normal((9, 9))
        for h in (1e-3, 2e-3)
    }
    e = e.replace(
        hessian_quality=assess_hessians(raws, COORDS, MASSES, noise_floor_cm=NOISE_FLOOR_CM)
    )
    assert "G3" in failed(e)


def test_column_spike_fails_g3_and_names_the_column() -> None:
    raws = {h: raw_hessian(TRUE_TS, COORDS, h) for h in (1e-3, 2e-3)}
    raws[1e-3][:, 5] += 0.02
    q = assess_hessians(raws, COORDS, MASSES, noise_floor_cm=NOISE_FLOOR_CM)
    r = G.g3_hessian(evidence_for(TRUE_TS).replace(hessian_quality=q))
    assert not r.passed and "5" in r.reason


def test_forced_pcm_non_convergence_fails_g2() -> None:
    e = evidence_for(TRUE_TS).replace(uses_pcm=True, pcm_converged=False)
    assert failed(e) == ["G2"]


def test_rotated_structure_on_coarse_grid_fails_g8() -> None:
    e = evidence_for(TRUE_TS).replace(
        robustness_verdicts={
            "grid": "SUPPORTED",
            "precision": "SUPPORTED",
            "functional": "SUPPORTED",
            "rotation": "OUTCOMPETED",
        }
    )
    assert failed(e) == ["G8"]


def test_ots_vs_oms_ranking_is_unresolved() -> None:
    budget = UncertaintyBudget(
        tuple(
            Term(c, v, "benchmark" if c == "method" else "measured")
            for c, v in [
                ("numerical", 0.05),
                ("grid", 0.05),
                ("basis", 0.1),
                ("method", 0.4),
                ("thermo", 0.2),
                ("solvent", 0.2),
            ]
        )
    )
    gates = G.evaluate_all(evidence_for(TRUE_TS))
    ots = PathwayOutcome("OTs", gates=gates, barrier_kcal_mol=20.15, uncertainty=budget)
    oms = dataclasses.replace(ots, name="OMs", barrier_kcal_mol=20.00)
    assert not rank_pair(ots, oms).resolved
    verdicts = decide([ots, oms], limit_kcal_mol=23.0)
    assert verdicts["OTs"].verdict is not Verdict.OUTCOMPETED
