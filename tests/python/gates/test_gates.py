import numpy as np
import pytest

from saddle import units
from saddle.analysis.hessian_quality import Check, QualityReport
from saddle.gates import gates as G
from saddle.gates.gates import Evidence

# SN2 geometry at the TS: Cl(1) ... C(2) ... Cl(6), H 3-5 planar.
XYZ_TS = (
    np.array(
        [[-2.3, 0, 0], [0, 0, 0], [0, 1.08, 0], [0, -0.54, 0.935], [0, -0.54, -0.935], [2.3, 0, 0]]
    ).ravel()
    / units.BOHR_TO_ANGSTROM
)
MAP = (1, 2, 3, 4, 5, 6)
BREAKING = ((2, 6),)
FORMING = ((1, 2),)


def _mode_sn2() -> np.ndarray:
    """Antisymmetric Cl-C-Cl stretch with carbon moving toward Cl(1)."""
    m = np.zeros(18)
    m[0] = 0.2  # Cl1
    m[3] = -0.9  # C
    m[15] = 0.2  # Cl6
    return m / np.linalg.norm(m)


def _mode_torsion() -> np.ndarray:
    """H atoms rotate about the x axis; no bond lengths change to first order."""
    m = np.zeros(18)
    coords = XYZ_TS.reshape(6, 3)
    for i in (2, 3, 4):
        m[3 * i : 3 * i + 3] = np.cross([1.0, 0.0, 0.0], coords[i])
    return m / np.linalg.norm(m)


def _clean_quality() -> QualityReport:
    return QualityReport((Check("asymmetry", 1e-6, 1e-4, True),), ())


R_GRAPH = {frozenset(b) for b in [(2, 3), (2, 4), (2, 5), (2, 6)]}
P_GRAPH = {frozenset(b) for b in [(2, 3), (2, 4), (2, 5), (1, 2)]}


def good() -> Evidence:
    return Evidence(
        gradient=np.full(18, 1e-6),
        scf_converged=True,
        orbitals_stable=True,
        uses_pcm=False,
        pcm_converged=None,
        hessian_quality=_clean_quality(),
        frequencies_cm=np.array([-450.0, 80.0, 200.0, 900.0]),
        noise_floor_cm=8.0,
        ts_mode_cartesian=_mode_sn2(),
        coords_bohr=XYZ_TS,
        atom_map=MAP,
        breaking=BREAKING,
        forming=FORMING,
        reactant_graph=R_GRAPH,
        expected_product_graph=P_GRAPH,
        irc_forward_graph=P_GRAPH,
        irc_reverse_graph=R_GRAPH,
        declared_competitors=("anti_e2",),
        competitors_evaluated={"anti_e2": True},
        baseline_verdict="SUPPORTED",
        robustness_verdicts={
            "grid": "SUPPORTED",
            "precision": "SUPPORTED",
            "functional": "SUPPORTED",
        },
    )


def test_all_gates_pass_on_good_evidence() -> None:
    results = G.evaluate_all(good())
    assert [r.gate for r in results] == ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_g1_thresholds_and_failure() -> None:
    assert G.g1_stationarity(good()).passed
    e = good().replace(gradient=np.full(18, 1.2e-5))  # rms > 1e-5
    r = G.g1_stationarity(e)
    assert not r.passed and r.failure_action == "refine_again"
    e = good().replace(gradient=np.r_[np.zeros(17), 2e-5])  # max > 1.5e-5, rms small
    assert not G.g1_stationarity(e).passed


@pytest.mark.parametrize(
    "change",
    [
        {"scf_converged": False},
        {"orbitals_stable": False},
        {"uses_pcm": True, "pcm_converged": False},
    ],
)
def test_g2_failures(change: dict[str, object]) -> None:
    r = G.g2_electronic(good().replace(**change))
    assert not r.passed and r.failure_action == "change_guess_or_reject"


def test_g2_pcm_flag_required_when_pcm_used() -> None:
    r = G.g2_electronic(good().replace(uses_pcm=True, pcm_converged=None))
    assert not r.passed and "missing" in r.reason


def test_g3_fails_on_bad_quality_report() -> None:
    bad = QualityReport((Check("asymmetry", 1e-2, 1e-4, False),), (4,))
    r = G.g3_hessian(good().replace(hessian_quality=bad))
    assert not r.passed and r.failure_action == "tighten_and_recompute"
    assert "asymmetry" in r.reason


def test_g4_requires_exactly_one_imaginary_mode() -> None:
    r = G.g4_saddle_order(good().replace(frequencies_cm=np.array([50.0, 300.0])))
    assert not r.passed and "minimum" in r.reason
    r = G.g4_saddle_order(good().replace(frequencies_cm=np.array([-450.0, -300.0, 300.0])))
    assert not r.passed and "second-order" in r.reason
    assert r.failure_action == "displace_and_reoptimize"


def test_g4_ignores_noise_level_imaginary_modes() -> None:
    e = good().replace(frequencies_cm=np.array([-450.0, -6.0, 300.0]), noise_floor_cm=8.0)
    assert G.g4_saddle_order(e).passed


def test_g4_small_second_mode_needs_tight_recheck() -> None:
    e = good().replace(frequencies_cm=np.array([-450.0, -20.0, 300.0]), noise_floor_cm=8.0)
    r = G.g4_saddle_order(e)
    assert (
        not r.passed and "within 3x" in r.reason and r.failure_action == "displace_and_reoptimize"
    )
    assert G.g4_saddle_order(e.replace(second_mode_persists=False)).passed
    assert not G.g4_saddle_order(e.replace(second_mode_persists=True)).passed


def test_g5_accepts_reactive_mode_and_rejects_torsion() -> None:
    assert G.g5_mode_character(good()).passed
    r = G.g5_mode_character(good().replace(ts_mode_cartesian=_mode_torsion()))
    assert not r.passed and r.failure_action == "reject_wrong_saddle"


def test_g5_rejects_inconsistent_bond_change_signs() -> None:
    m = np.zeros(18)
    m[0], m[15] = -0.5, 0.5  # both Cl move away: both bonds stretch together
    r = G.g5_mode_character(good().replace(ts_mode_cartesian=m / np.linalg.norm(m)))
    assert not r.passed and "sign" in r.reason


def test_g6_connectivity() -> None:
    assert G.g6_connectivity(
        good().replace(irc_forward_graph=R_GRAPH, irc_reverse_graph=P_GRAPH)
    ).passed
    r = G.g6_connectivity(good().replace(irc_forward_graph=R_GRAPH))
    assert not r.passed and r.failure_action == "relabel_or_reject"


def test_g6_identifies_other_declared_pathway() -> None:
    e2_products = {frozenset(b) for b in [(2, 3), (2, 4), (1, 5)]}
    e = good().replace(
        irc_forward_graph=e2_products, alternative_product_graphs={"anti_e2": e2_products}
    )
    r = G.g6_connectivity(e)
    assert not r.passed and "anti_e2" in r.reason


def test_g7_competition() -> None:
    r = G.g7_competition(good().replace(competitors_evaluated={}))
    assert not r.passed and r.failure_action == "mark_incomplete"
    assert G.g7_competition(
        good().replace(declared_competitors=(), competitors_evaluated={})
    ).passed


def test_g8_robustness() -> None:
    e = good().replace(
        robustness_verdicts={
            "grid": "SUPPORTED",
            "precision": "INCONCLUSIVE",
            "functional": "SUPPORTED",
        }
    )
    r = G.g8_robustness(e)
    assert not r.passed and r.failure_action == "mark_inconclusive" and "precision" in r.reason
    r = G.g8_robustness(good().replace(robustness_verdicts={"grid": "SUPPORTED"}))
    assert not r.passed and "missing" in r.reason


@pytest.mark.parametrize(
    ("gate", "field"),
    [
        (G.g1_stationarity, "gradient"),
        (G.g2_electronic, "scf_converged"),
        (G.g3_hessian, "hessian_quality"),
        (G.g4_saddle_order, "frequencies_cm"),
        (G.g5_mode_character, "ts_mode_cartesian"),
        (G.g6_connectivity, "irc_forward_graph"),
        (G.g7_competition, "competitors_evaluated"),
        (G.g8_robustness, "baseline_verdict"),
    ],
)
def test_missing_evidence_fails(gate, field) -> None:  # type: ignore[no-untyped-def]
    r = gate(good().replace(**{field: None}))
    assert not r.passed
    assert "missing" in r.reason


def test_g1_uses_internal_gradient_and_reports_removed_torque() -> None:
    raw = np.full(18, 1e-6)
    raw[0] += 3e-5  # spurious torque-like component removed by projection
    r = G.g1_stationarity(good().replace(raw_gradient=raw))
    assert r.passed and "removed" in r.reason
