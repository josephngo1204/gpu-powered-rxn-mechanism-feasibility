"""Spec §11 negative controls repeated with real electronic structure (cuEST).

The fake-surface versions live in test_negative_controls.py. These controls
need an SCF: forced PCM non-convergence and a torsional saddle submitted as a
reaction TS.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cupy")

from saddle import _core, elements, units
from saddle.gates import gates as G
from saddle.graphs import bond_graph, expected_products
from saddle.hypothesis import Pathway
from saddle.workflows.models import SystemInfo, build_provider

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]

WATER = ("O", "H", "H")
WATER_BOHR = (
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]).ravel()
    / units.BOHR_TO_ANGSTROM
)


def test_forced_pcm_non_convergence_fails_g2() -> None:
    info = SystemInfo(WATER, (WATER,), (0,), 0, "water")
    # One PCM iteration cannot reach the 1e-10 charge residual.
    spec = {"type": "cuest", "functional": "HF", "basis": "def2-SVP", "pcm": {"max_iterations": 1}}
    provider = build_provider(spec, None, info)
    try:
        g2 = provider.g2_evidence(provider.system(), WATER_BOHR)
    finally:
        provider.close()
    assert g2["uses_pcm"] and g2["pcm_converged"] is False
    assert "PCM" in g2["note"]
    e = G.Evidence(
        scf_converged=g2["scf_converged"],
        orbitals_stable=g2["orbitals_stable"],
        uses_pcm=g2["uses_pcm"],
        pcm_converged=g2["pcm_converged"],
    )
    r = G.g2_electronic(e)
    assert not r.passed and "PCM not converged" in r.reason


def eclipsed_ethane_bohr() -> np.ndarray:
    """D3h eclipsed ethane: the torsional saddle (atoms C1 C2 H3-5 on C1, H6-8 on C2)."""
    rcc, rch, angle = 1.54, 1.09, np.deg2rad(111.0)
    x = [[0.0, 0.0, -rcc / 2], [0.0, 0.0, rcc / 2]]
    for sign in (-1.0, 1.0):
        for k in range(3):
            phi = 2.0 * np.pi * k / 3.0
            x.append(
                [
                    rch * np.sin(angle) * np.cos(phi),
                    rch * np.sin(angle) * np.sin(phi),
                    sign * (rcc / 2 - rch * np.cos(angle)),
                ]
            )
    return np.asarray(x).ravel() / units.BOHR_TO_ANGSTROM


def torsion_direction(x: np.ndarray) -> np.ndarray:
    """Counter-rotate the two methyl groups about the C-C (z) axis."""
    xyz = x.reshape(-1, 3)
    d = np.zeros_like(xyz)
    for i, sign in ((2, 1.0), (3, 1.0), (4, 1.0), (5, -1.0), (6, -1.0), (7, -1.0)):
        d[i] = sign * np.array([-xyz[i, 1], xyz[i, 0], 0.0])
    return d.ravel() / np.linalg.norm(d)


@pytest.mark.slow
def test_ethane_torsional_ts_as_reaction_ts_fails_g5_and_g6() -> None:
    """Spec §11: the eclipsed-ethane saddle has one imaginary mode (passes G4)
    but is not a 1,2-H shift TS: the mode carries no reactive bond change (G5)
    and the IRC returns staggered ethane on both sides (G6)."""
    symbols = ("C", "C", "H", "H", "H", "H", "H", "H")
    atom_map = tuple(range(1, 9))
    z = [elements.atomic_number(s) for s in symbols]
    masses = np.array([elements.isotope_mass(s) for s in symbols])
    info = SystemInfo(symbols, (symbols,), (0,), 0, None)
    provider = build_provider(
        {"type": "cuest", "functional": "HF", "basis": "def2-SVP"}, None, info
    )
    model = provider.system()
    try:
        guess = eclipsed_ethane_bohr()
        ts = _core.saddle_search(
            model, guess, z, follow_direction=torsion_direction(guess), gtol_rms=1e-6, gtol_max=2e-6
        )
        assert ts["converged"]
        x = np.asarray(ts["coords"])
        h = 5e-3
        points = _core.central_stencil(x, h)
        raw, _, _ = _core.assemble_central([model.evaluate(q)[1] for _, _, q in points], h)
        hess = 0.5 * (np.asarray(raw) + np.asarray(raw).T)
        vib = _core.analyze_vibrations(hess, x, masses)
        _, gradient = model.evaluate(x)
        path = _core.irc(model, x, z, hess, masses, 0.05, 300, 1e-3)
        ends = {
            b: bond_graph(symbols, np.asarray(path[b]["endpoint"]["coords"]), atom_map)
            for b in ("forward", "reverse")
        }
    finally:
        provider.close()

    claimed = Pathway("h_shift", breaking=((1, 3),), forming=((2, 3),))
    reactant = bond_graph(symbols, x, atom_map)
    e = G.Evidence(
        gradient=np.asarray(_core.internal_gradient(x, z, np.asarray(gradient))),
        frequencies_cm=np.asarray(vib.frequencies_cm),
        noise_floor_cm=5.0,
        ts_mode_cartesian=np.asarray(vib.modes_cartesian)[:, 0],
        coords_bohr=x,
        atom_map=atom_map,
        breaking=claimed.breaking,
        forming=claimed.forming,
        reactant_graph=reactant,
        expected_product_graph=expected_products(reactant, claimed),
        irc_forward_graph=ends["forward"],
        irc_reverse_graph=ends["reverse"],
    )
    assert G.g1_stationarity(e).passed
    g4 = G.g4_saddle_order(e)
    assert g4.passed, g4.reason
    assert float(np.sort(vib.frequencies_cm)[0]) < -100.0  # a real torsional barrier mode
    g5, g6 = G.g5_mode_character(e), G.g6_connectivity(e)
    assert not g5.passed, g5.reason
    assert not g6.passed and ends["forward"] == ends["reverse"] == reactant, g6.reason


HCN_SYMBOLS = ("H", "C", "N")
# Spec §11 rotated-structure control: the decision limit is the scenario, not a
# gate threshold. It sits between the canonical and rotated PBE/(20,26)
# barriers measured in development (60.4 and 45.7 kcal/mol), so a grid
# orientation artifact decides the verdict and G8 must catch it.
ROTATION_CONTROL_LIMIT = 53.0


def _barrier_verdicts(grid: str) -> tuple[str, str, float]:
    from saddle.analysis.uncertainty import Term, UncertaintyBudget
    from saddle.gates.verdict import PathwayOutcome, decide
    from saddle.scf.model import CuestModel, LevelOfTheory
    from saddle.workflows.pipeline import ROTATION_SEED, _random_rotation, _rotate

    species = ROOT / "campaigns" / "smoke" / "species"
    geo = {
        k: np.loadtxt(species / f, skiprows=2, usecols=(1, 2, 3)).ravel() / units.BOHR_TO_ANGSTROM
        for k, f in (("min", "hcn.xyz"), ("ts", "ts_hcn_hnc.xyz"))
    }
    model = CuestModel(LevelOfTheory("PBE", "def2-SVP", grid=grid), HCN_SYMBOLS)
    barriers = []
    try:
        for rot in (np.eye(3), _random_rotation(ROTATION_SEED)):
            e = {
                k: model.run(_rotate(v, rot), gradient=False, warm=False).energy
                for k, v in geo.items()
            }
            barriers.append(units.hartree_to_kcal_mol(e["ts"] - e["min"]))
    finally:
        model.close()
    # sigma about 1 kcal/mol; the grid term is the quantity under test, so it is
    # not inflated to absorb the orientation shift.
    budget = UncertaintyBudget(
        tuple(
            Term(c, v, "benchmark" if c == "method" else "measured")
            for c, v in [
                ("numerical", 0.1),
                ("grid", 0.1),
                ("basis", 0.5),
                ("method", 0.8),
                ("thermo", 0.2),
                ("solvent", 0.0),
            ]
        )
    )
    passing = [G.GateResult("G1", "stationarity", True, "stand-in", None)]
    verdicts = [
        decide([PathwayOutcome("isomerization", passing, b, budget)], ROTATION_CONTROL_LIMIT)[
            "isomerization"
        ].verdict.value
        for b in barriers
    ]
    return verdicts[0], verdicts[1], abs(barriers[1] - barriers[0])


def _g8(baseline: str, rotation: str) -> G.GateResult:
    robust = {"grid": baseline, "precision": baseline, "functional": baseline, "rotation": rotation}
    return G.g8_robustness(G.Evidence(baseline_verdict=baseline, robustness_verdicts=robust))


def test_rotated_structure_on_coarse_grid_fails_g8() -> None:
    baseline, rotated, shift = _barrier_verdicts("(20,26)")
    assert shift > 5.0  # the orientation artifact is real on a coarse grid
    g8 = _g8(baseline, rotated)
    assert not g8.passed and "rotation" in g8.reason


def test_rotated_structure_on_converged_grid_passes_g8() -> None:
    baseline, rotated, shift = _barrier_verdicts("(75,302)")
    assert shift < 0.05
    assert _g8(baseline, rotated).passed
