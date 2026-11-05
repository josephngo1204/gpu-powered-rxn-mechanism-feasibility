"""End-to-end feasibility pipeline (spec §4 "Pipeline").

hypothesis -> reactant minimum -> RS-P-RFO saddle -> noise probe -> step
-> restartable Hessians at h and 2h -> quality -> vibrations -> IRC ->
graph match -> thermochemistry -> gates -> robustness -> verdict -> results

Separated reactants are minimized and treated species by species, so a
bimolecular barrier is measured from separated reactants with the per-species
standard-state term (spec §4). Fake surfaces describe only the whole system,
so they support single-species reactants; cuEST models evaluate fragments.

Reading order: ``run_campaign`` (top level) -> ``_run_variant`` (one model)
-> ``analyze_pathway`` (one TS), whose stages are the ``_stage`` helpers in
pipeline order. Model evaluations happen in a fixed order because cuEST SCF
warm-starts from the previous geometry.
"""

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from saddle import _core, units
from saddle.analysis import thermo, tunneling
from saddle.analysis.hessian_quality import QualityReport, assess_hessians
from saddle.analysis.kinetics import barrier_limit_kcal_mol
from saddle.analysis.uncertainty import COMPONENTS, Term, UncertaintyBudget
from saddle.errors import PCMNotConverged, SCFError
from saddle.gates import gates as G
from saddle.gates.verdict import PathwayOutcome, Verdict, VerdictResult, decide, rank_pair
from saddle.graphs import Graph, bond_graph, compare_graphs, expected_products
from saddle.hypothesis import Pathway
from saddle.store.geometry import geometry_hash
from saddle.store.provenance import NOT_APPLICABLE, Provenance, collect_environment
from saddle.store.results import write_result
from saddle.store.stencil import StencilStore, run_central_hessian
from saddle.workflows.campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    Atoms,
    Campaign,
    CampaignError,
    load_campaign,
)
from saddle.workflows.models import Model, ModelError, ModelProvider, build_provider

FloatArray = npt.NDArray[np.float64]

# Spec §6 design estimate of |E''''| used to choose the FD step.
FOURTH_DERIVATIVE_ESTIMATE = 2.0
# Below this the probe cannot resolve noise (clean analytic surfaces).
NOISE_RESOLUTION_FLOOR = 1e-10
MIN_STEP, MAX_STEP = 1e-4, 2e-2
# FD step for reactant-species Hessians (bohr); their noise is not probed.
SPECIES_STEP = 1e-3
# Reference-level refinement before a Hessian (spec §6 "Residual force"). At
# the G1 limit (|g| ~ 1e-5) the residual force alone gives ~14 cm^-1 of
# rotation-vibration mixing for hydrogen, above the 5 cm^-1 target.
# PROPOSED value (spec gives none); see docs/adr/0002.
REFERENCE_GTOL_RMS = 1e-7
REFERENCE_GTOL_MAX = 1.5e-7
# SCF orbital-gradient threshold for Hessian stencil gradients. At the 1e-8
# reference threshold, SCF incompleteness acts as ~2e-9 Eh/bohr gradient noise
# across stencil points (seen in the 1/h growth of the sum-rule error), which
# the line-scan probe misses because the error varies smoothly along a
# warm-started line. PROPOSED value; see docs/adr/0002.
HESSIAN_ORBITAL_GRADIENT = 1e-10
IRC_STEP = 0.05
IRC_MAX_POINTS = 300
IRC_SWITCH_GRADIENT = 1e-3
ROTATION_SEED = 20261105


# --------------------------------------------------------------------------- results


@dataclass
class StationaryPoint:
    coords: FloatArray
    energy: float
    hessian: FloatArray
    frequencies_cm: FloatArray
    thermo_qh: thermo.ThermoResult | None
    thermo_rrho: thermo.ThermoResult | None
    note: str = ""

    def g_total(self, quasi: bool) -> float | None:
        """Electronic energy plus the (q)RRHO free-energy correction, in Eh."""
        t = self.thermo_qh if quasi else self.thermo_rrho
        return None if t is None else self.energy + t.g_correction


@dataclass
class ReactantSet:
    """Separated reactant species; energies, free energies and ZPEs add up."""

    species: list[StationaryPoint]

    @property
    def energy(self) -> float:
        return float(sum(p.energy for p in self.species))

    @property
    def coords(self) -> FloatArray:
        """Concatenated per-species coordinates; species live in separate frames."""
        return np.concatenate([p.coords.ravel() for p in self.species])

    def graph(self, c: Campaign) -> Graph:
        """Union of each species' own bond graph (frames are independent)."""
        bonds: Graph = set()
        for i, point in enumerate(self.species):
            atoms = c.species(i)
            bonds |= bond_graph(atoms.symbols, point.coords, atoms.atom_map)
        return bonds

    def g_total(self, quasi: bool) -> float | None:
        parts = [p.g_total(quasi) for p in self.species]
        return (
            None if any(v is None for v in parts) else float(sum(v for v in parts if v is not None))
        )

    @property
    def zpe(self) -> float | None:
        if any(p.thermo_qh is None for p in self.species):
            return None
        return float(sum(p.thermo_qh.zpe for p in self.species if p.thermo_qh is not None))


@dataclass
class PathwayAnalysis:
    pathway: Pathway
    evidence: G.Evidence
    ts: StationaryPoint | None
    barrier_kcal_mol: float | None
    barrier_by_step: dict[str, float]
    numerical_sigma: float | None
    thermo_sigma: float | None
    noise_epsilon: float
    noise_floor_cm: float
    step: float
    tunneling: dict[str, Any]
    irc: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    n_gradients: int = 0
    completed: bool = True
    saddle_found: bool = True
    electronic_barrier_kcal_mol: float | None = None
    # Electronic energies of the IRC endpoint minima relative to separated reactants.
    endpoint_minima_kcal_mol: dict[str, float] = field(default_factory=dict)

    def uncertainty(self, campaign_terms: Mapping[str, Term]) -> UncertaintyBudget:
        terms = dict(campaign_terms)
        if self.numerical_sigma is not None:
            terms["numerical"] = Term(
                "numerical", self.numerical_sigma, "measured", "barrier at h vs 2h"
            )
        if self.thermo_sigma is not None:
            terms["thermo"] = Term("thermo", self.thermo_sigma, "measured", "qRRHO vs RRHO")
        return UncertaintyBudget(tuple(terms[c] for c in COMPONENTS if c in terms))


@dataclass(frozen=True)
class VariantOutcome:
    """A pathway's preliminary verdict under one robustness variant."""

    verdict: str
    barrier_kcal_mol: float | None
    reason: str


@dataclass
class CampaignResult:
    campaign: Campaign
    limit_kcal_mol: float
    analyses: dict[str, PathwayAnalysis]
    verdicts: dict[str, Any]
    summary: dict[str, Any]
    out_dir: Path


# --------------------------------------------------------------------------- helpers


def _model_fingerprint(spec: Mapping[str, Any], variant: str) -> str:
    text = json.dumps({"model": spec, "variant": variant}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _provenance(
    c: Campaign,
    coords: FloatArray,
    variant: str,
    env: Mapping[str, str],
    described: Mapping[str, Any] | None = None,
) -> Provenance:
    """Provenance record; cuEST models carry their level of theory and GPU."""
    model = f"{c.model_spec['type']}:{_model_fingerprint(c.model_spec, variant)}"
    common: dict[str, Any] = {
        "geometry_hash": geometry_hash(coords, c.atomic_numbers, model),
        "atom_order": c.symbols,
        "atom_map": c.atom_map,
        "charge": c.hypothesis.charge,
        "multiplicity": c.hypothesis.multiplicity,
        "model": model,
        "git_commit": env["git_commit"],
        "saddle_version": env["saddle_version"],
    }
    d = dict(described or {})
    if d.get("type") == "cuest":
        return Provenance(
            **common,
            basis=str(d["basis"]),
            auxiliary_basis=str(d["auxiliary_basis"]),
            grid=str(d["grid"]),
            functional=str(d["functional"]),
            thresholds={
                "thresholdPQ": float(d["thresholdPQ"]),
                "orbital_gradient": float(d["orbital_gradient"]),
                "g1_rms": G.G1_RMS_GRADIENT,
                "g1_max": G.G1_MAX_GRADIENT,
                "reference_gtol_rms": REFERENCE_GTOL_RMS,
            },
            math_mode=str(d["math_mode"]),
            cuest_version=str(d["cuest_version"]),
            cuda_version=env.get("cuda_runtime", env["cuda_version"]),
            gpu_model=env["gpu_model"],
            compute_capability=env["compute_capability"],
            extra={"git_dirty": env["git_dirty"], "variant": variant, "solvent": str(d["solvent"])},
        )
    return Provenance(
        **common,
        basis=NOT_APPLICABLE,
        auxiliary_basis=NOT_APPLICABLE,
        grid=NOT_APPLICABLE,
        functional=NOT_APPLICABLE,
        thresholds={"g1_rms": G.G1_RMS_GRADIENT, "g1_max": G.G1_MAX_GRADIENT},
        math_mode=NOT_APPLICABLE,
        cuest_version=NOT_APPLICABLE,
        cuda_version=NOT_APPLICABLE,
        gpu_model=NOT_APPLICABLE,
        compute_capability=NOT_APPLICABLE,
        extra={"git_dirty": env["git_dirty"], "variant": variant},
    )


def _noise_floor_cm(epsilon: float, step: float, lightest_amu: float) -> float:
    """Smallest frequency the FD Hessian resolves, from noise and truncation error."""
    dk = epsilon / step + FOURTH_DERIVATIVE_ESTIMATE * step * step / 6.0
    return math.sqrt(dk / (lightest_amu * units.AMU_TO_ELECTRON_MASS)) * units.HARTREE_TO_WAVENUMBER


def _bond_change_direction(c: Campaign, coords: FloatArray, p: Pathway) -> FloatArray:
    """Cartesian direction that stretches breaking bonds and shortens forming bonds."""
    x = coords.reshape(-1, 3)
    index = {a: i for i, a in enumerate(c.atom_map)}
    d = np.zeros_like(x)
    for bonds, sign in ((p.breaking, 1.0), (p.forming, -1.0)):
        for a, b in bonds:
            i, j = index[a], index[b]
            u = (x[j] - x[i]) / np.linalg.norm(x[j] - x[i])
            d[j] += sign * u
            d[i] -= sign * u
    return d.ravel() / np.linalg.norm(d)


def _thermo(
    c: Campaign, atoms: Atoms, coords: FloatArray, freqs: FloatArray, symmetry: int, quasi: bool
) -> thermo.ThermoResult:
    return thermo.rrho(
        coords_bohr=coords,
        masses_amu=atoms.masses,
        frequencies_cm=freqs,
        temperature_k=c.hypothesis.temperature_k,
        symmetry_number=symmetry,
        multiplicity=atoms.multiplicity,
        quasi_harmonic=quasi,
        standard_state="1M",
    )


def _with_thermo(
    point: StationaryPoint, c: Campaign, atoms: Atoms, freqs: FloatArray, symmetry: int
) -> StationaryPoint:
    """Attach qRRHO and RRHO thermochemistry (both kept: their gap is sigma_thermo)."""
    point.thermo_qh = _thermo(c, atoms, point.coords, freqs, symmetry, True)
    point.thermo_rrho = _thermo(c, atoms, point.coords, freqs, symmetry, False)
    return point


class _Counter:
    """Counts model evaluations (reported as n_gradients)."""

    def __init__(self, inner: Model) -> None:
        self.inner = inner
        self.calls = 0

    def evaluate(self, coords: FloatArray) -> tuple[float, FloatArray]:
        self.calls += 1
        return self.inner.evaluate(coords)


def _hessian(
    model: Model, atoms: Atoms, coords: FloatArray, step: float, store: StencilStore, settings: str
) -> FloatArray:
    run = run_central_hessian(model.evaluate, coords, atoms.atomic_numbers, step, store, settings)
    return np.asarray(run.raw)


def _frequencies(raw_hessian: FloatArray, coords: FloatArray, masses: FloatArray) -> Any:
    """Vibrational analysis of the symmetrized FD Hessian."""
    return _core.analyze_vibrations(0.5 * (raw_hessian + raw_hessian.T), coords, masses)


def _random_rotation(seed: int) -> FloatArray:
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _rotate(coords: FloatArray, rotation: FloatArray) -> FloatArray:
    return (np.asarray(coords).reshape(-1, 3) @ rotation.T).ravel()


def _is_dft(spec: Mapping[str, Any]) -> bool:
    f = spec.get("functional")
    return (
        str(spec.get("type")) == "cuest" and f is not None and str(f).upper() not in ("HF", "RHF")
    )


def _refined(
    search: Any, model: Model, guess: FloatArray, numbers: list[int], **kw: Any
) -> dict[str, Any]:
    """Two-stage search: converge to G1, then restart with a fresh exact Hessian.

    BFGS/Bofill updates underestimate soft modes and converge only linearly,
    so tightening to the reference threshold (spec §6 "Residual force") is
    done by a Newton-RFO restart from the G1 geometry, which converges
    quadratically. The second stage starts with an exact FD Hessian.
    """
    first = search(model, guess, numbers, **kw)
    second = search(
        model,
        np.asarray(first["coords"]),
        numbers,
        gtol_rms=REFERENCE_GTOL_RMS,
        gtol_max=REFERENCE_GTOL_MAX,
        **{k: v for k, v in kw.items() if k not in ("gtol_rms", "gtol_max")},
    )

    # The refinement must never be worse than its starting point. A DFT grid
    # leaves a spurious torque (~1e-6 at (99,590)) that internal steps cannot
    # remove; when stage 2 cannot reach the reference threshold it may wander,
    # so the stage with the smaller gradient is kept and the shortfall noted.
    def rms(r: Mapping[str, Any]) -> float:
        g = np.asarray(r["gradient"])
        return float(np.sqrt(np.mean(g**2)))

    best = second
    if not second["converged"] and rms(first) <= rms(second):
        best = first
    best = dict(best)
    best["iterations"] = int(first["iterations"]) + int(second["iterations"])
    best["n_gradients"] = int(first["n_gradients"]) + int(second["n_gradients"])
    best["stage1_converged"] = bool(first["converged"])
    best["refined"] = bool(second["converged"])
    best["converged"] = bool(first["converged"] or second["converged"])
    return best


def _minimum(
    model: Model,
    c: Campaign,
    atoms: Atoms,
    guess: FloatArray,
    step: float,
    store: StencilStore,
    settings: str,
    symmetry: int,
) -> tuple[StationaryPoint, bool]:
    """Minimize with reference-level refinement, then a Hessian and thermochemistry.

    Returns the point and whether the minimization converged. A point with
    non-positive modes carries a note and no thermochemistry.
    """
    if len(atoms.symbols) == 1:
        energy, _ = model.evaluate(guess)
        atom = StationaryPoint(guess, float(energy), np.zeros((3, 3)), np.zeros(0), None, None)
        return _with_thermo(atom, c, atoms, np.zeros(0), symmetry), True
    r = _refined(_core.minimize, model, guess, atoms.atomic_numbers)
    coords = np.asarray(r["coords"])
    raw = _hessian(model, atoms, coords, step, store, settings)
    freqs = np.asarray(_frequencies(raw, coords, atoms.masses).frequencies_cm)
    point = StationaryPoint(coords, float(r["energy"]), raw, freqs, None, None)
    if np.any(freqs <= 0):
        point.note = f"minimum has non-positive modes {freqs[freqs <= 0].round(1).tolist()}"
        return point, bool(r["converged"])
    return _with_thermo(point, c, atoms, freqs, symmetry), bool(r["converged"])


# --------------------------------------------------------------------------- one pathway


def _competitors(c: Campaign, p: Pathway) -> tuple[str, ...]:
    return tuple(n for n in c.hypothesis.compare if n != p.name)


def _failed_search(c: Campaign, p: Pathway, exc: Exception, n_gradients: int) -> PathwayAnalysis:
    """The search budget was spent without a valid saddle (spec §5)."""
    evidence = G.Evidence(declared_competitors=_competitors(c, p))
    if isinstance(exc, PCMNotConverged):
        evidence = evidence.replace(
            scf_converged=False, orbitals_stable=False, uses_pcm=True, pcm_converged=False
        )
    return PathwayAnalysis(
        pathway=p,
        evidence=evidence,
        ts=None,
        barrier_kcal_mol=None,
        barrier_by_step={},
        numerical_sigma=None,
        thermo_sigma=None,
        noise_epsilon=float("nan"),
        noise_floor_cm=float("nan"),
        step=float("nan"),
        tunneling={},
        irc={},
        notes=[f"saddle search failed: {exc}"],
        n_gradients=n_gradients,
        saddle_found=False,
    )


def _search_notes(ts_run: Mapping[str, Any]) -> list[str]:
    notes = []
    if not ts_run.get("refined", True):
        notes.append(
            "TS refinement did not reach the reference threshold "
            f"(RMS {REFERENCE_GTOL_RMS:g}); kept the better stage"
        )
    if not ts_run["converged"]:
        notes.append(f"saddle search did not converge in {ts_run['iterations']} iterations")
    return notes


@dataclass(frozen=True)
class _FDStep:
    """Measured gradient noise and the finite-difference step chosen from it."""

    epsilon: float  # as probed (may be below resolution)
    step: float
    noise_floor_cm: float
    note: str | None


def _stage_fd_step(model: Model, x: FloatArray, mode: FloatArray, lightest_amu: float) -> _FDStep:
    """Probe gradient noise along the followed mode, then pick the optimal FD step."""
    eps, _ = _core.probe_noise(model, x, mode)
    eps_used = max(eps, NOISE_RESOLUTION_FLOOR)
    note = None
    if eps < NOISE_RESOLUTION_FLOOR:
        note = f"noise {eps:.1e} below probe resolution; using {NOISE_RESOLUTION_FLOOR:g}"
    h = float(np.clip(_core.optimal_step(eps_used, FOURTH_DERIVATIVE_ESTIMATE), MIN_STEP, MAX_STEP))
    return _FDStep(float(eps), h, _noise_floor_cm(eps_used, h, lightest_amu), note)


def _stage_evidence(
    c: Campaign,
    p: Pathway,
    x: FloatArray,
    gradient: FloatArray,
    g2: Mapping[str, Any],
    quality: QualityReport,
    vib: Any,
    noise_floor_cm: float,
    reactant_graph: Graph,
) -> G.Evidence:
    """Everything gates G1-G7 need about the TS, before the IRC is run."""
    return G.Evidence(
        gradient=np.asarray(_core.internal_gradient(x, c.atomic_numbers, np.asarray(gradient))),
        raw_gradient=np.asarray(gradient),
        scf_converged=g2["scf_converged"],
        orbitals_stable=g2["orbitals_stable"],
        uses_pcm=g2["uses_pcm"],
        pcm_converged=g2["pcm_converged"],
        hessian_quality=quality,
        frequencies_cm=np.asarray(vib.frequencies_cm),
        noise_floor_cm=noise_floor_cm,
        ts_mode_cartesian=np.asarray(vib.modes_cartesian)[:, 0],
        coords_bohr=x,
        atom_map=c.atom_map,
        breaking=p.breaking,
        forming=p.forming,
        reactant_graph=reactant_graph,
        expected_product_graph=expected_products(reactant_graph, p),
        alternative_product_graphs={
            q.name: expected_products(reactant_graph, q)
            for q in c.hypothesis.pathways
            if q.name != p.name
        },
        declared_competitors=_competitors(c, p),
    )


def _stage_irc(
    c: Campaign,
    model: Model,
    x: FloatArray,
    hessian: FloatArray,
    reactant_graph: Graph,
    product_graph: Graph,
) -> tuple[dict[str, Any], dict[str, FloatArray], dict[str, Graph]]:
    """IRC in both directions; returns the per-branch report, endpoints and their graphs."""
    path = _core.irc(
        model, x, c.atomic_numbers, hessian, c.masses, IRC_STEP, IRC_MAX_POINTS, IRC_SWITCH_GRADIENT
    )
    report: dict[str, Any] = {}
    endpoints: dict[str, FloatArray] = {}
    graphs: dict[str, Graph] = {}
    for branch in ("forward", "reverse"):
        end = path[branch]["endpoint"]
        endpoints[branch] = np.asarray(end["coords"])
        graphs[branch] = bond_graph(c.symbols, endpoints[branch], c.atom_map)
        report[branch] = {
            "stop_reason": path[branch]["stop_reason"],
            "points": len(path[branch]["path"]),
            "endpoint_converged": bool(end["converged"]),
            "endpoint_energy_eh": float(end["energy"]),
            "matches_reactant": compare_graphs(graphs[branch], reactant_graph).matches,
            "matches_product": compare_graphs(graphs[branch], product_graph).matches,
        }
    return report, endpoints, graphs


def _known(value: float | None, what: str) -> float:
    if value is None:
        raise ValueError(f"{what} needs thermochemistry, which is missing")
    return value


@dataclass(frozen=True)
class _Barrier:
    value: float | None = None
    by_step: dict[str, float] = field(default_factory=dict)
    numerical_sigma: float | None = None
    thermo_sigma: float | None = None


def _stage_barrier(
    c: Campaign,
    ts: StationaryPoint,
    raws: Mapping[float, FloatArray],
    h: float,
    real: FloatArray,
    reactant: ReactantSet,
) -> _Barrier:
    """Free-energy barrier at h, plus its h-vs-2h and qRRHO-vs-RRHO spreads.

    Attaches thermochemistry to ``ts``. Only called for a clean first-order
    saddle with complete reactant thermochemistry.
    """
    system = c.system
    _with_thermo(ts, c, system, real, c.symmetry_ts)
    quasi = c.quasi_harmonic
    g_r = _known(reactant.g_total(quasi), "reactant free energy")
    g_r_other = _known(reactant.g_total(not quasi), "reactant free energy")
    g_ts = _known(ts.g_total(quasi), "TS free energy")
    g_ts_other = _known(ts.g_total(not quasi), "TS free energy")
    barrier = units.hartree_to_kcal_mol(g_ts - g_r)
    by_step = {f"{h:.3g}": barrier}
    numerical_sigma = None
    f2 = np.asarray(_frequencies(raws[2 * h], ts.coords, c.masses).frequencies_cm)
    real2 = f2[f2 > 0]
    if real2.size == real.size:
        t2 = _thermo(c, system, ts.coords, real2, c.symmetry_ts, quasi)
        b2 = units.hartree_to_kcal_mol(ts.energy + t2.g_correction - g_r)
        by_step[f"{2 * h:.3g}"] = b2
        numerical_sigma = abs(barrier - b2)
    thermo_sigma = abs(barrier - units.hartree_to_kcal_mol(g_ts_other - g_r_other))
    return _Barrier(barrier, by_step, numerical_sigma, thermo_sigma)


def _stage_tunneling(
    c: Campaign,
    model: Model,
    imaginary_cm: float,
    ts: StationaryPoint,
    irc: Mapping[str, Any],
    endpoints: Mapping[str, FloatArray],
    h: float,
    store: StencilStore,
    settings: str,
    reactant: ReactantSet,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Tunneling factors (reported separately) and the IRC endpoint minima.

    Eckart barriers are ZPE-corrected and measured from the IRC endpoint
    minima on each side (for bimolecular reactions these are the pre- and
    post-reaction complexes).
    """
    tun: dict[str, Any] = {"wigner": tunneling.wigner(imaginary_cm, c.hypothesis.temperature_k)}
    endpoint_minima: dict[str, float] = {}
    side_branch = {
        "reactant": next((b for b, i in irc.items() if i["matches_reactant"]), None),
        "product": next((b for b, i in irc.items() if i["matches_product"]), None),
    }
    if None not in side_branch.values() and ts.thermo_qh is not None:
        minima = {
            side: _minimum(model, c, c.system, endpoints[branch], h, store, settings, 1)[0]
            for side, branch in side_branch.items()
            if branch is not None
        }
        for side, pt in minima.items():
            endpoint_minima[side] = units.hartree_to_kcal_mol(pt.energy - reactant.energy)
        zpe_r, zpe_p = (minima[s].thermo_qh for s in ("reactant", "product"))
        if zpe_r is not None and zpe_p is not None:
            e_ts = ts.energy + ts.thermo_qh.zpe
            v_f = e_ts - (minima["reactant"].energy + zpe_r.zpe)
            v_r = e_ts - (minima["product"].energy + zpe_p.zpe)
            if v_f > 0 and v_r > 0:
                tun["eckart"] = tunneling.eckart(imaginary_cm, v_f, v_r, c.hypothesis.temperature_k)
                tun["forward_barrier_kcal_mol"] = units.hartree_to_kcal_mol(v_f)
                tun["reverse_barrier_kcal_mol"] = units.hartree_to_kcal_mol(v_r)
    if "eckart" not in tun:
        tun["eckart_note"] = "Eckart not computed: IRC endpoints missing or not ZPE-corrected"
    return tun, endpoint_minima


def analyze_pathway(
    c: Campaign,
    p: Pathway,
    provider: ModelProvider,
    reactant: ReactantSet,
    store: StencilStore,
    settings: str,
    start: FloatArray | None = None,
) -> PathwayAnalysis:
    """Find, characterize and validate one pathway's TS against separated reactants."""
    model = provider.system()
    counter = _Counter(model)
    guess = c.ts_guesses[p.name] if start is None else start
    try:
        ts_run = _refined(
            _core.saddle_search,
            counter,
            guess,
            c.atomic_numbers,
            follow_direction=_bond_change_direction(c, guess, p),
        )
    except (SCFError, ModelError, RuntimeError, ValueError) as exc:
        return _failed_search(c, p, exc, counter.calls)
    x = np.asarray(ts_run["coords"])
    notes = _search_notes(ts_run)

    fd = _stage_fd_step(counter, x, np.asarray(ts_run["followed_mode"]), float(c.masses.min()))
    if fd.note:
        notes.append(fd.note)
    h = fd.step
    raws = {
        h: _hessian(counter, c.system, x, h, store, settings),
        2 * h: _hessian(counter, c.system, x, 2 * h, store, settings),
    }
    quality = assess_hessians(raws, x, c.masses, noise_floor_cm=fd.noise_floor_cm)
    hess = 0.5 * (raws[h] + raws[h].T)
    vib = _core.analyze_vibrations(hess, x, c.masses)
    freqs = np.asarray(vib.frequencies_cm)
    energy, gradient = counter.evaluate(x)
    g2 = provider.g2_evidence(model, x)
    if g2.get("note"):
        notes.append(f"G2: {g2['note']}")
    reactant_graph = reactant.graph(c)
    evidence = _stage_evidence(
        c, p, x, gradient, g2, quality, vib, fd.noise_floor_cm, reactant_graph
    )

    irc: dict[str, Any] = {}
    endpoints: dict[str, FloatArray] = {}
    if G.g4_saddle_order(evidence).passed:
        product_graph = expected_products(reactant_graph, p)
        irc, endpoints, graphs = _stage_irc(c, counter, x, hess, reactant_graph, product_graph)
        evidence = evidence.replace(
            irc_forward_graph=graphs["forward"], irc_reverse_graph=graphs["reverse"]
        )
    else:
        notes.append("IRC skipped: G4 failed")

    imag = freqs[freqs < -fd.noise_floor_cm]
    real = freqs[freqs > 0]
    ts = StationaryPoint(x, float(energy), raws[h], freqs, None, None)
    clean_saddle = imag.size == 1 and real.size == freqs.size - 1
    reactant_thermo = reactant.g_total(True) is not None and reactant.g_total(False) is not None
    if clean_saddle and reactant_thermo:
        barrier = _stage_barrier(c, ts, raws, h, real, reactant)
    else:
        barrier = _Barrier()
        notes.append("barrier not computed: TS is not a clean first-order saddle")

    tun: dict[str, Any] = {}
    endpoint_minima: dict[str, float] = {}
    if imag.size == 1:
        tun, endpoint_minima = _stage_tunneling(
            c, counter, float(imag[0]), ts, irc, endpoints, h, store, settings, reactant
        )

    return PathwayAnalysis(
        pathway=p,
        evidence=evidence,
        ts=ts,
        barrier_kcal_mol=barrier.value,
        barrier_by_step=barrier.by_step,
        numerical_sigma=barrier.numerical_sigma,
        thermo_sigma=barrier.thermo_sigma,
        noise_epsilon=fd.epsilon,
        noise_floor_cm=fd.noise_floor_cm,
        step=h,
        tunneling=tun,
        irc=irc,
        notes=notes,
        n_gradients=counter.calls,
        electronic_barrier_kcal_mol=units.hartree_to_kcal_mol(float(energy) - reactant.energy),
        endpoint_minima_kcal_mol=endpoint_minima,
    )


# --------------------------------------------------------------------------- one model


def _run_variant(
    c: Campaign,
    name: str,
    variant: Mapping[str, Any] | None,
    starts: Mapping[str, FloatArray] | None,
    store: StencilStore,
    env: Mapping[str, str],
) -> tuple[ReactantSet, dict[str, PathwayAnalysis], dict[str, Any]]:
    """Minimize reactants and analyze every pathway with one (variant) model.

    ``starts`` replaces the campaign TS guesses (variants restart from the
    baseline TSs). A ``rotation_seed`` in the variant rigidly rotates every
    input geometry (spec §6 rotation test). Returns the reactants, the
    analyses and the model's self-description.
    """
    settings_overrides = dict(variant or {})
    seed = settings_overrides.pop("rotation_seed", None)
    reactant_coords = c.reactant_coords
    if seed is not None:
        rotation = _random_rotation(int(seed))
        reactant_coords = _rotate(reactant_coords, rotation)
        source = starts if starts is not None else c.ts_guesses
        starts = {k: _rotate(v, rotation) for k, v in source.items()}
    provider = build_provider(c.model_spec, settings_overrides or None, c.system_info())
    try:
        described = dict(provider.describe())
        settings = _provenance(c, c.reactant_coords, name, env, described).settings_fingerprint()
        species = []
        for i in range(len(c.hypothesis.reactants)):
            point, converged = _minimum(
                provider.fragment(i),
                c,
                c.species(i),
                c.species_coords(reactant_coords, i),
                SPECIES_STEP,
                store,
                f"{settings}/species{i}",
                c.symmetry_species[i],
            )
            if not converged:
                raise CampaignError(f"[{name}] minimization of species {i} did not converge")
            species.append(point)
        reactant = ReactantSet(species)
        analyses = {
            p.name: analyze_pathway(
                c,
                p,
                provider,
                reactant,
                store,
                settings,
                None if starts is None else starts.get(p.name),
            )
            for p in c.hypothesis.pathways
        }
        return reactant, analyses, described
    finally:
        provider.close()


def _outcome(
    a: PathwayAnalysis, gates: Sequence[G.GateResult], budget: UncertaintyBudget
) -> PathwayOutcome:
    return PathwayOutcome(
        name=a.pathway.name,
        gates=list(gates),
        barrier_kcal_mol=a.barrier_kcal_mol,
        uncertainty=budget,
        saddle_found=a.saddle_found,
        search_exhausted=not a.saddle_found,
    )


def _preliminary_verdicts(
    analyses: Mapping[str, PathwayAnalysis],
    terms: Mapping[str, Term],
    evaluated: Mapping[str, bool],
    limit: float,
) -> dict[str, VerdictResult]:
    """Verdicts from gates G1-G7 only; G8 compares these across variants."""
    for a in analyses.values():
        a.evidence = a.evidence.replace(competitors_evaluated=evaluated)
    return decide(
        [
            _outcome(
                a, [gate(a.evidence) for gate in G.GATES_BEFORE_ROBUSTNESS], a.uncertainty(terms)
            )
            for a in analyses.values()
        ],
        limit,
    )


def _with_measured_grid_term(
    c: Campaign, a: PathwayAnalysis, variants: Mapping[str, VariantOutcome]
) -> UncertaintyBudget:
    """Budget with sigma_grid taken from the rotation and grid variants when run."""
    shifts = [
        abs(v.barrier_kcal_mol - a.barrier_kcal_mol)
        for k in ("rotation", "grid")
        if (v := variants.get(k)) is not None
        and v.barrier_kcal_mol is not None
        and a.barrier_kcal_mol is not None
    ]
    if not shifts:
        return a.uncertainty(c.uncertainty)
    measured = max(shifts)
    stated = c.uncertainty.get("grid")
    stated_value = stated.value_kcal_mol if stated is not None else None
    terms = dict(c.uncertainty)
    terms["grid"] = Term(
        "grid",
        max(measured, stated_value or 0.0),
        "measured",
        f"max barrier shift under rotation/grid variants {measured:.3f}",
    )
    return a.uncertainty(terms)


def _rankings(c: Campaign, outcomes: Mapping[str, PathwayOutcome]) -> list[dict[str, Any]]:
    """Every declared pair of competing pathways, ranked at 2 sigma."""
    rankings = []
    compare = list(c.hypothesis.compare)
    for i, a_name in enumerate(compare):
        for b_name in compare[i + 1 :]:
            r = rank_pair(outcomes[a_name], outcomes[b_name])
            rankings.append(
                {
                    "pair": [a_name, b_name],
                    "resolved": r.resolved,
                    "lower": r.lower,
                    "reason": r.reason,
                }
            )
    return rankings


def _none_if_nan(x: float) -> float | None:
    return None if math.isnan(x) else x


def _pathway_summary(
    c: Campaign,
    a: PathwayAnalysis,
    verdict: VerdictResult,
    budget: UncertaintyBudget,
    gates: Sequence[G.GateResult],
    variants: Mapping[str, VariantOutcome],
    limit: float,
) -> dict[str, Any]:
    """One pathway's entry in campaign.json (also stored with its TS result)."""
    ts = a.ts
    return {
        "pathway": a.pathway.name,
        "campaign": str(c.path.resolve()),
        "hypothesis": c.hypothesis.id,
        "verdict": verdict.verdict.value,
        "reason": verdict.reason,
        "barrier_kcal_mol": a.barrier_kcal_mol,
        "electronic_barrier_kcal_mol": a.electronic_barrier_kcal_mol,
        "endpoint_minima_kcal_mol": a.endpoint_minima_kcal_mol,
        "sigma_kcal_mol": verdict.sigma_kcal_mol,
        "limit_kcal_mol": limit,
        "uncertainty": [
            {"component": comp, "value": val, "source": src, "note": note}
            for comp, val, src, note in budget.breakdown()
        ],
        "gates": [dataclasses.asdict(g) for g in gates],
        "imaginary_frequency_cm": float(ts.frequencies_cm[0]) if ts is not None else None,
        "frequencies_cm": [float(f) for f in ts.frequencies_cm] if ts is not None else [],
        "noise_epsilon_eh_bohr": _none_if_nan(a.noise_epsilon),
        "noise_floor_cm": _none_if_nan(a.noise_floor_cm),
        "fd_step_bohr": _none_if_nan(a.step),
        "barrier_by_step": a.barrier_by_step,
        "robustness": {
            k: {"verdict": v.verdict, "barrier_kcal_mol": v.barrier_kcal_mol, "reason": v.reason}
            for k, v in variants.items()
        },
        "tunneling": a.tunneling,
        "irc": a.irc,
        "notes": a.notes,
        "n_gradients": a.n_gradients,
    }


# --------------------------------------------------------------------------- campaign


def run_campaign(
    path: str | Path, out_dir: str | Path, *, overwrite: bool = False
) -> CampaignResult:
    """Run every pathway, robustness variant and verdict; write results under out_dir.

    Existing results are kept unless overwrite=True. Cached stencil points in
    out_dir/stencils.sqlite are reused, so an interrupted run resumes.
    """
    c = load_campaign(path)
    out = Path(out_dir)
    summary_path = out / "campaign.json"
    if summary_path.exists() and not overwrite:
        raise CampaignError(f"{summary_path} exists; pass overwrite=True to replace it")
    out.mkdir(parents=True, exist_ok=True)
    store = StencilStore(out / "stencils.sqlite")
    env = collect_environment()
    limit = barrier_limit_kcal_mol(c.hypothesis.temperature_k, c.hypothesis.target_half_life_s)

    # 1. Baseline model: every pathway, then verdicts from G1-G7.
    reactant, analyses, described = _run_variant(c, "baseline", None, None, store, env)
    evaluated = {n: a.completed for n, a in analyses.items()}
    baseline = _preliminary_verdicts(analyses, c.uncertainty, evaluated, limit)

    # 2. Robustness: rerun from the baseline TS geometries with each variant model.
    starts = {n: a.ts.coords for n, a in analyses.items() if a.ts is not None}
    robustness = dict(c.robustness)
    if _is_dft(c.model_spec):
        # DFT grids are not rotationally invariant; G8 includes a rotation test.
        robustness.setdefault("rotation", {"rotation_seed": ROTATION_SEED})
    variants: dict[str, dict[str, VariantOutcome]] = {n: {} for n in analyses}
    for vname, vspec in robustness.items():
        _, va, _ = _run_variant(c, vname, vspec, starts, store, env)
        vd = _preliminary_verdicts(va, c.uncertainty, evaluated, limit)
        for n in analyses:
            variants[n][vname] = VariantOutcome(
                vd[n].verdict.value, va[n].barrier_kcal_mol, vd[n].reason
            )

    # 3. Final gates (now including G8), measured sigma_grid, verdicts, rankings.
    final_gates: dict[str, list[G.GateResult]] = {}
    for n, a in analyses.items():
        a.evidence = a.evidence.replace(
            baseline_verdict=baseline[n].verdict.value,
            robustness_verdicts={k: v.verdict for k, v in variants[n].items()},
        )
        final_gates[n] = G.evaluate_all(a.evidence)
    budgets = {n: _with_measured_grid_term(c, a, variants[n]) for n, a in analyses.items()}
    outcomes = {n: _outcome(a, final_gates[n], budgets[n]) for n, a in analyses.items()}
    verdicts = decide(list(outcomes.values()), limit)

    # 4. Write results: one HDF5 per TS and the campaign summary.
    pathway_summaries: dict[str, Any] = {}
    for n, a in analyses.items():
        summary = _pathway_summary(
            c, a, verdicts[n], budgets[n], final_gates[n], variants[n], limit
        )
        pathway_summaries[n] = summary
        if a.ts is not None:
            (out / n).mkdir(parents=True, exist_ok=True)
            write_result(
                out / n / "ts.h5",
                _provenance(c, a.ts.coords, "baseline", env, described),
                {
                    "coords": a.ts.coords,
                    "hessian_raw": a.ts.hessian,
                    "frequencies": a.ts.frequencies_cm,
                },
                {"coords": "bohr", "hessian_raw": "Eh/bohr^2", "frequencies": "cm^-1"},
                summary,
                overwrite=overwrite,
            )
    campaign_summary = {
        "campaign": str(c.path),
        "hypothesis": c.hypothesis.id,
        "temperature_K": c.hypothesis.temperature_k,
        "target_half_life_s": c.hypothesis.target_half_life_s,
        "limit_kcal_mol": limit,
        "reactant_energy_eh": reactant.energy,
        "model": described,
        "pathways": pathway_summaries,
        "rankings": _rankings(c, outcomes),
        "environment": env,
    }
    summary_path.write_text(
        json.dumps(campaign_summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    store.close()
    return CampaignResult(
        c, limit, analyses, {n: v.verdict for n, v in verdicts.items()}, campaign_summary, out
    )


__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "Campaign",
    "CampaignError",
    "CampaignResult",
    "Verdict",
    "analyze_pathway",
    "load_campaign",
    "run_campaign",
]
