"""End-to-end feasibility pipeline (spec §4 "Pipeline").

hypothesis -> reactant minimum -> RS-P-RFO saddle -> noise probe -> step
-> restartable Hessians at h and 2h -> quality -> vibrations -> IRC ->
graph match -> thermochemistry -> gates -> robustness -> verdict -> results

Separated reactants are minimized and treated species by species, so a
bimolecular barrier is measured from separated reactants with the per-species
standard-state term (spec §4). Fake surfaces describe only the whole system,
so they support single-species reactants; cuEST models evaluate fragments.
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
import yaml

from saddle import _core, elements, units
from saddle.analysis import thermo, tunneling
from saddle.analysis.hessian_quality import assess_hessians
from saddle.analysis.kinetics import barrier_limit_kcal_mol
from saddle.analysis.uncertainty import COMPONENTS, Term, UncertaintyBudget
from saddle.errors import PCMNotConverged, SCFError
from saddle.gates import gates as G
from saddle.gates.verdict import PathwayOutcome, Verdict, decide, rank_pair
from saddle.graphs import Graph, bond_graph, compare_graphs, expected_products
from saddle.hypothesis import Hypothesis, Pathway, load
from saddle.store.geometry import geometry_hash
from saddle.store.provenance import NOT_APPLICABLE, Provenance, collect_environment
from saddle.store.results import write_result
from saddle.store.stencil import StencilStore, run_central_hessian
from saddle.workflows.models import Model, ModelError, ModelProvider, SystemInfo, build_provider

FloatArray = npt.NDArray[np.float64]

CAMPAIGN_SCHEMA_VERSION = 1
# Spec §6 design estimate of |E''''| used to choose the FD step.
FOURTH_DERIVATIVE_ESTIMATE = 2.0
# Below this the probe cannot resolve noise (clean analytic surfaces).
NOISE_RESOLUTION_FLOOR = 1e-10
MIN_STEP, MAX_STEP = 1e-4, 2e-2
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
IRC_SWITCH_GRADIENT = 1e-3


class CampaignError(ValueError):
    pass


@dataclass(frozen=True)
class Campaign:
    path: Path
    hypothesis: Hypothesis
    model_spec: Mapping[str, Any]
    ts_guesses: Mapping[str, FloatArray]
    symmetry_reactant: int
    symmetry_ts: int
    symmetry_by_name: Mapping[str, int]
    quasi_harmonic: bool
    uncertainty: Mapping[str, Term]
    robustness: Mapping[str, Mapping[str, Any]]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(s for sp in self.hypothesis.reactants for s in sp.symbols)

    @property
    def atom_map(self) -> tuple[int, ...]:
        return tuple(i for sp in self.hypothesis.reactants for i in sp.atom_map)

    @property
    def masses(self) -> FloatArray:
        return np.array([elements.isotope_mass(s) for s in self.symbols])

    @property
    def atomic_numbers(self) -> list[int]:
        return [elements.atomic_number(s) for s in self.symbols]

    @property
    def reactant_coords(self) -> FloatArray:
        return np.concatenate([sp.coords_bohr.ravel() for sp in self.hypothesis.reactants])

    @property
    def symmetry_species(self) -> tuple[int, ...]:
        return tuple(
            self.symmetry_by_name.get(sp.name, self.symmetry_reactant)
            for sp in self.hypothesis.reactants
        )

    def species_slice(self, index: int) -> slice:
        start = sum(len(sp.symbols) for sp in self.hypothesis.reactants[:index])
        return slice(start, start + len(self.hypothesis.reactants[index].symbols))

    def system_info(self) -> SystemInfo:
        return SystemInfo(
            symbols=self.symbols,
            species_symbols=tuple(tuple(sp.symbols) for sp in self.hypothesis.reactants),
            species_charges=tuple(sp.charge for sp in self.hypothesis.reactants),
            charge=self.hypothesis.charge,
            solvent=self.hypothesis.solvent,
        )


@dataclass(frozen=True)
class Atoms:
    """A subset of the system (the whole system or one reactant species)."""

    symbols: tuple[str, ...]
    atom_map: tuple[int, ...]
    multiplicity: int

    @property
    def masses(self) -> FloatArray:
        return np.array([elements.isotope_mass(s) for s in self.symbols])

    @property
    def atomic_numbers(self) -> list[int]:
        return [elements.atomic_number(s) for s in self.symbols]


def _species_atoms(c: "Campaign", index: int) -> Atoms:
    sp = c.hypothesis.reactants[index]
    electrons = sum(elements.atomic_number(x) for x in sp.symbols) - sp.charge
    return Atoms(tuple(sp.symbols), tuple(sp.atom_map), 1 if electrons % 2 == 0 else 2)


def _system_atoms(c: "Campaign") -> Atoms:
    return Atoms(c.symbols, c.atom_map, c.hypothesis.multiplicity)


def _read_guess(path: Path, n_atoms: int) -> FloatArray:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    if n != n_atoms:
        raise CampaignError(f"{path}: TS guess has {n} atoms, hypothesis has {n_atoms}")
    xyz = [[float(v) for v in ln.split()[1:4]] for ln in lines[2 : 2 + n]]
    return np.asarray(xyz).ravel() / units.BOHR_TO_ANGSTROM


def load_campaign(path: str | Path) -> Campaign:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if raw.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise CampaignError(f"campaign schema_version must be {CAMPAIGN_SCHEMA_VERSION}")
    base = path.parent
    hyp = load(base / raw["hypothesis"])
    if len(hyp.reactants) != 1 and str(raw["model"].get("type", "")).startswith("fake:"):
        raise CampaignError(
            "fake-surface campaigns support one reactant species; bimolecular reactants need "
            "a model that evaluates separated fragments (type: cuest)"
        )
    names = {p.name for p in hyp.pathways}
    guesses_raw = raw.get("pathways", {})
    if set(guesses_raw) != names:
        raise CampaignError(
            f"campaign pathways {sorted(guesses_raw)} != hypothesis {sorted(names)}"
        )
    guesses = {n: _read_guess(base / g["ts_guess"], hyp.n_atoms) for n, g in guesses_raw.items()}
    terms: dict[str, Term] = {}
    for name, t in (raw.get("uncertainty") or {}).items():
        if name in ("numerical", "thermo"):
            raise CampaignError(f"uncertainty '{name}' is measured by the pipeline; do not set it")
        value = t.get("value")
        terms[name] = Term(
            name, None if value is None else float(value), t["source"], t.get("note", "")
        )
    sym = raw.get("symmetry_numbers", {})
    return Campaign(
        path=path,
        hypothesis=hyp,
        model_spec=raw["model"],
        ts_guesses=guesses,
        symmetry_reactant=int(sym.get("reactant", 1)),
        symmetry_ts=int(sym.get("ts", 1)),
        symmetry_by_name={str(k): int(v) for k, v in sym.items() if k not in ("reactant", "ts")},
        quasi_harmonic=bool(raw.get("quasi_harmonic", True)),
        uncertainty=terms,
        robustness=raw.get("robustness") or {},
    )


@dataclass
class StationaryPoint:
    coords: FloatArray
    energy: float
    hessian: FloatArray
    frequencies_cm: FloatArray
    thermo_qh: thermo.ThermoResult | None
    thermo_rrho: thermo.ThermoResult | None
    note: str = ""


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
    model = f"{c.model_spec['type']}:{_model_fingerprint(c.model_spec, variant)}"
    d = dict(described or {})
    if d.get("type") == "cuest":
        return Provenance(
            geometry_hash=geometry_hash(coords, c.atomic_numbers, model),
            atom_order=c.symbols,
            atom_map=c.atom_map,
            charge=c.hypothesis.charge,
            multiplicity=c.hypothesis.multiplicity,
            model=model,
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
            git_commit=env["git_commit"],
            saddle_version=env["saddle_version"],
            extra={"git_dirty": env["git_dirty"], "variant": variant, "solvent": str(d["solvent"])},
        )
    return Provenance(
        geometry_hash=geometry_hash(coords, c.atomic_numbers, model),
        atom_order=c.symbols,
        atom_map=c.atom_map,
        charge=c.hypothesis.charge,
        multiplicity=c.hypothesis.multiplicity,
        model=model,
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
        git_commit=env["git_commit"],
        saddle_version=env["saddle_version"],
        extra={"git_dirty": env["git_dirty"], "variant": variant},
    )


def _noise_floor_cm(epsilon: float, step: float, lightest_amu: float) -> float:
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


class _Counter:
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


ROTATION_SEED = 20261105


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
    """Minimize with reference-level refinement, then a Hessian and thermochemistry."""
    if len(atoms.symbols) == 1:
        energy, _ = model.evaluate(guess)
        point = StationaryPoint(guess, float(energy), np.zeros((3, 3)), np.zeros(0), None, None)
        point.thermo_qh = _thermo(c, atoms, guess, np.zeros(0), symmetry, True)
        point.thermo_rrho = _thermo(c, atoms, guess, np.zeros(0), symmetry, False)
        return point, True
    r = _refined(_core.minimize, model, guess, atoms.atomic_numbers)
    coords = np.asarray(r["coords"])
    raw = _hessian(model, atoms, coords, step, store, settings)
    vib = _core.analyze_vibrations(0.5 * (raw + raw.T), coords, atoms.masses)
    freqs = np.asarray(vib.frequencies_cm)
    point = StationaryPoint(coords, float(r["energy"]), raw, freqs, None, None)
    if np.any(freqs <= 0):
        point.note = f"minimum has non-positive modes {freqs[freqs <= 0].round(1).tolist()}"
        return point, bool(r["converged"])
    point.thermo_qh = _thermo(c, atoms, coords, freqs, symmetry, True)
    point.thermo_rrho = _thermo(c, atoms, coords, freqs, symmetry, False)
    return point, bool(r["converged"])


@dataclass
class ReactantSet:
    """Separated reactant species; free energies and ZPEs add up."""

    species: list[StationaryPoint]

    @property
    def energy(self) -> float:
        return float(sum(p.energy for p in self.species))

    @property
    def coords(self) -> FloatArray:
        """Concatenated per-species coordinates; species live in separate frames."""
        return np.concatenate([p.coords.ravel() for p in self.species])

    def graph(self, c: "Campaign") -> Graph:
        """Union of each species' own bond graph (frames are independent)."""
        bonds: Graph = set()
        for i, point in enumerate(self.species):
            atoms = _species_atoms(c, i)
            bonds |= bond_graph(atoms.symbols, point.coords, atoms.atom_map)
        return bonds

    def g_total(self, quasi: bool) -> float | None:
        parts = [_g_total(p, quasi) for p in self.species]
        return (
            None if any(v is None for v in parts) else float(sum(v for v in parts if v is not None))
        )

    @property
    def zpe(self) -> float | None:
        if any(p.thermo_qh is None for p in self.species):
            return None
        return float(sum(p.thermo_qh.zpe for p in self.species if p.thermo_qh is not None))


def _g_total(point: StationaryPoint, quasi: bool) -> float | None:
    t = point.thermo_qh if quasi else point.thermo_rrho
    return None if t is None else point.energy + t.g_correction


def analyze_pathway(
    c: Campaign,
    p: Pathway,
    provider: ModelProvider,
    reactant: ReactantSet,
    store: StencilStore,
    settings: str,
    start: FloatArray | None = None,
) -> PathwayAnalysis:
    system = _system_atoms(c)
    model = provider.system()
    counter = _Counter(model)
    guess = c.ts_guesses[p.name] if start is None else start
    follow = _bond_change_direction(c, guess, p)
    notes: list[str] = []
    try:
        ts_run = _refined(
            _core.saddle_search, counter, guess, c.atomic_numbers, follow_direction=follow
        )
    except (SCFError, ModelError, RuntimeError, ValueError) as exc:
        # The search budget was spent without a valid saddle (spec §5).
        notes.append(f"saddle search failed: {exc}")
        failed = G.Evidence(
            declared_competitors=tuple(n for n in c.hypothesis.compare if n != p.name)
        )
        if isinstance(exc, PCMNotConverged):
            failed = failed.replace(
                scf_converged=False, orbitals_stable=False, uses_pcm=True, pcm_converged=False
            )
        return PathwayAnalysis(
            pathway=p,
            evidence=failed,
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
            notes=notes,
            n_gradients=counter.calls,
            saddle_found=False,
        )
    x = np.asarray(ts_run["coords"])
    if not ts_run.get("refined", True):
        notes.append(
            "TS refinement did not reach the reference threshold "
            f"(RMS {REFERENCE_GTOL_RMS:g}); kept the better stage"
        )
    if not ts_run["converged"]:
        notes.append(f"saddle search did not converge in {ts_run['iterations']} iterations")

    # Noise probe along the followed mode, then the step.
    eps, _ = _core.probe_noise(counter, x, np.asarray(ts_run["followed_mode"]))
    eps_used = max(eps, NOISE_RESOLUTION_FLOOR)
    if eps < NOISE_RESOLUTION_FLOOR:
        notes.append(f"noise {eps:.1e} below probe resolution; using {NOISE_RESOLUTION_FLOOR:g}")
    h = float(np.clip(_core.optimal_step(eps_used, FOURTH_DERIVATIVE_ESTIMATE), MIN_STEP, MAX_STEP))
    floor_cm = _noise_floor_cm(eps_used, h, float(c.masses.min()))

    raws = {
        h: _hessian(counter, system, x, h, store, settings),
        2 * h: _hessian(counter, system, x, 2 * h, store, settings),
    }
    quality = assess_hessians(raws, x, c.masses, noise_floor_cm=floor_cm)
    hess = 0.5 * (raws[h] + raws[h].T)
    vib = _core.analyze_vibrations(hess, x, c.masses)
    freqs = np.asarray(vib.frequencies_cm)
    energy, gradient = counter.evaluate(x)
    g2 = provider.g2_evidence(model, x)
    if g2.get("note"):
        notes.append(f"G2: {g2['note']}")

    reactant_graph = reactant.graph(c)
    product_graph = expected_products(reactant_graph, p)
    alternatives = {
        q.name: expected_products(reactant_graph, q)
        for q in c.hypothesis.pathways
        if q.name != p.name
    }

    evidence = G.Evidence(
        gradient=np.asarray(_core.internal_gradient(x, c.atomic_numbers, np.asarray(gradient))),
        raw_gradient=np.asarray(gradient),
        scf_converged=g2["scf_converged"],
        orbitals_stable=g2["orbitals_stable"],
        uses_pcm=g2["uses_pcm"],
        pcm_converged=g2["pcm_converged"],
        hessian_quality=quality,
        frequencies_cm=freqs,
        noise_floor_cm=floor_cm,
        ts_mode_cartesian=np.asarray(vib.modes_cartesian)[:, 0],
        coords_bohr=x,
        atom_map=c.atom_map,
        breaking=p.breaking,
        forming=p.forming,
        reactant_graph=reactant_graph,
        expected_product_graph=product_graph,
        alternative_product_graphs=alternatives,
        declared_competitors=tuple(n for n in c.hypothesis.compare if n != p.name),
    )

    # IRC and endpoints.
    irc_info: dict[str, Any] = {}
    ends: dict[str, Graph] = {}
    end_points: dict[str, FloatArray] = {}
    if G.g4_saddle_order(evidence).passed:
        path = _core.irc(
            counter, x, c.atomic_numbers, hess, c.masses, IRC_STEP, 300, IRC_SWITCH_GRADIENT
        )
        for branch in ("forward", "reverse"):
            end = path[branch]["endpoint"]
            coords_end = np.asarray(end["coords"])
            ends[branch] = bond_graph(c.symbols, coords_end, c.atom_map)
            end_points[branch] = coords_end
            irc_info[branch] = {
                "stop_reason": path[branch]["stop_reason"],
                "points": len(path[branch]["path"]),
                "endpoint_converged": bool(end["converged"]),
                "endpoint_energy_eh": float(end["energy"]),
                "matches_reactant": compare_graphs(ends[branch], reactant_graph).matches,
                "matches_product": compare_graphs(ends[branch], product_graph).matches,
            }
        evidence = evidence.replace(
            irc_forward_graph=ends["forward"], irc_reverse_graph=ends["reverse"]
        )
    else:
        notes.append("IRC skipped: G4 failed")

    # Thermochemistry at both steps; barrier only for a proper first-order saddle.
    imag = freqs[freqs < -floor_cm]
    real = freqs[freqs > 0]
    barrier = None
    by_step: dict[str, float] = {}
    numerical_sigma = thermo_sigma = None
    ts_point = StationaryPoint(x, float(energy), raws[h], freqs, None, None)
    g_r_qh = reactant.g_total(True)
    g_r_rrho = reactant.g_total(False)
    if (
        imag.size == 1
        and real.size == freqs.size - 1
        and g_r_qh is not None
        and g_r_rrho is not None
    ):
        ts_point.thermo_qh = _thermo(c, system, x, real, c.symmetry_ts, True)
        ts_point.thermo_rrho = _thermo(c, system, x, real, c.symmetry_ts, False)
        quasi = c.quasi_harmonic
        g_r = g_r_qh if quasi else g_r_rrho
        g_ts = _g_total(ts_point, quasi)
        assert g_ts is not None
        barrier = units.hartree_to_kcal_mol(g_ts - g_r)
        by_step[f"{h:.3g}"] = barrier
        raw2 = raws[2 * h]
        f2 = np.asarray(_core.analyze_vibrations(0.5 * (raw2 + raw2.T), x, c.masses).frequencies_cm)
        real2 = f2[f2 > 0]
        if real2.size == real.size:
            t2 = _thermo(c, system, x, real2, c.symmetry_ts, quasi)
            b2 = units.hartree_to_kcal_mol(energy + t2.g_correction - g_r)
            by_step[f"{2 * h:.3g}"] = b2
            numerical_sigma = abs(barrier - b2)
        g_ts_other = _g_total(ts_point, not quasi)
        g_r_other = g_r_rrho if quasi else g_r_qh
        assert g_ts_other is not None
        thermo_sigma = abs(barrier - units.hartree_to_kcal_mol(g_ts_other - g_r_other))
    else:
        notes.append("barrier not computed: TS is not a clean first-order saddle")

    # Tunneling, reported separately. Eckart barriers are ZPE-corrected and
    # measured from the IRC endpoint minima on each side (for bimolecular
    # reactions these are the pre- and post-reaction complexes).
    tun: dict[str, Any] = {}
    endpoint_minima: dict[str, float] = {}
    if imag.size == 1:
        tun["wigner"] = tunneling.wigner(float(imag[0]), c.hypothesis.temperature_k)
        reactant_branch = next((b for b, i in irc_info.items() if i["matches_reactant"]), None)
        product_branch = next((b for b, i in irc_info.items() if i["matches_product"]), None)
        if reactant_branch and product_branch and ts_point.thermo_qh is not None:
            ends_min = {}
            for label, branch in (("reactant", reactant_branch), ("product", product_branch)):
                pt, _ = _minimum(counter, c, system, end_points[branch], h, store, settings, 1)
                ends_min[label] = pt
            for label, pt in ends_min.items():
                endpoint_minima[label] = units.hartree_to_kcal_mol(pt.energy - reactant.energy)
            if all(pt.thermo_qh is not None for pt in ends_min.values()):
                e_ts = energy + ts_point.thermo_qh.zpe
                v_f = e_ts - (ends_min["reactant"].energy + ends_min["reactant"].thermo_qh.zpe)  # type: ignore[union-attr]
                v_r = e_ts - (ends_min["product"].energy + ends_min["product"].thermo_qh.zpe)  # type: ignore[union-attr]
                if v_f > 0 and v_r > 0:
                    tun["eckart"] = tunneling.eckart(
                        float(imag[0]), v_f, v_r, c.hypothesis.temperature_k
                    )
                    tun["forward_barrier_kcal_mol"] = units.hartree_to_kcal_mol(v_f)
                    tun["reverse_barrier_kcal_mol"] = units.hartree_to_kcal_mol(v_r)
        if "eckart" not in tun:
            tun["eckart_note"] = "Eckart not computed: IRC endpoints missing or not ZPE-corrected"

    return PathwayAnalysis(
        pathway=p,
        evidence=evidence,
        ts=ts_point,
        barrier_kcal_mol=barrier,
        barrier_by_step=by_step,
        numerical_sigma=numerical_sigma,
        thermo_sigma=thermo_sigma,
        noise_epsilon=float(eps),
        noise_floor_cm=floor_cm,
        step=h,
        tunneling=tun,
        irc=irc_info,
        notes=notes,
        n_gradients=counter.calls,
        electronic_barrier_kcal_mol=units.hartree_to_kcal_mol(float(energy) - reactant.energy),
        endpoint_minima_kcal_mol=endpoint_minima,
    )


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


@dataclass
class CampaignResult:
    campaign: Campaign
    limit_kcal_mol: float
    analyses: dict[str, PathwayAnalysis]
    verdicts: dict[str, Any]
    summary: dict[str, Any]
    out_dir: Path


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

    described: dict[str, Any] = {}

    def run_variant(
        name: str, variant: Mapping[str, Any] | None, starts: Mapping[str, FloatArray] | None
    ) -> tuple[ReactantSet, dict[str, PathwayAnalysis]]:
        variant = dict(variant or {})
        seed = variant.pop("rotation_seed", None)
        rotation = None if seed is None else _random_rotation(int(seed))
        reactant_coords = (
            c.reactant_coords if rotation is None else _rotate(c.reactant_coords, rotation)
        )
        if rotation is not None:
            # Rigid reorientation of every input geometry (spec §6 rotation test).
            source = starts if starts is not None else c.ts_guesses
            starts = {k: _rotate(v, rotation) for k, v in source.items()}
        provider = build_provider(c.model_spec, variant or None, c.system_info())
        try:
            if name == "baseline":
                described.update(provider.describe())
            settings = _provenance(
                c, c.reactant_coords, name, env, provider.describe()
            ).settings_fingerprint()
            species = []
            for i in range(len(c.hypothesis.reactants)):
                atoms = _species_atoms(c, i)
                point, ok = _minimum(
                    provider.fragment(i),
                    c,
                    atoms,
                    reactant_coords[3 * c.species_slice(i).start : 3 * c.species_slice(i).stop],
                    1e-3,
                    store,
                    f"{settings}/species{i}",
                    c.symmetry_species[i],
                )
                if not ok:
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
            return reactant, analyses
        finally:
            provider.close()

    reactant, analyses = run_variant("baseline", None, None)
    evaluated = {n: a.completed for n, a in analyses.items()}
    for a in analyses.values():
        a.evidence = a.evidence.replace(competitors_evaluated=evaluated)

    budgets = {n: a.uncertainty(c.uncertainty) for n, a in analyses.items()}
    pre_gates = {n: [g(a.evidence) for g in G.ALL_GATES[:7]] for n, a in analyses.items()}
    baseline = decide([_outcome(a, pre_gates[n], budgets[n]) for n, a in analyses.items()], limit)

    # Robustness: rerun from the baseline TS geometries with each variant model.
    starts = {n: a.ts.coords for n, a in analyses.items() if a.ts is not None}
    variant_verdicts: dict[str, dict[str, str]] = {n: {} for n in analyses}
    variant_barriers: dict[str, dict[str, float | None]] = {n: {} for n in analyses}
    variant_reasons: dict[str, dict[str, str]] = {n: {} for n in analyses}
    robustness = dict(c.robustness)
    if _is_dft(c.model_spec):
        # DFT grids are not rotationally invariant; G8 includes a rotation test.
        robustness.setdefault("rotation", {"rotation_seed": ROTATION_SEED})
    for vname, vspec in robustness.items():
        _, va = run_variant(vname, vspec, starts)
        for a in va.values():
            a.evidence = a.evidence.replace(competitors_evaluated=evaluated)
        vb = {n: a.uncertainty(c.uncertainty) for n, a in va.items()}
        vg = {n: [g(a.evidence) for g in G.ALL_GATES[:7]] for n, a in va.items()}
        vd = decide([_outcome(a, vg[n], vb[n]) for n, a in va.items()], limit)
        for n in analyses:
            variant_verdicts[n][vname] = vd[n].verdict.value
            variant_barriers[n][vname] = va[n].barrier_kcal_mol
            variant_reasons[n][vname] = vd[n].reason

    final_gates: dict[str, list[G.GateResult]] = {}
    for n, a in analyses.items():
        a.evidence = a.evidence.replace(
            baseline_verdict=baseline[n].verdict.value, robustness_verdicts=variant_verdicts[n]
        )
        final_gates[n] = G.evaluate_all(a.evidence)
    # sigma_grid is measured from the rotation and grid variants when available.
    for n, a in analyses.items():
        shifts = [
            abs(b - a.barrier_kcal_mol)
            for k in ("rotation", "grid")
            if (b := variant_barriers[n].get(k)) is not None and a.barrier_kcal_mol is not None
        ]
        if shifts:
            measured = max(shifts)
            stated = c.uncertainty.get("grid")
            stated_value = stated.value_kcal_mol if stated is not None else None
            value = max(measured, stated_value or 0.0)
            terms = dict(c.uncertainty)
            terms["grid"] = Term(
                "grid",
                value,
                "measured",
                f"max barrier shift under rotation/grid variants {measured:.3f}",
            )
            budgets[n] = a.uncertainty(terms)
    outcomes = {n: _outcome(a, final_gates[n], budgets[n]) for n, a in analyses.items()}
    verdicts = decide(list(outcomes.values()), limit)

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

    pathway_summaries: dict[str, Any] = {}
    for n, a in analyses.items():
        v = verdicts[n]
        summary = {
            "pathway": n,
            "campaign": str(c.path.resolve()),
            "hypothesis": c.hypothesis.id,
            "verdict": v.verdict.value,
            "reason": v.reason,
            "barrier_kcal_mol": a.barrier_kcal_mol,
            "electronic_barrier_kcal_mol": a.electronic_barrier_kcal_mol,
            "endpoint_minima_kcal_mol": a.endpoint_minima_kcal_mol,
            "sigma_kcal_mol": v.sigma_kcal_mol,
            "limit_kcal_mol": limit,
            "uncertainty": [
                {"component": comp, "value": val, "source": src, "note": note}
                for comp, val, src, note in budgets[n].breakdown()
            ],
            "gates": [dataclasses.asdict(g) for g in final_gates[n]],
            "imaginary_frequency_cm": float(a.ts.frequencies_cm[0]) if a.ts is not None else None,
            "frequencies_cm": [float(f) for f in a.ts.frequencies_cm] if a.ts is not None else [],
            "noise_epsilon_eh_bohr": None if math.isnan(a.noise_epsilon) else a.noise_epsilon,
            "noise_floor_cm": None if math.isnan(a.noise_floor_cm) else a.noise_floor_cm,
            "fd_step_bohr": None if math.isnan(a.step) else a.step,
            "barrier_by_step": a.barrier_by_step,
            "robustness": {
                k: {
                    "verdict": variant_verdicts[n][k],
                    "barrier_kcal_mol": variant_barriers[n][k],
                    "reason": variant_reasons[n][k],
                }
                for k in variant_verdicts[n]
            },
            "tunneling": a.tunneling,
            "irc": a.irc,
            "notes": a.notes,
            "n_gradients": a.n_gradients,
        }
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
        "rankings": rankings,
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
    "CampaignError",
    "CampaignResult",
    "Verdict",
    "analyze_pathway",
    "load_campaign",
    "run_campaign",
]
