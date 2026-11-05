# Saddle: a cuEST mechanism feasibility engine

Saddle takes a proposed organic reaction mechanism and returns a verdict backed by evidence. It
finds the transition state (TS), proves with an IRC that it connects the intended species,
computes the free-energy barrier with an uncertainty budget, and checks that barrier against a
feasibility limit set by the temperature and target half-life. Electronic structure (DF-RHF/RKS,
wB97X-V, VV10, C-PCM, analytic gradients) runs on NVIDIA cuEST. Hessians come from finite
differences of analytic gradients, sized by a measured gradient-noise probe.

- Specification: [`spec.md`](spec.md)
- Implementation plan and status: [`tasks/plan.md`](tasks/plan.md)
- Architecture decisions: [`docs/adr/`](docs/adr)
- Results baseline (MVP): [`results/mvp/`](results/mvp) (open `dashboard.html`)

## MVP status (2026-11-05)

| Campaign | System | Verdict | Key numbers |
|---|---|---|---|
| fake | H-transfer on a fake double-well surface, two competing products | `to_c` SUPPORTED, `to_d` INFEASIBLE, `to_c` ranked lower | CPU only, 28 s |
| smoke | HCN → HNC, wB97X-V/def2-SVP | INFEASIBLE | ΔG‡ 42.4 ± 2.7 kcal/mol (from the plan; the run directory was not kept, so it is not in the dashboard) |
| **C1** | Cl⁻ + CH₃Cl SN2, gas, wB97X-V/def2-SVPD | **SUPPORTED** | All G1–G8 gates pass. TS +2.47 vs CCSD(T) +2.52 kcal/mol; complex −11.67 vs −10.42. ΔG‡ 7.4 ± 1.8 kcal/mol, 83 min on one MIG slice |
| C2 | Cl⁻ + CH₃Cl SN2, C-PCM water | in progress | — |
| C3 | X⁻ + CH₃CH₂X (X = F, Cl): SN2 vs anti-E2 vs syn-E2 | inputs ready, not run | — |

Delivered: spec stages 0–4 (Phases A–F in the plan) and the molecular negative controls.
Still open: the rest of the reaction ladder (C2–C8), UKS/open-shell, and the cuEST performance
benchmarks (`saddle bench`).

## Setup

This machine has CUDA ≤ 12.8 (driver 570), so Saddle uses the Python cuEST bindings with CuPy
([ADR 0001](docs/adr/0001-cuda12-python-cuest-adapter.md)).

```bash
mamba env create -f environment.yml
mamba activate saddle
pip install nvidia-cuest-cu12==0.2.2 && pip install --no-build-isolation -e ".[dev]"
cmake --preset cpu-debug && cmake --build --preset cpu-debug    # C++ unit tests
```

The repo files are owned by another user, so run git as `git -c safe.directory=$PWD ...`.

## Usage

```bash
saddle run campaigns/c1/c1_sn2.yaml --out runs/c1          # GPU; hours for real chemistry
saddle run campaigns/fake/h_transfer.yaml --out runs/fake  # CPU; seconds
saddle report runs/c1 --format markdown                    # text report for one run
saddle dashboard runs --out dashboard.html                 # HTML dashboard over every run
saddle noise runs/c1/sn2/ts.h5 --modes lowest:6            # gradient noise along TS modes
saddle hessian runs/c1/sn2/ts.h5 --step auto               # recompute a TS Hessian with checks
```

A campaign YAML names a hypothesis file (species, charges, atom maps, competing pathways), the
level of theory, a TS guess per pathway, and the robustness variants. Every pathway goes through
gates G1–G8: stationarity, electronic quality, Hessian quality, saddle order, mode character, IRC
connectivity, competition, and robustness. A barrier only gets a verdict when every gate passes.

`saddle dashboard` writes one self-contained HTML file with no dependencies and no network
access. It contains barrier-vs-limit plots with 2σ intervals, a gate matrix, energy profiles with
reference overlays from `data/references.yaml`, uncertainty budgets, and robustness variants.
It supports light and dark themes, and every chart also has a table view.

## Tests

```bash
pytest -m "not gpu and not slow"         # CPU suite, ~15 s
pytest -m gpu                            # needs the GPU and cuEST
pytest tests/chemistry -m ladder         # reaction ladder; set SADDLE_<ID>_RUN to reuse a finished run
ctest --test-dir build/cpu-debug         # C++ under ASan/UBSan
ruff check python tests && mypy --strict python/saddle
```

## Layout

```
cpp/          C++ core: FD Hessian, noise probe, vibrations, RS-RFO / P-RFO search, IRC, fake surfaces
python/saddle/
  cuest/      the only code that imports cuEST (handles, basis, integrals, DF, grids, XC, PCM)
  scf/        RHF/RKS with DIIS, gradients, stability; CuestModel.evaluate(coords)
  analysis/   thermochemistry, conformers, tunneling, KIE, kinetics, uncertainty, Hessian quality
  gates/      G1–G8 and the verdict engine
  store/      provenance, HDF5 results, SQLite task and stencil stores
  workflows/  pipeline, markdown report, HTML dashboard
campaigns/    reaction-ladder inputs (fake, smoke, c1, c2, c3)
data/         literature references with citations
results/mvp/  committed result summaries for the MVP (runs/ itself is not tracked)
```
