# Saddle: GPU-accelerated reaction mechanism feasibility

Saddle takes a proposed reaction mechanism and decides, with quantified uncertainty, whether it can
happen under given conditions. For each proposed step it:

1. locates the transition state (TS) with quantum chemistry running on NVIDIA's cuEST GPU library,
2. proves with an intrinsic reaction coordinate (IRC) that the TS connects the intended reactants and products,
3. computes the free-energy barrier ΔG‡ with a full uncertainty budget, and
4. compares that barrier with the feasibility limit implied by the temperature and a target half-life.

A verdict is only issued when the TS passes eight validation gates. Missing evidence always fails a
gate, so a result can never pass because a check was skipped.

## Headline result: C1, Cl⁻ + CH₃Cl SN2

The identity SN2 reaction is a classic benchmark with a double-well energy profile (an ion–dipole
complex on each side of a central TS). Saddle reproduces it from scratch on one GPU slice:

| Quantity (kcal/mol, vs. separated reactants) | Saddle (ωB97X-V/def2-SVPD) | CCSD(T) reference |
|---|---|---|
| Transition state | **+2.47** | +2.52 |
| Ion–dipole complex | **−11.67** | −10.42 |

- **Verdict: SUPPORTED.** ΔG‡ = 7.4 ± 1.8 kcal/mol, well below the 22.5 kcal/mol limit for a 1 h half-life at 298 K.
- All eight gates (G1–G8) pass. Imaginary frequency −463 cm⁻¹, residual gradient 6 × 10⁻¹¹ Eh/bohr.
- The verdict is stable under changes of functional (M06-2X), grid, precision mode and molecular orientation.
- Wall time: 83 minutes on a single H200 MIG slice (1/7 of the GPU), 619 analytic gradients.

Full report: [`results/mvp/c1/report.md`](results/mvp/c1/report.md). Interactive dashboard:
[`results/mvp/dashboard.html`](results/mvp/dashboard.html).

## What I built

- **Electronic structure on the GPU (Python + CuPy + cuEST):** density-fitted RHF and RKS with DIIS,
  range-separated hybrid ωB97X-V with VV10 nonlocal correlation, C-PCM implicit solvation, analytic
  nuclear gradients, and orbital stability analysis. Cross-validated against PySCF to 10⁻¹⁰ Eh where
  conventions agree, with the remaining range-separated density-fitting gap traced to its root cause.
- **Numerical core (C++20 + Eigen, Python bindings via nanobind):** finite-difference Hessians from
  analytic gradients with an automatically sized step from a measured gradient-noise probe, vibrational
  analysis with translation/rotation projection, Davidson eigensolver, RS-RFO minimization, P-RFO
  saddle search, and IRC integration.
- **Validation gates and verdict engine:** stationarity, SCF quality and stability, Hessian quality,
  saddle order, reactive-mode character, IRC connectivity, competing pathways, and robustness variants.
- **Thermochemistry and kinetics:** RRHO with quasi-harmonic entropy, standard-state corrections,
  conformer ensembles, Wigner and Eckart tunneling, kinetic isotope effects, and a per-component
  uncertainty budget.
- **Reproducibility:** every result records code commit, library version, GPU and math mode; Hessian
  stencils run through a restartable SQLite task store; results go to HDF5 with a JSON summary.
- **Reporting:** markdown reports and a dependency-free HTML dashboard with barrier-vs-limit plots,
  gate matrix, energy profiles with reference overlays, and uncertainty budgets.
- **Testing:** 342 Python tests (unit, property-based, PySCF cross-code, negative controls built to
  fool the gates) and 93 C++ tests under AddressSanitizer/UBSan; `ruff` and `mypy --strict` clean.

## Roadmap

| Campaign | System | Status |
|---|---|---|
| C1 | Cl⁻ + CH₃Cl SN2, gas phase | **Done: SUPPORTED** |
| C2 | Cl⁻ + CH₃Cl SN2, C-PCM water | Inputs and tests ready |
| C3 | X⁻ + CH₃CH₂X (X = F, Cl): SN2 vs anti-E2 vs syn-E2 | Inputs and tests ready |

## Setup

Requires an NVIDIA GPU with CUDA 12.

```bash
mamba env create -f environment.yml
mamba activate saddle
pip install nvidia-cuest-cu12==0.2.2 && pip install --no-build-isolation -e ".[dev]"
cmake --preset cpu-debug && cmake --build --preset cpu-debug
```

## Usage

```bash
saddle run campaigns/c1/c1_sn2.yaml --out runs/c1          # GPU, about 1.5 h
saddle run campaigns/fake/h_transfer.yaml --out runs/fake  # CPU demo on a model surface, seconds
saddle report runs/c1 --format markdown
saddle dashboard runs --out dashboard.html
```

A campaign YAML names a hypothesis file (species, charges, atom maps, competing pathways), the level
of theory, a TS guess per pathway, and the robustness variants to test.

## Tests

```bash
pytest -m "not gpu and not slow"     # CPU suite, ~15 s
pytest -m gpu                        # requires a GPU and cuEST
ctest --test-dir build/cpu-debug     # C++ suite
ruff check python tests && mypy --strict python/saddle
```

## Layout

```
cpp/             C++ core: FD Hessian, noise probe, vibrations, RS-RFO / P-RFO, IRC, model surfaces
python/saddle/
  cuest/         GPU integrals, density fitting, grids, XC and PCM (the only code that touches cuEST)
  scf/           RHF/RKS, gradients, stability
  analysis/      thermochemistry, tunneling, KIE, kinetics, uncertainty, Hessian quality
  gates/         validation gates G1–G8 and the verdict engine
  store/         provenance, HDF5 results, SQLite task store
  workflows/     campaign input, pipeline, markdown report
  dashboard/     self-contained HTML dashboard
campaigns/       reaction inputs (c1, c2, c3, plus CPU demo and smoke tests)
data/            literature reference energies
results/mvp/     C1 results
```
