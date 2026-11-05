# Saddle report: c1-cl-ch3cl-gas

T = 298.15 K, target half-life 3600 s, feasibility limit ΔG‡ = 22.52 kcal/mol.
Code 6429ab7443f3 (dirty: false), GPU NVIDIA H200 NVL, cuEST 0.2.2.

## Verdicts

| Pathway | Verdict | ΔG‡ (kcal/mol) | 2σ | Failed gates | Reason |
|---|---|---|---|---|---|
| sn2 | **SUPPORTED** | 7.39 | 3.61 | none | 7.39 + 2σ below limit 22.52 |

## Pathway `sn2`

Imaginary frequency -463.3 cm⁻¹; gradient noise 5.2e-13 Eh/bohr; FD step 5.31e-04 bohr; noise floor 2.7 cm⁻¹; 619 gradients.

### Gates

| Gate | Result | Reason |
|---|---|---|
| G1 stationarity | pass | rms 5.78e-11, max 1.45e-10 Eh/bohr (limits 1e-05, 1.5e-05); translation/rotation part removed (max 1.6e-12) |
| G2 electronic quality | pass | converged and stable |
| G3 Hessian quality | pass | all quality targets met |
| G4 saddle order | pass | one imaginary mode -463.3 cm^-1 |
| G5 mode character | pass | reactive bonds carry 100% of the mode |
| G6 connectivity | pass | IRC connects reactants and intended products |
| G7 competition | pass | all declared competitors evaluated |
| G8 robustness | pass | verdict stable |

### Uncertainty (kcal/mol)

| Component | σ | Source | Note |
|---|---|---|---|
| numerical | 0.000 | measured | barrier at h vs 2h |
| grid | 0.100 | measured | max barrier shift under rotation/grid variants 0.006 |
| basis | 1.000 | estimate | def2-SVPD search profile; def2-TZVPD single points pending |
| method | 1.500 | estimate | DFT vs CCSD(T) for SN2 barriers; refine from C3 benchmarks |
| thermo | 0.003 | measured | qRRHO vs RRHO |
| solvent | 0.000 | estimate | gas phase |
| **total** | 1.806 | | |

### Sensitivity of the barrier

| Setting | ΔG‡ (kcal/mol) | Verdict |
|---|---|---|
| FD step 0.000531 bohr | 7.391 | |
| FD step 0.00106 bohr | 7.391 | |
| variant: functional | 6.060 | SUPPORTED (6.06 + 2σ below limit 22.52) |
| variant: grid | 7.385 | SUPPORTED (7.39 + 2σ below limit 22.52) |
| variant: precision | 7.391 | SUPPORTED (7.39 + 2σ below limit 22.52) |
| variant: rotation | 7.385 | SUPPORTED (7.39 + 2σ below limit 22.52) |

### Tunneling (reported separately, not applied)

- Wigner κ = 1.208
- Eckart κ = 1.250 (reverse barrier 13.60 kcal/mol, ZPE-corrected)

### IRC

- forward: 2 points, stopped on energy_rise, endpoint reactants (converged: True)
- reverse: 2 points, stopped on energy_rise, endpoint intended products (converged: True)

### Notes

- noise 5.2e-13 below probe resolution; using 1e-10
- G2: internal RHF->RHF stability, lowest A+B eigenvalue 0.2106 Eh; RHF->UHF not tested
