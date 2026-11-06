"""Analytic gradients against central finite differences of the energy.

Target: RMS error <= 1e-6 Eh/bohr. Each method isolates terms: HF covers
the one-electron, overlap and J/K terms; PBE the XC term (including any
grid-weight contribution); wB97X-V adds long-range K and VV10;
C-PCM adds the solvation term.
"""

import numpy as np
import pytest

pytest.importorskip("cupy")

from saddle import units
from saddle.basis import load_basis
from saddle.cuest import Handle, MathMode
from saddle.grids import GridSpec
from saddle.scf.rhf import SCFError, nuclear_repulsion, nuclear_repulsion_gradient, rks
from saddle.solvation import PCMModel

pytestmark = pytest.mark.gpu

SYMBOLS = ["O", "H", "H"]
XYZ = (
    np.array([[0.02, -0.01, 0.1173], [0.0, 0.7672, -0.4592], [0.01, -0.7472, -0.4792]])
    / units.BOHR_TO_ANGSTROM
).ravel()
STEP = 1e-3
TOL_RMS = 1e-6


def run(xyz: np.ndarray, **kw: object) -> object:
    with Handle(MathMode.NATIVE_FP64) as h:
        return rks(
            h,
            load_basis("def2-SVP", SYMBOLS),
            load_basis("def2-universal-jkfit", SYMBOLS),
            SYMBOLS,
            xyz,
            orbital_gradient_tol=1e-10,
            energy_tol=1e-12,
            **kw,  # type: ignore[arg-type]
        )


def fd_gradient(**kw: object) -> np.ndarray:
    g = np.zeros(XYZ.size)
    for i in range(XYZ.size):
        plus, minus = XYZ.copy(), XYZ.copy()
        plus[i] += STEP
        minus[i] -= STEP
        g[i] = (run(plus, **kw).energy - run(minus, **kw).energy) / (2 * STEP)  # type: ignore[attr-defined]
    return g.reshape(-1, 3)


def check(**kw: object) -> None:
    analytic = run(XYZ, gradient=True, **kw).gradient  # type: ignore[attr-defined]
    numeric = fd_gradient(**kw)
    err = analytic - numeric
    rms = float(np.sqrt(np.mean(err**2)))
    assert rms <= TOL_RMS, f"rms {rms:.2e}\nanalytic\n{analytic}\nfd\n{numeric}"
    assert np.abs(analytic.sum(axis=0)).max() < 1e-6


def test_nuclear_repulsion_gradient_matches_fd() -> None:
    z = [8, 1, 1]
    g = nuclear_repulsion_gradient(XYZ, z)
    for i in range(XYZ.size):
        p, m = XYZ.copy(), XYZ.copy()
        p[i] += 1e-5
        m[i] -= 1e-5
        assert g.ravel()[i] == pytest.approx(
            (nuclear_repulsion(p, z) - nuclear_repulsion(m, z)) / 2e-5, abs=1e-8
        )


def test_rhf_gradient() -> None:
    check()


def test_pbe_gradient_including_grid_weights() -> None:
    check(functional="PBE", grid=GridSpec(75, 302))


@pytest.mark.slow
def test_wb97x_v_gradient() -> None:
    check(functional="WB97X-V", grid=GridSpec(75, 302))


def test_cpcm_gradient() -> None:
    check(solvent=PCMModel(epsilon=78.3553))


def test_gradient_requires_converged_scf() -> None:
    with Handle(MathMode.NATIVE_FP64) as h, pytest.raises(SCFError, match="unconverged"):
        rks(
            h,
            load_basis("def2-SVP", SYMBOLS),
            load_basis("def2-universal-jkfit", SYMBOLS),
            SYMBOLS,
            XYZ,
            gradient=True,
            max_iterations=2,
        )


@pytest.mark.slow
def test_wb97x_v_gradient_with_separate_vv10_grid() -> None:
    check(functional="WB97X-V", grid=GridSpec(75, 302), nonlocal_grid=GridSpec(50, 194))
