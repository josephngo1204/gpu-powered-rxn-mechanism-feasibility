"""RKS against PySCF in the converged-grid limit."""

import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("pyscf")

from pyscf import dft, gto

from saddle import units
from saddle.basis import load_basis
from saddle.cuest import Handle, MathMode
from saddle.grids import GridSpec
from saddle.scf.rhf import SCFError, rks

pytestmark = pytest.mark.gpu

SYMBOLS = ["O", "H", "H"]
XYZ = (
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    / units.BOHR_TO_ANGSTROM
)
FINE = GridSpec(150, 974)
TOL = 2e-6


def pyscf_df_rks(xc: str) -> float:
    mol = gto.M(
        atom=[(s, tuple(x)) for s, x in zip(SYMBOLS, XYZ, strict=True)],
        unit="Bohr",
        basis="def2-svp",
        verbose=0,
    )
    mf = dft.RKS(mol).density_fit(auxbasis="def2-universal-jkfit")
    mf.xc = xc
    mf.grids.atom_grid = (150, 974)
    mf.grids.prune = None
    mf.conv_tol = 1e-11
    return float(mf.kernel())


def ours(functional: str, grid: GridSpec = FINE) -> object:
    with Handle(MathMode.NATIVE_FP64) as h:
        return rks(
            h,
            load_basis("def2-SVP", SYMBOLS),
            load_basis("def2-universal-jkfit", SYMBOLS),
            SYMBOLS,
            XYZ.ravel(),
            functional=functional,
            grid=grid,
        )


@pytest.mark.parametrize(("ours_name", "pyscf_name"), [("PBE", "pbe,pbe"), ("PBE0", "pbe0")])
def test_rks_matches_pyscf(ours_name: str, pyscf_name: str) -> None:
    r = ours(ours_name)
    assert r.converged  # type: ignore[attr-defined]
    assert r.energy == pytest.approx(pyscf_df_rks(pyscf_name), abs=TOL)  # type: ignore[attr-defined]


def test_grid_convergence_is_monotone_in_spread() -> None:
    e = [ours("PBE", g).energy for g in (GridSpec(75, 302), GridSpec(99, 590), FINE)]  # type: ignore[attr-defined]
    assert abs(e[1] - e[2]) < abs(e[0] - e[2]) + 1e-9
    assert abs(e[1] - e[2]) < 1e-4


def test_dft_without_grid_is_rejected() -> None:
    with Handle(MathMode.NATIVE_FP64) as h, pytest.raises(SCFError, match="grid"):
        rks(
            h,
            load_basis("def2-SVP", SYMBOLS),
            load_basis("def2-universal-jkfit", SYMBOLS),
            SYMBOLS,
            XYZ.ravel(),
            functional="PBE",
        )


def test_hf_is_not_an_xc_functional() -> None:
    from saddle.cuest.xc import functional_key

    with pytest.raises(ValueError, match="HF"):
        functional_key("HF")
    assert functional_key("wB97X-V") == "WB97X-V"


@pytest.mark.slow
def test_wb97x_v_gap_to_pyscf_is_the_long_range_df_convention() -> None:
    """The ~4e-5 Eh gap is -1/2 f_LRC times the long-range DF exchange difference.

    See tests/cross_code/test_range_separated_df.py; VV10 is
    not the cause (wB97X without VV10 shows the same gap).
    """
    r = ours("WB97X-V")
    assert r.converged  # type: ignore[attr-defined]
    mol = gto.M(
        atom=[(s, tuple(x)) for s, x in zip(SYMBOLS, XYZ, strict=True)],
        unit="Bohr",
        basis="def2-svp",
        verbose=0,
    )
    mf = dft.RKS(mol).density_fit(auxbasis="def2-universal-jkfit")
    mf.xc = "wb97x-v"
    for g in (mf.grids, mf.nlcgrids):
        g.atom_grid = (150, 974)
        g.prune = None
    mf.conv_tol = 1e-11
    gap = r.energy - float(mf.kernel())  # type: ignore[attr-defined]
    assert -5e-5 < gap < -2.5e-5
    assert r.nonlocal_energy > 0  # type: ignore[attr-defined]
