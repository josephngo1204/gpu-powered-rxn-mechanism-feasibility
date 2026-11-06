"""Density fitting of range-separated exchange differs between cuEST and PySCF.

DF and range-separated conventions are treated as explicit test variables.
On one fixed density, full-range DF exchange agrees with PySCF to 1e-10, while
long-range (erf) DF exchange differs from both PySCF DF and exact exchange by
about 1e-4.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
pytest.importorskip("pyscf")

from pyscf import gto, scf

from saddle import units
from saddle.basis import load_basis
from saddle.crosscheck.pyscf_map import permutation, phases
from saddle.cuest import Handle, MathMode
from saddle.cuest.basis import AOBasis, PairList
from saddle.cuest.df import DFIntPlan
from saddle.cuest.oei import OEIntPlan

pytestmark = pytest.mark.gpu

SYMBOLS = ["O", "H", "H"]
XYZ = (
    np.array([[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]]) / units.BOHR_TO_ANGSTROM
)
OMEGA = 0.3


@pytest.fixture(scope="module")
def exchange_energies() -> dict[str, float]:
    mol = gto.M(
        atom=[(s, tuple(x)) for s, x in zip(SYMBOLS, XYZ, strict=True)],
        unit="Bohr",
        basis="def2-svp",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    c_occ = mf.mo_coeff[:, :5]
    dm = 2 * c_occ @ c_occ.T
    mfdf = scf.RHF(mol).density_fit(auxbasis="def2-universal-jkfit")
    _, k_df_full = mfdf.with_df.get_jk(dm, hermi=1, with_j=False)
    _, k_df_lr = mfdf.with_df.get_jk(dm, hermi=1, with_j=False, omega=OMEGA)
    out = {
        "pyscf_df_full": float(np.vdot(dm, k_df_full)) / 2,
        "pyscf_df_lr": float(np.vdot(dm, k_df_lr)) / 2,
        "exact_lr": float(np.vdot(dm, mf.get_k(mol, dm, omega=OMEGA))) / 2,
    }
    basis = load_basis("def2-SVP", SYMBOLS)
    with Handle(MathMode.NATIVE_FP64) as h:
        ao = AOBasis(h, basis, SYMBOLS)
        aux = AOBasis(h, load_basis("def2-universal-jkfit", SYMBOLS), SYMBOLS)
        pairs = PairList(h, ao, XYZ.ravel(), threshold_pq=1e-14)
        oei = OEIntPlan(h, ao, pairs)
        perm = permutation(basis, SYMBOLS, mol)
        sign = phases(cupy.asnumpy(oei.overlap()), mol.intor("int1e_ovlp"), perm)
        c = np.empty_like(c_occ)
        c[perm] = c_occ * sign[:, None]
        rows = cupy.asarray(np.ascontiguousarray(c.T))
        d = 2 * c @ c.T
        for key, frac, lrc, w in (("cuest_df_full", 1, 0, 0), ("cuest_df_lr", 0, 1, OMEGA)):
            df = DFIntPlan(
                h, ao, aux, pairs, exchange_fraction=frac, lrc_exchange_fraction=lrc, lrc_omega=w
            )
            out[key] = float(np.vdot(d, cupy.asnumpy(df.exchange(rows))))
            df.close()
        oei.close()
        pairs.close()
        aux.close()
        ao.close()
    return out


def test_full_range_df_exchange_matches_pyscf(exchange_energies: dict[str, float]) -> None:
    e = exchange_energies
    assert e["cuest_df_full"] == pytest.approx(e["pyscf_df_full"], abs=1e-10)


def test_long_range_df_exchange_convention_differs(exchange_energies: dict[str, float]) -> None:
    e = exchange_energies
    gap = e["cuest_df_lr"] - e["exact_lr"]
    assert abs(e["pyscf_df_lr"] - e["exact_lr"]) < 1e-7
    assert 5e-5 < abs(gap) < 2e-4
