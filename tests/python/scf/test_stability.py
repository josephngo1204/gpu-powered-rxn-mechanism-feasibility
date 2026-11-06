"""Internal (RHF -> RHF) SCF stability for gate G2."""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

pytest.importorskip("cupy")
pytest.importorskip("pyscf")

from pyscf import gto, scf
from pyscf.soscf import newton_ah

from saddle import units
from saddle.basis import load_basis
from saddle.cuest import Handle, MathMode
from saddle.grids import GridSpec
from saddle.scf.rhf import SCFError, rks

pytestmark = pytest.mark.gpu

SYMBOLS = ["O", "H", "H"]
XYZ = (
    np.array([[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]]) / units.BOHR_TO_ANGSTROM
)


def ours(**kw: object) -> object:
    with Handle(MathMode.NATIVE_FP64) as h:
        return rks(
            h,
            load_basis("def2-SVP", SYMBOLS),
            load_basis("def2-universal-jkfit", SYMBOLS),
            SYMBOLS,
            XYZ.ravel(),
            stability=True,
            orbital_gradient_tol=1e-10,
            **kw,  # type: ignore[arg-type]
        )


def pyscf_lowest(n: int) -> np.ndarray:
    mol = gto.M(
        atom=[(s, tuple(x)) for s, x in zip(SYMBOLS, XYZ, strict=True)],
        unit="Bohr",
        basis="def2-svp",
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit(auxbasis="def2-universal-jkfit")
    mf.conv_tol = 1e-12
    mf.kernel()
    _, hop, hdiag = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=False)
    op = spla.LinearOperator(
        (hdiag.size, hdiag.size), matvec=lambda v: 2.0 * hop(v.ravel()).real.ravel()
    )
    vals = spla.eigsh(op, k=n, which="SA", tol=1e-10)[0]
    return np.sort(vals)


def test_rhf_stability_roots_match_pyscf_up_to_normalization() -> None:
    r = ours()
    assert r.orbitals_stable is True  # type: ignore[attr-defined]
    eigs = r.stability_eigenvalues  # type: ignore[attr-defined]
    ref = pyscf_lowest(3)
    ratio = ref / eigs
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-4)
    assert ratio[0] == pytest.approx(round(ratio[0]), rel=1e-4)


def test_dft_stability_runs_and_is_stable() -> None:
    r = ours(functional="PBE", grid=GridSpec(75, 302))
    assert r.orbitals_stable is True  # type: ignore[attr-defined]
    assert r.stability_eigenvalues[0] > 0  # type: ignore[attr-defined, index]


def test_stability_requires_converged_scf() -> None:
    with Handle(MathMode.NATIVE_FP64) as h, pytest.raises(SCFError, match="unconverged"):
        rks(
            h,
            load_basis("def2-SVP", SYMBOLS),
            load_basis("def2-universal-jkfit", SYMBOLS),
            SYMBOLS,
            XYZ.ravel(),
            stability=True,
            max_iterations=2,
        )
