"""One-electron integrals against PySCF, using invariants that do not depend on
AO ordering or phase (cuEST orders solid harmonics 0, 1c, 1s, 2c, 2s, ...)."""

import numpy as np
import pytest
import scipy.linalg

cupy = pytest.importorskip("cupy")
pyscf = pytest.importorskip("pyscf")

from pyscf import gto

from saddle import elements, units
from saddle.basis import load_basis
from saddle.cuest import Handle, MathMode
from saddle.cuest.basis import AOBasis, PairList
from saddle.cuest.oei import OEIntPlan

pytestmark = pytest.mark.gpu

SYMBOLS = ["O", "H", "H"]
XYZ = (
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    / units.BOHR_TO_ANGSTROM
)
TOL = 1e-9


@pytest.fixture(scope="module")
def matrices() -> dict[str, np.ndarray]:
    basis = load_basis("def2-SVP", SYMBOLS)
    with Handle(MathMode.NATIVE_FP64) as h:
        ao = AOBasis(h, basis, SYMBOLS)
        pairs = PairList(h, ao, XYZ.ravel(), threshold_pq=1e-14)
        plan = OEIntPlan(h, ao, pairs)
        z = np.array([elements.atomic_number(s) for s in SYMBOLS], dtype=float)
        out = {
            "S": cupy.asnumpy(plan.overlap()),
            "T": cupy.asnumpy(plan.kinetic()),
            "V": cupy.asnumpy(plan.nuclear_attraction(XYZ.ravel(), z)),
        }
        plan.close()
        pairs.close()
        ao.close()
    mol = gto.M(
        atom=[(s, tuple(x)) for s, x in zip(SYMBOLS, XYZ, strict=True)],
        unit="Bohr",
        basis="def2-svp",
        cart=False,
    )
    out["S_ref"] = mol.intor("int1e_ovlp")
    out["T_ref"] = mol.intor("int1e_kin")
    out["V_ref"] = mol.intor("int1e_nuc")
    return out


def test_shapes_and_symmetry(matrices: dict[str, np.ndarray]) -> None:
    for k in ("S", "T", "V"):
        m = matrices[k]
        assert m.shape == (24, 24)
        np.testing.assert_allclose(m, m.T, atol=1e-12)


def test_overlap_is_normalized(matrices: dict[str, np.ndarray]) -> None:
    np.testing.assert_allclose(np.diag(matrices["S"]), 1.0, atol=1e-12)


def test_overlap_spectrum_matches_pyscf(matrices: dict[str, np.ndarray]) -> None:
    np.testing.assert_allclose(
        np.linalg.eigvalsh(matrices["S"]), np.linalg.eigvalsh(matrices["S_ref"]), atol=TOL
    )


@pytest.mark.parametrize("op", ["T", "V"])
def test_operator_spectra_match_pyscf(matrices: dict[str, np.ndarray], op: str) -> None:
    ours = scipy.linalg.eigh(matrices[op], matrices["S"], eigvals_only=True)
    ref = scipy.linalg.eigh(matrices[f"{op}_ref"], matrices["S_ref"], eigvals_only=True)
    np.testing.assert_allclose(ours, ref, atol=TOL, rtol=1e-10)


def test_nuclear_attraction_is_attractive(matrices: dict[str, np.ndarray]) -> None:
    assert np.all(np.diag(matrices["V"]) < 0)
