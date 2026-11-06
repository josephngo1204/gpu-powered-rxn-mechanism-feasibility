import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
pytest.importorskip("pyscf")

from pyscf import gto

from saddle import elements, units
from saddle.basis import load_basis
from saddle.crosscheck.pyscf_map import from_pyscf, permutation, phases, to_pyscf
from saddle.cuest import Handle, MathMode
from saddle.cuest.basis import AOBasis, PairList
from saddle.cuest.oei import OEIntPlan

pytestmark = pytest.mark.gpu

SYMBOLS = ["C", "Cl", "H", "H", "H"]
XYZ = (
    np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.785],
            [1.027, 0.0, -0.363],
            [-0.513, 0.889, -0.363],
            [-0.513, -0.889, -0.363],
        ]
    )
    / units.BOHR_TO_ANGSTROM
)


@pytest.mark.parametrize("basis_name", ["def2-SVP", "def2-TZVP"])
def test_mapped_one_electron_matrices_match_elementwise(basis_name: str) -> None:
    basis = load_basis(basis_name, SYMBOLS)
    with Handle(MathMode.NATIVE_FP64) as h:
        ao = AOBasis(h, basis, SYMBOLS)
        pairs = PairList(h, ao, XYZ.ravel(), threshold_pq=1e-14)
        plan = OEIntPlan(h, ao, pairs)
        z = [elements.atomic_number(s) for s in SYMBOLS]
        s = cupy.asnumpy(plan.overlap())
        t = cupy.asnumpy(plan.kinetic())
        v = cupy.asnumpy(plan.nuclear_attraction(XYZ.ravel(), z))
        plan.close()
        pairs.close()
        ao.close()
    mol = gto.M(
        atom=[(a, tuple(x)) for a, x in zip(SYMBOLS, XYZ, strict=True)],
        unit="Bohr",
        basis=basis_name.lower(),
        verbose=0,
    )
    perm = permutation(basis, SYMBOLS, mol)
    sign = phases(s, mol.intor("int1e_ovlp"), perm)
    for ours, ref in ((s, "int1e_ovlp"), (t, "int1e_kin"), (v, "int1e_nuc")):
        np.testing.assert_allclose(to_pyscf(ours, perm, sign), mol.intor(ref), atol=1e-9)
    np.testing.assert_allclose(from_pyscf(to_pyscf(s, perm, sign), perm, sign), s, atol=0)
