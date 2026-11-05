import numpy as np
import pytest

pytest.importorskip("cupy")

from saddle import units
from saddle.basis import load_basis
from saddle.cuest import Handle, MathMode
from saddle.cuest.basis import AOBasis, PairList

pytestmark = pytest.mark.gpu

WATER = ["O", "H", "H"]
XYZ = (
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]).ravel()
    / units.BOHR_TO_ANGSTROM
)


def test_ao_basis_matches_python_counts() -> None:
    basis = load_basis("def2-SVP", WATER)
    with Handle(MathMode.NATIVE_FP64) as h:
        ao = AOBasis(h, basis, WATER)
        assert ao.num_ao == basis.n_functions(WATER) == 24
        assert ao.num_shell == sum(len(basis.shells[s]) for s in WATER)
        assert ao.num_atom == 3
        assert ao.max_l == 2
        assert ao.is_pure
        ao.close()
        ao.close()


def test_pair_list_builds_for_geometry() -> None:
    basis = load_basis("def2-SVP", WATER)
    with Handle(MathMode.NATIVE_FP64) as h:
        ao = AOBasis(h, basis, WATER)
        pairs = PairList(h, ao, XYZ, threshold_pq=1e-14)
        pairs.close()
        ao.close()


def test_pair_list_rejects_wrong_coordinate_count() -> None:
    basis = load_basis("def2-SVP", WATER)
    with Handle(MathMode.NATIVE_FP64) as h:
        ao = AOBasis(h, basis, WATER)
        with pytest.raises(ValueError, match="coordinates"):
            PairList(h, ao, XYZ[:6], threshold_pq=1e-14)
        with pytest.raises(ValueError, match="threshold"):
            PairList(h, ao, XYZ, threshold_pq=-1.0)
        ao.close()
