import numpy as np
import pytest

pytest.importorskip("cupy")

from saddle import units
from saddle.cuest import Handle, MathMode
from saddle.cuest.grid import MolecularGrid
from saddle.grids import GridSpec

pytestmark = pytest.mark.gpu

SYMBOLS = ["O", "H", "H"]
XYZ = (
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]).ravel()
    / units.BOHR_TO_ANGSTROM
)


@pytest.mark.parametrize("spec", [GridSpec(75, 302), GridSpec(99, 590)])
def test_unpruned_molecular_grid_point_count(spec: GridSpec) -> None:
    with Handle(MathMode.NATIVE_FP64) as h:
        grid = MolecularGrid(h, SYMBOLS, XYZ, spec)
        assert grid.num_atom == 3
        assert grid.num_point == 3 * spec.radial * spec.angular
        assert grid.max_angular_point == spec.angular
        grid.close()
        grid.close()


def test_geometry_size_is_checked() -> None:
    with Handle(MathMode.NATIVE_FP64) as h, pytest.raises(ValueError, match="coordinates"):
        MolecularGrid(h, SYMBOLS, XYZ[:6], GridSpec(75, 302))
