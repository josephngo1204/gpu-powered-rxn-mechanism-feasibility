import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("pyscf")

from pyscf import gto, scf

from saddle import units
from saddle.basis import load_basis
from saddle.cuest import Handle, MathMode
from saddle.scf.rhf import SCFError, nuclear_repulsion, rhf

pytestmark = pytest.mark.gpu

SYMBOLS = ["O", "H", "H"]
XYZ = (
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    / units.BOHR_TO_ANGSTROM
)


def pyscf_df_rhf(symbols: list[str], xyz: np.ndarray, basis: str, aux: str) -> float:
    mol = gto.M(
        atom=[(s, tuple(x)) for s, x in zip(symbols, xyz, strict=True)],
        unit="Bohr",
        basis=basis,
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit(auxbasis=aux)
    mf.conv_tol = 1e-12
    return float(mf.kernel())


@pytest.fixture(scope="module")
def water() -> object:
    with Handle(MathMode.NATIVE_FP64) as h:
        return rhf(
            h,
            load_basis("def2-SVP", SYMBOLS),
            load_basis("def2-universal-jkfit", SYMBOLS),
            SYMBOLS,
            XYZ.ravel(),
        )


def test_nuclear_repulsion() -> None:
    assert nuclear_repulsion([0, 0, 0, 0, 0, 1.4], [1, 1]) == pytest.approx(1 / 1.4)


def test_rhf_converges(water) -> None:  # type: ignore[no-untyped-def]
    assert water.converged
    assert water.orbital_gradient < 1e-8
    assert water.iterations < 40


def test_rhf_matches_pyscf_density_fitted(water) -> None:  # type: ignore[no-untyped-def]
    ref = pyscf_df_rhf(SYMBOLS, XYZ, "def2-svp", "def2-universal-jkfit")
    assert water.energy == pytest.approx(ref, abs=1e-8)


def test_electron_count_and_aufbau(water) -> None:  # type: ignore[no-untyped-def]
    import cupy

    from saddle.basis import load_basis as lb
    from saddle.cuest.basis import AOBasis, PairList
    from saddle.cuest.oei import OEIntPlan

    with Handle(MathMode.NATIVE_FP64) as h:
        ao = AOBasis(h, lb("def2-SVP", SYMBOLS), SYMBOLS)
        pairs = PairList(h, ao, XYZ.ravel(), threshold_pq=1e-14)
        plan = OEIntPlan(h, ao, pairs)
        s = plan.overlap()
        plan.close()
        pairs.close()
        ao.close()
    assert float(cupy.vdot(water.density, s)) == pytest.approx(10.0, abs=1e-10)
    eps = water.orbital_energies
    assert eps[4] < 0 < eps[5]  # HOMO bound, LUMO above


def test_odd_electron_count_is_rejected() -> None:
    with Handle(MathMode.NATIVE_FP64) as h, pytest.raises(SCFError, match="even"):
        rhf(
            h,
            load_basis("def2-SVP", ["H"]),
            load_basis("def2-universal-jkfit", ["H"]),
            ["H"],
            [0, 0, 0],
        )
