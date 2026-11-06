import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("pyscf")

from pyscf import gto, scf

from saddle import units
from saddle.basis import load_basis
from saddle.cuest import Handle, MathMode
from saddle.cuest.basis import AOBasis, PairList
from saddle.cuest.oei import OEIntPlan
from saddle.cuest.pcm import PCMIntPlan, PCMNotConverged
from saddle.scf.rhf import rhf
from saddle.solvation import PCMModel, SolvationError, solvent_model

pytestmark = pytest.mark.gpu

SYMBOLS = ["O", "H", "H"]
XYZ = (
    np.array([[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]]) / units.BOHR_TO_ANGSTROM
)


def test_model_validation() -> None:
    with pytest.raises(SolvationError):
        PCMModel(epsilon=1.0)
    with pytest.raises(SolvationError):
        PCMModel(epsilon=78.0, lebedev_points=303)
    assert solvent_model(None) is None
    assert solvent_model("water").epsilon == 78.3553  # type: ignore[union-attr]
    assert PCMModel(epsilon=2.0, method="COSMO").x_prefactor == 0.5


def _pyscf_cpcm() -> float:
    mol = gto.M(
        atom=[(s, tuple(x)) for s, x in zip(SYMBOLS, XYZ, strict=True)],
        unit="Bohr",
        basis="def2-svp",
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit(auxbasis="def2-universal-jkfit").PCM()
    mf.with_solvent.method = "C-PCM"
    mf.with_solvent.eps = 78.3553
    mf.with_solvent.lebedev_order = 29
    mf.conv_tol = 1e-11
    return float(mf.kernel())


@pytest.fixture(scope="module")
def solvated() -> object:
    with Handle(MathMode.NATIVE_FP64) as h:
        return rhf(
            h,
            load_basis("def2-SVP", SYMBOLS),
            load_basis("def2-universal-jkfit", SYMBOLS),
            SYMBOLS,
            XYZ.ravel(),
            solvent=PCMModel(epsilon=78.3553),
        )


def test_cpcm_rhf_matches_pyscf(solvated) -> None:  # type: ignore[no-untyped-def]
    """SwiG discretization differs slightly between the codes.

    On a fixed density the dielectric energies differ by 7.8e-6 Eh (0.08%,
    0.005 kcal/mol); the SCF energies by ~1e-5 Eh.
    """
    assert solvated.converged
    assert solvated.energy == pytest.approx(_pyscf_cpcm(), abs=2e-5)
    assert solvated.solvation_energy < 0


def test_non_convergence_is_raised_not_returned() -> None:
    with Handle(MathMode.NATIVE_FP64) as h:
        basis = load_basis("def2-SVP", SYMBOLS)
        ao = AOBasis(h, basis, SYMBOLS)
        pairs = PairList(h, ao, XYZ.ravel(), threshold_pq=1e-14)
        oei = OEIntPlan(h, ao, pairs)
        pcm = PCMIntPlan(h, oei, SYMBOLS, PCMModel(epsilon=78.3553, convergence_threshold=1e-14))
        import cupy

        d = cupy.eye(ao.num_ao)
        with pytest.raises(PCMNotConverged):
            pcm.potential(d, max_iterations=1)
        ok = pcm.potential(d)
        assert ok.converged and ok.iterations >= 1
        pcm.close()
        oei.close()
        pairs.close()
        ao.close()
