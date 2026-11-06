"""Implicit-solvent (PCM) model settings (solvent "as hypothesized").

Surface discretization: SwiG (A. W. Lange and J. M. Herbert, J. Chem. Phys.
133, 244111 (2010)), done inside cuEST. Gaussian widths zeta for each Lebedev
order are the York-Karplus optimized values (D. M. York and M. Karplus,
J. Phys. Chem. A 103, 11060 (1999)), as tabulated in PySCF's pcm module.
Cavity radii: Bondi van der Waals radii (A. Bondi, J. Phys. Chem. 68, 441
(1964)) with H = 1.1 A (R. S. Rowland and R. Taylor, J. Phys. Chem. 100, 7384
(1996)), scaled by 1.2 (Q-Chem and PySCF default).

PCM is electrostatic only here and is not treated as evidence
of explicit solvent behaviour, and non-electrostatic terms remain an open
question.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from saddle import units

ZETA = {
    6: 4.84566077868, 14: 4.86458714334, 26: 4.85478226219, 38: 4.90105812685,
    50: 4.89250673295, 86: 4.89741372580, 110: 4.90101060987, 146: 4.89825187392,
    170: 4.90685517725, 194: 4.90337644248, 302: 4.90498088169, 350: 4.86879474832,
    434: 4.90567349080, 590: 4.90624071359, 770: 4.90656435779, 974: 4.90685167998,
}  # fmt: skip

BONDI_ANGSTROM = {
    "H": 1.10, "He": 1.40, "Li": 1.82, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "Ne": 1.54, "Na": 2.27, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75,
    "Ar": 1.88, "Br": 1.85, "I": 1.98,
}  # fmt: skip

SOLVENT_EPSILON = {"water": 78.3553, "dmso": 46.826, "acetonitrile": 35.688, "methanol": 32.613}


class SolvationError(ValueError):
    pass


@dataclass(frozen=True)
class PCMModel:
    epsilon: float
    method: Literal["C-PCM", "COSMO"] = "C-PCM"
    lebedev_points: int = 302
    radii_scale: float = 1.2
    weight_cutoff: float = 1e-10
    convergence_threshold: float = 1e-10
    max_iterations: int = 500

    def __post_init__(self) -> None:
        if not self.epsilon > 1.0:
            raise SolvationError("PCM dielectric constant must exceed 1")
        if self.lebedev_points not in ZETA:
            raise SolvationError(f"no York-Karplus zeta for {self.lebedev_points} points")
        if self.method not in ("C-PCM", "COSMO"):
            raise SolvationError(f"unknown PCM method {self.method}")

    @property
    def x_prefactor(self) -> float:
        """f(eps) = (eps - 1) / (eps + x): x = 0 for C-PCM, 1/2 for COSMO."""
        return 0.0 if self.method == "C-PCM" else 0.5

    def radii_bohr(self, symbols: Sequence[str]) -> list[float]:
        try:
            return [self.radii_scale * BONDI_ANGSTROM[s] / units.BOHR_TO_ANGSTROM for s in symbols]
        except KeyError as exc:
            raise SolvationError(f"no Bondi radius for {exc.args[0]}") from exc


def solvent_model(solvent: str | None, **kwargs: object) -> PCMModel | None:
    """PCMModel for a named solvent (hypothesis `solvent:`), or None for gas phase."""
    if solvent is None:
        return None
    key = solvent.lower()
    if key not in SOLVENT_EPSILON:
        raise SolvationError(f"unknown solvent {solvent!r}; known: {sorted(SOLVENT_EPSILON)}")
    return PCMModel(epsilon=SOLVENT_EPSILON[key], **kwargs)  # type: ignore[arg-type]
