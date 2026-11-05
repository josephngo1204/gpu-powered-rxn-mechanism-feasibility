"""Barrier uncertainty budget (spec §6 "Uncertainty report").

sigma_total^2 = sigma_numerical^2 + sigma_grid^2 + sigma_basis^2
              + sigma_method^2 + sigma_thermo^2 + sigma_solvent^2

Values are in kcal/mol. A qualitative or missing component makes the budget
non-confident, which prevents a confident verdict; it is never treated as 0.
"""

import math
from dataclasses import dataclass
from typing import Literal

COMPONENTS = ("numerical", "grid", "basis", "method", "thermo", "solvent")

Source = Literal["measured", "benchmark", "estimate", "qualitative"]


@dataclass(frozen=True)
class Term:
    name: str
    value_kcal_mol: float | None
    source: Source
    note: str = ""

    def __post_init__(self) -> None:
        if self.name not in COMPONENTS:
            raise ValueError(f"unknown uncertainty component '{self.name}'")
        if self.source == "qualitative":
            if self.value_kcal_mol is not None:
                raise ValueError("a qualitative term must not carry a number")
        else:
            v = self.value_kcal_mol
            if v is None or not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{self.name}: value must be a finite number >= 0")


@dataclass(frozen=True)
class UncertaintyBudget:
    terms: tuple[Term, ...]

    def __post_init__(self) -> None:
        names = [t.name for t in self.terms]
        if len(set(names)) != len(names):
            raise ValueError("duplicate uncertainty component")

    @property
    def sigma_total(self) -> float:
        return math.sqrt(
            sum(t.value_kcal_mol**2 for t in self.terms if t.value_kcal_mol is not None)
        )

    @property
    def qualitative(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.terms if t.source == "qualitative")

    @property
    def missing(self) -> tuple[str, ...]:
        present = {t.name for t in self.terms}
        return tuple(c for c in COMPONENTS if c not in present)

    @property
    def confident(self) -> bool:
        return not self.qualitative and not self.missing

    def breakdown(self) -> list[tuple[str, float | None, str, str]]:
        """Rows (component, value, source, note) in spec order; missing rows say so."""
        by_name = {t.name: t for t in self.terms}
        rows: list[tuple[str, float | None, str, str]] = []
        for c in COMPONENTS:
            t = by_name.get(c)
            rows.append(
                (c, None, "missing", "") if t is None else (c, t.value_kcal_mol, t.source, t.note)
            )
        return rows
