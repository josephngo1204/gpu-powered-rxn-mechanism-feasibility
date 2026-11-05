"""Gaussian basis sets from the Basis Set Exchange, normalized centrally (spec §8).

cuEST needs normalized contraction coefficients, one family of shells (all
pure or all Cartesian), and angular momentum contiguous from L = 0. General
contractions and multi-L (sp) shells are split into segmented shells.

Normalization: each primitive is scaled by
N(a, L) = sqrt(2^L (2a)^(L+3/2) / (pi^(3/2) (2L-1)!!)), then the contraction
is rescaled to unit self-overlap.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import basis_set_exchange as bse  # type: ignore[import-untyped]

from saddle import elements


class BasisError(ValueError):
    pass


@dataclass(frozen=True)
class Shell:
    angular_momentum: int
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]  # as published, unnormalized
    pure: bool

    def __post_init__(self) -> None:
        if self.angular_momentum < 0:
            raise BasisError("negative angular momentum")
        if len(self.exponents) != len(self.coefficients) or not self.exponents:
            raise BasisError("exponents and coefficients must have the same nonzero length")
        if any(not (a > 0.0 and math.isfinite(a)) for a in self.exponents):
            raise BasisError(f"exponents must be positive and finite: {self.exponents}")
        if any(not math.isfinite(c) for c in self.coefficients):
            raise BasisError("coefficients must be finite")

    @property
    def n_functions(self) -> int:
        L = self.angular_momentum
        return 2 * L + 1 if self.pure else (L + 1) * (L + 2) // 2

    def normalized_coefficients(self) -> tuple[float, ...]:
        L = self.angular_momentum
        dfact = math.prod(range(1, 2 * L, 2)) if L else 1
        prim = [
            math.sqrt(2.0**L * (2.0 * a) ** (L + 1.5) / (math.pi**1.5 * dfact)) * c
            for a, c in zip(self.exponents, self.coefficients, strict=True)
        ]
        q = 0.0
        for ai, ci in zip(self.exponents, self.coefficients, strict=True):
            for aj, cj in zip(self.exponents, self.coefficients, strict=True):
                q += (math.sqrt(4.0 * ai * aj) / (ai + aj)) ** (L + 1.5) * ci * cj
        if not q > 0.0:
            raise BasisError("contraction has non-positive self-overlap")
        scale = q**-0.5
        return tuple(p * scale for p in prim)


@dataclass(frozen=True)
class BasisSet:
    name: str
    shells: Mapping[str, tuple[Shell, ...]]
    source: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        all_shells = [s for group in self.shells.values() for s in group]
        if not all_shells:
            raise BasisError(f"basis {self.name} has no shells")
        if len({s.pure for s in all_shells}) > 1:
            raise BasisError(f"basis {self.name} mixes pure and Cartesian shells")
        for el, group in self.shells.items():
            ls = sorted({s.angular_momentum for s in group})
            if ls != list(range(len(ls))):
                raise BasisError(
                    f"{self.name}/{el}: angular momentum must be contiguous from 0, got {ls}"
                )

    @property
    def pure(self) -> bool:
        return next(iter(self.shells.values()))[0].pure

    def n_functions(self, symbols: Sequence[str]) -> int:
        try:
            return sum(s.n_functions for sym in symbols for s in self.shells[sym])
        except KeyError as exc:
            raise BasisError(f"basis {self.name} has no shells for {exc.args[0]}") from exc


def load_basis(name: str, symbols: Sequence[str], *, pure: bool = True) -> BasisSet:
    """Load `name` for the distinct elements in `symbols` from the Basis Set Exchange."""
    try:
        distinct = sorted(set(symbols), key=elements.atomic_number)
    except KeyError as exc:
        raise BasisError(f"unknown element {exc.args[0]}") from exc
    try:
        data = bse.get_basis(name, elements=[elements.atomic_number(s) for s in distinct])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise BasisError(f"cannot load basis {name!r} for {distinct}: {exc}") from exc
    shells: dict[str, tuple[Shell, ...]] = {}
    for sym in distinct:
        el = data["elements"].get(str(elements.atomic_number(sym)))
        if el is None or "electron_shells" not in el:
            raise BasisError(f"basis {name!r} has no shells for {sym}")
        out: list[Shell] = []
        for sh in el["electron_shells"]:
            exps = tuple(float(x) for x in sh["exponents"])
            ams = sh["angular_momentum"]
            columns = sh["coefficients"]
            # One L per column for sp-type shells; otherwise one L, many columns.
            ls = ams if len(ams) == len(columns) else ams * len(columns)
            for L, col in zip(ls, columns, strict=True):
                coefs = tuple(float(c) for c in col)
                keep = [(a, c) for a, c in zip(exps, coefs, strict=True) if c != 0.0]
                out.append(
                    Shell(int(L), tuple(a for a, _ in keep), tuple(c for _, c in keep), pure)
                )
        shells[sym] = tuple(out)
    return BasisSet(
        name=name,
        shells=shells,
        source=f"Basis Set Exchange {bse.version()}",
        metadata={
            "bse_name": str(data.get("name", name)),
            "revision": str(data.get("revision_description", "")),
        },
    )
