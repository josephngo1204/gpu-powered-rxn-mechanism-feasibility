"""Typed views of ``campaign.json``, parsed once at the dashboard's input boundary.

Result files are untrusted input: any field may be missing, null or of the
wrong type. Everything is validated here, so chart and page code works with
plain typed values (``float | None``) and never re-checks the raw document.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def number(x: Any) -> float | None:
    """``x`` as a float when it is a finite number (booleans excluded), else None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def _text(x: Any) -> str:
    return "" if x is None else str(x)


def _mapping(x: Any) -> Mapping[str, Any]:
    return x if isinstance(x, Mapping) else {}


def _sequence(x: Any) -> Sequence[Any]:
    return x if isinstance(x, (list, tuple)) else ()


@dataclass(frozen=True)
class Gate:
    code: str  # "G1" .. "G8"
    name: str
    passed: bool
    reason: str
    failure_action: str


@dataclass(frozen=True)
class Term:
    """One uncertainty component (kcal/mol)."""

    component: str
    value: float | None
    source: str  # measured | estimate | qualitative | missing
    note: str


@dataclass(frozen=True)
class Variant:
    """The barrier under one setting change (FD step or robustness variant)."""

    label: str
    barrier: float
    verdict: str


@dataclass(frozen=True)
class Pathway:
    name: str
    verdict: str
    reason: str
    barrier: float | None  # free-energy barrier, kcal/mol
    sigma: float | None
    limit: float | None
    electronic_barrier: float | None
    reactant_minimum: float | None  # IRC endpoint minima vs separated reactants
    product_minimum: float | None
    imaginary_cm: float | None
    noise_epsilon: float | None
    noise_floor_cm: float | None
    n_gradients: float | None
    wigner: float | None
    eckart: float | None
    gates: tuple[Gate, ...]
    uncertainty: tuple[Term, ...]
    sensitivity: tuple[Variant, ...]
    frequencies_cm: tuple[float, ...]
    notes: tuple[str, ...]

    @property
    def two_sigma(self) -> float | None:
        return None if self.sigma is None else 2.0 * self.sigma

    @classmethod
    def from_json(cls, name: str, doc: Mapping[str, Any], limit: float | None) -> Pathway:
        minima = _mapping(doc.get("endpoint_minima_kcal_mol"))
        tunneling = _mapping(doc.get("tunneling"))
        gates = tuple(
            Gate(
                _text(g.get("gate")),
                _text(g.get("name")),
                g.get("passed") is True,
                _text(g.get("reason")),
                _text(g.get("failure_action")),
            )
            for g in map(_mapping, _sequence(doc.get("gates")))
        )
        terms = tuple(
            Term(
                _text(u.get("component")),
                number(u.get("value")),
                _text(u.get("source")),
                _text(u.get("note")),
            )
            for u in map(_mapping, _sequence(doc.get("uncertainty")))
        )
        sensitivity = [
            Variant(f"FD step {step} bohr", value, "")
            for step, raw in _mapping(doc.get("barrier_by_step")).items()
            if (value := number(raw)) is not None
        ] + [
            Variant(f"variant: {k}", value, _text(_mapping(v).get("verdict")))
            for k, v in _mapping(doc.get("robustness")).items()
            if (value := number(_mapping(v).get("barrier_kcal_mol"))) is not None
        ]
        return cls(
            name=name,
            verdict=_text(doc.get("verdict")) or "UNKNOWN",
            reason=_text(doc.get("reason")),
            barrier=number(doc.get("barrier_kcal_mol")),
            sigma=number(doc.get("sigma_kcal_mol")),
            limit=number(doc.get("limit_kcal_mol")) or limit,
            electronic_barrier=number(doc.get("electronic_barrier_kcal_mol")),
            reactant_minimum=number(minima.get("reactant")),
            product_minimum=number(minima.get("product")),
            imaginary_cm=number(doc.get("imaginary_frequency_cm")),
            noise_epsilon=number(doc.get("noise_epsilon_eh_bohr")),
            noise_floor_cm=number(doc.get("noise_floor_cm")),
            n_gradients=number(doc.get("n_gradients")),
            wigner=number(tunneling.get("wigner")),
            eckart=number(tunneling.get("eckart")),
            gates=gates,
            uncertainty=terms,
            sensitivity=tuple(sensitivity),
            frequencies_cm=tuple(
                f for x in _sequence(doc.get("frequencies_cm")) if (f := number(x)) is not None
            ),
            notes=tuple(_text(n) for n in _sequence(doc.get("notes"))),
        )


@dataclass(frozen=True)
class Reference:
    """Literature values for one campaign, from ``data/references.yaml``."""

    citation: str
    complex_energy: float | None  # kcal/mol vs separated reactants
    ts_energy: float | None
    barrier: float | None
    pathway: str | None  # which pathway the values describe, when not the only one

    @classmethod
    def from_yaml(cls, doc: Mapping[str, Any]) -> Reference:
        return cls(
            citation=_text(doc.get("citation")).strip(),
            complex_energy=number(doc.get("complex_kcal_mol")),
            ts_energy=number(doc.get("ts_kcal_mol")),
            barrier=number(doc.get("barrier_kcal_mol")),
            pathway=_text(doc.get("pathway")) or None,
        )


@dataclass(frozen=True)
class Ranking:
    first: str
    second: str
    resolved: bool
    lower: str
    reason: str


@dataclass(frozen=True)
class Campaign:
    label: str
    anchor: str  # unique HTML id
    hypothesis: str
    model: str  # one-line level of theory
    temperature_k: float | None
    target_half_life_s: float | None
    limit: float | None
    commit: str
    gpu: str
    cuest: str
    pathways: tuple[Pathway, ...]
    rankings: tuple[Ranking, ...]
    reference: Reference | None

    def reference_pathway(self) -> Pathway | None:
        """The pathway the reference describes: the named one, else the only one."""
        if self.reference is None:
            return None
        if self.reference.pathway is not None:
            return next((p for p in self.pathways if p.name == self.reference.pathway), None)
        return self.pathways[0] if len(self.pathways) == 1 else None

    @classmethod
    def from_json(
        cls, label: str, anchor: str, doc: Mapping[str, Any], references: Mapping[str, Any]
    ) -> Campaign:
        env = _mapping(doc.get("environment"))
        model = _mapping(doc.get("model"))
        limit = number(doc.get("limit_kcal_mol"))
        model_text = " · ".join(
            _text(model[k])
            for k in ("functional", "basis", "grid", "solvent", "math_mode")
            if model.get(k)
        ) or _text(model.get("type"))
        rankings = []
        for r in map(_mapping, _sequence(doc.get("rankings"))):
            pair = [_text(x) for x in _sequence(r.get("pair"))] + ["", ""]
            rankings.append(
                Ranking(
                    pair[0],
                    pair[1],
                    r.get("resolved") is True,
                    _text(r.get("lower")),
                    _text(r.get("reason")),
                )
            )
        hypothesis = _text(doc.get("hypothesis"))
        ref = reference_for(hypothesis, references)
        return cls(
            label=label,
            anchor=anchor,
            hypothesis=hypothesis,
            model=model_text,
            temperature_k=number(doc.get("temperature_K")),
            target_half_life_s=number(doc.get("target_half_life_s")),
            limit=limit,
            commit=_text(env.get("git_commit"))[:10] or "unknown",
            gpu=_text(env.get("gpu_model")) or "CPU",
            cuest=_text(env.get("cuest_version")) or "—",
            pathways=tuple(
                Pathway.from_json(str(n), _mapping(p), limit)
                for n, p in _mapping(doc.get("pathways")).items()
            ),
            rankings=tuple(rankings),
            reference=None if ref is None else Reference.from_yaml(ref),
        )


def _key(s: str) -> str:
    return s.replace("-", "_").lower()


def reference_for(hypothesis: str, references: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The references.yaml entry keyed by the hypothesis id (``-`` and ``_`` equivalent)."""
    for key, entry in references.items():
        if _key(str(key)) == _key(hypothesis) and isinstance(entry, Mapping):
            return entry
    return None


def unique_anchors(labels: Sequence[str]) -> list[str]:
    """HTML ids from labels: lowercase, ``[a-z0-9-]`` only, de-duplicated."""
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        base = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "run"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return out
