"""sigma_grid: the budget reports whichever of the stated and measured values it used."""

import dataclasses
from pathlib import Path

import numpy as np

from saddle.analysis.uncertainty import Term
from saddle.gates import gates as G
from saddle.workflows.pipeline import (
    PathwayAnalysis,
    VariantOutcome,
    _with_measured_grid_term,
    load_campaign,
)

CAMPAIGN = Path(__file__).resolve().parents[3] / "campaigns" / "fake" / "h_transfer.yaml"


def _analysis(barrier: float) -> PathwayAnalysis:
    campaign = load_campaign(CAMPAIGN)
    return PathwayAnalysis(
        pathway=campaign.hypothesis.pathways[0],
        evidence=G.Evidence(),
        ts=None,
        barrier_kcal_mol=barrier,
        barrier_by_step={},
        numerical_sigma=None,
        thermo_sigma=None,
        noise_epsilon=float("nan"),
        noise_floor_cm=float("nan"),
        step=float("nan"),
        tunneling={},
        irc={},
    )


def _grid_term(stated: float, shift: float) -> Term:
    campaign = load_campaign(CAMPAIGN)
    campaign = dataclasses.replace(
        campaign, uncertainty={"grid": Term("grid", stated, "estimate", "stated")}
    )
    variants = {"rotation": VariantOutcome("SUPPORTED", 7.0 + shift, "")}
    budget = _with_measured_grid_term(campaign, _analysis(7.0), variants)
    return next(t for t in budget.terms if t.name == "grid")


def test_measured_shift_above_estimate_is_used_and_labelled_measured() -> None:
    term = _grid_term(stated=0.1, shift=0.4)
    assert np.isclose(term.value_kcal_mol, 0.4)
    assert term.source == "measured"


def test_estimate_above_measured_shift_keeps_its_estimate_label() -> None:
    term = _grid_term(stated=0.1, shift=0.006)
    assert np.isclose(term.value_kcal_mol, 0.1)
    assert term.source == "estimate"
    assert "0.006" in term.note
