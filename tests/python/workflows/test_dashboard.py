"""HTML dashboard over finished campaigns (saddle.workflows.dashboard)."""

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from saddle.cli import main
from saddle.workflows.dashboard import Run, nice_ticks, render_dashboard


def _pathway(barrier: float | None, verdict: str = "SUPPORTED") -> dict[str, Any]:
    return {
        "verdict": verdict,
        "reason": "test",
        "barrier_kcal_mol": barrier,
        "sigma_kcal_mol": 0.5 if barrier is not None else None,
        "electronic_barrier_kcal_mol": None if barrier is None else barrier + 2.0,
        "endpoint_minima_kcal_mol": {"reactant": -1.0, "product": 0.5},
        "imaginary_frequency_cm": -900.0,
        "noise_epsilon_eh_bohr": 1e-9,
        "noise_floor_cm": 3.0,
        "n_gradients": 10,
        "fd_step_bohr": 1e-3,
        "gates": [
            {"gate": "G1", "name": "stationarity", "passed": True, "reason": "ok"},
            {
                "gate": "G4",
                "name": "saddle order",
                "passed": False,
                "reason": "two modes",
                "failure_action": "re-search",
            },
        ],
        "uncertainty": [
            {"component": "method", "value": 0.4, "source": "estimate", "note": ""},
            {"component": "solvent", "value": None, "source": "qualitative", "note": "n/a"},
        ],
        "barrier_by_step": {"0.001": barrier, "0.002": barrier},
        "robustness": {"grid": {"barrier_kcal_mol": barrier, "verdict": verdict, "reason": ""}},
        "tunneling": {"wigner": 1.1},
        "frequencies_cm": [-900.0, 100.0],
        "notes": ["a note"],
    }


def _campaign(hypothesis: str = "c9-test", **pathways: dict[str, Any]) -> dict[str, Any]:
    return {
        "hypothesis": hypothesis,
        "temperature_K": 298.15,
        "target_half_life_s": 3600.0,
        "limit_kcal_mol": 22.52,
        "environment": {"git_commit": "abc123"},
        "model": {"type": "fake:pair_model"},
        "pathways": pathways,
        "rankings": [{"pair": ["a", "b"], "lower": "a", "resolved": True, "reason": "gap"}],
    }


def test_nice_ticks_cover_range_with_round_steps() -> None:
    ticks = nice_ticks(-11.7, 2.5, 5)
    assert ticks[0] <= -11.7 and ticks[-1] >= 2.5
    steps = {round(b - a, 9) for a, b in itertools.pairwise(ticks)}
    assert steps == {5.0}


def test_nice_ticks_degenerate_range() -> None:
    assert len(nice_ticks(3.0, 3.0)) >= 2


def test_dashboard_shows_every_pathway_verdict_and_gate_failure() -> None:
    doc = _campaign(a=_pathway(9.8), b=_pathway(28.4, "INFEASIBLE"))
    html = render_dashboard([Run("fake", doc)])
    assert html.startswith("<!DOCTYPE html>")
    for text in ("fake / a", "fake / b", "SUPPORTED", "INFEASIBLE", "a is lower", "re-search"):
        assert text in html
    assert html.count("<svg") >= 5  # overview + profile + per-pathway charts


def test_dashboard_survives_pathway_without_barrier() -> None:
    doc = _campaign(failed=_pathway(None, "NO_SADDLE_FOUND"))
    html = render_dashboard([Run("x", doc)])
    assert "NO_SADDLE_FOUND" in html
    assert "No pathway produced a barrier" in html


def test_dashboard_escapes_untrusted_text() -> None:
    doc = _campaign(hypothesis="<script>alert(1)</script>", ok=_pathway(5.0))
    html = render_dashboard([Run("<b>run</b>", doc)])
    assert "<script>alert(1)" not in html
    assert "&lt;b&gt;run&lt;/b&gt;" in html


def test_reference_overlay_matches_hypothesis_id() -> None:
    doc = _campaign(hypothesis="c9-test", sn2=_pathway(7.0))
    refs = {"c9_test": {"citation": "ref", "complex_kcal_mol": -1.2, "ts_kcal_mol": 9.5}}
    html = render_dashboard([Run("c9", doc)], refs)
    assert "Against reference" in html
    assert 'class="ref"' in html
    assert "-0.50" in html  # TS 9.0 computed vs 9.5 reference


def test_cli_dashboard_discovers_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("r1", "r2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "campaign.json").write_text(json.dumps(_campaign(p=_pathway(5.0))))
    out = tmp_path / "dash.html"
    assert main(["dashboard", str(tmp_path), "--out", str(out), "--references", ""]) == 0
    html = out.read_text()
    assert "r1 / p" in html and "r2 / p" in html
    assert "2 campaigns" in capsys.readouterr().out


def test_cli_dashboard_rejects_empty_dir(tmp_path: Path) -> None:
    assert main(["dashboard", str(tmp_path), "--out", str(tmp_path / "d.html")]) == 2
