"""HTML dashboard over finished campaigns (saddle.dashboard)."""

import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from saddle.cli import main
from saddle.dashboard import Run, charts, explain, page, render_dashboard
from saddle.dashboard.svg import nice_ticks
from saddle.dashboard.views import Campaign, Pathway, number, unique_anchors


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


# --- behaviors pinned after mutation testing ------------------------------------------


def _view(barrier: float | None, verdict: str = "SUPPORTED", limit: float = 22.52) -> Pathway:
    return Pathway.from_json("p", _pathway(barrier, verdict), limit)


def _attr(svg: str, element_class: str, attribute: str) -> float:
    match = re.search(rf'class="{element_class}"[^>]*?\b{attribute}="([-\d.]+)"', svg)
    assert match, (element_class, attribute)
    return float(match.group(1))


def test_verdict_badges_carry_their_status() -> None:
    assert "b-good" in page.verdict_badge("SUPPORTED")
    assert "b-critical" in page.verdict_badge("INFEASIBLE")
    assert "b-warning" in page.verdict_badge("INCONCLUSIVE")


def test_barrier_chart_geometry_whisker_dot_and_limit() -> None:
    svg = charts.barrier_chart([("a", _view(10.0))])  # 2 sigma = 1.0
    dot = _attr(svg, "dot s1", "cx")
    left, right = _attr(svg, "whisker", "x1"), _attr(svg, "whisker", "x2")
    limit = _attr(svg, "limit", "x1")
    assert left < dot < right and math.isclose(dot - left, right - dot, rel_tol=1e-3)
    assert limit > right  # 10 + 2 sigma is well below the 22.5 limit


def test_barrier_chart_whisker_widens_with_sigma() -> None:
    narrow = charts.barrier_chart([("a", _view(10.0))])
    doc = _pathway(10.0)
    doc["sigma_kcal_mol"] = 2.0
    wide = charts.barrier_chart([("a", Pathway.from_json("p", doc, 22.52))])
    width = lambda svg: _attr(svg, "whisker", "x2") - _attr(svg, "whisker", "x1")  # noqa: E731
    assert width(wide) > 3 * width(narrow)


def test_untrusted_verdict_is_escaped_inside_svg() -> None:
    svg = charts.barrier_chart([("a", _view(5.0, verdict="<b>x</b>"))])
    assert "<b>x</b>" not in svg and "&lt;b&gt;x&lt;/b&gt;" in svg


def test_reference_is_compared_only_with_the_pathway_it_describes() -> None:
    doc = _campaign(a=_pathway(5.0), b=_pathway(9.0))
    unnamed = {"c9_test": {"citation": "r", "ts_kcal_mol": 7.0}}
    html = render_dashboard([Run("c9", doc)], unnamed)
    assert "not compared" in html and "Against reference" not in html
    named = {"c9_test": {"citation": "r", "ts_kcal_mol": 7.0, "pathway": "b"}}
    html = render_dashboard([Run("c9", doc)], named)
    assert "Against reference (b)" in html
    assert "<td>11.00</td><td>7.00</td><td>4.00</td>" in html  # b: TS 9 + 2 vs 7


def test_half_life_chart_places_a_fast_reaction_before_its_target() -> None:
    row = charts.HalfLifeRow("a", _view(7.0), 298.15, 3600.0)
    svg = charts.half_life_chart([row])
    dot = _attr(svg, "dot s1", "cx")
    band_left = _attr(svg, "band-s1", "x")
    band_right = band_left + _attr(svg, "band-s1", "width")
    assert band_left < dot < band_right  # central estimate inside its ±2σ range
    assert dot < _attr(svg, "limit", "x1")


def test_durations_read_naturally() -> None:
    assert explain.duration(3600.0) == "1 hour"
    assert explain.duration(2.3e-6) == "2.3 microseconds"
    assert explain.duration(45.0) == "45 seconds"
    assert explain.duration(math.inf) == "practically forever"
    assert explain.duration(3.15576e7 * 2e6) == "2e+06 years"


def test_headline_states_half_life_and_meaning() -> None:
    text = explain.headline(_view(7.4), 298.15, 3600.0)
    assert "Half-life about" in text and "under 1 hour" in text and "fast enough" in text
    assert "no valid transition state" in explain.headline(
        _view(None, "NO_SADDLE_FOUND"), 298.15, 3600.0
    )


def test_huge_barrier_does_not_crash() -> None:
    assert explain.safe_half_life_s(5000.0, 298.15) == math.inf
    assert "forever" in explain.headline(_view(5000.0, "INFEASIBLE"), 298.15, 3600.0)


def test_anchors_are_unique_and_safe() -> None:
    assert unique_anchors(["c2", "c2", "C 2!"]) == ["c2", "c2-2", "c-2"]


def test_number_rejects_non_numbers() -> None:
    assert number(True) is None and number(float("nan")) is None and number("3") is None
    assert number(3) == 3.0


def test_malformed_document_renders_without_crashing() -> None:
    doc = {"pathways": {"x": {"gates": "oops", "uncertainty": [None, 3], "robustness": []}}}
    campaign = Campaign.from_json("bad", "bad", doc, {})
    assert campaign.pathways[0].verdict == "UNKNOWN"
    assert "<!DOCTYPE html>" in render_dashboard([Run("bad", doc)])


def test_cli_labels_same_named_runs_by_path(tmp_path: Path) -> None:
    for parent in ("a", "b"):
        (tmp_path / parent / "c1").mkdir(parents=True)
        (tmp_path / parent / "c1" / "campaign.json").write_text(
            json.dumps(_campaign(p=_pathway(5.0)))
        )
    out = tmp_path / "d.html"
    assert (
        main(
            [
                "dashboard",
                str(tmp_path / "a"),
                str(tmp_path / "b"),
                "--out",
                str(out),
                "--references",
                "",
            ]
        )
        == 0
    )
    html = out.read_text()
    assert f"{tmp_path / 'a' / 'c1'} / p" in html and f"{tmp_path / 'b' / 'c1'} / p" in html
