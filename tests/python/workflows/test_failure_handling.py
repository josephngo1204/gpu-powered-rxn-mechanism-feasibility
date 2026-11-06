"""Model failures after the TS search become gate failures, not crashed campaigns."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from saddle.errors import PCMNotConverged, SCFError, StabilityNotConverged
from saddle.gates import gates as G
from saddle.gates.verdict import Verdict
from saddle.workflows import pipeline
from saddle.workflows.models import CuestProvider
from saddle.workflows.pipeline import run_campaign

CAMPAIGN = Path(__file__).resolve().parents[3] / "campaigns" / "fake" / "h_transfer.yaml"


def _gate(result: Any, pathway: str, gate: str) -> dict[str, Any]:
    gates = result.summary["pathways"][pathway]["gates"]
    return next(g for g in gates if g["gate"] == gate)  # type: ignore[no-any-return]


def _raise_scf(*args: Any, **kwargs: Any) -> Any:
    raise SCFError("SCF did not converge in 100 iterations")


def test_hessian_failure_makes_pathway_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pipeline._hessian

    def failing_ts_hessian(*args: Any) -> Any:
        if "/species" not in args[-1]:
            _raise_scf()
        return original(*args)

    monkeypatch.setattr(pipeline, "_hessian", failing_ts_hessian)
    result = run_campaign(CAMPAIGN, tmp_path)
    for name in ("to_c", "to_d"):
        assert result.verdicts[name] is Verdict.INCONCLUSIVE
        assert not _gate(result, name, "G3")["passed"]
        notes = result.summary["pathways"][name]["notes"]
        assert any("TS characterization failed" in n for n in notes)


def test_irc_failure_fails_g6_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_stage_irc", _raise_scf)
    result = run_campaign(CAMPAIGN, tmp_path)
    for name in ("to_c", "to_d"):
        g6 = _gate(result, name, "G6")
        assert not g6["passed"] and "missing evidence" in g6["reason"]
        assert _gate(result, name, "G4")["passed"]
        assert result.verdicts[name] is Verdict.INCONCLUSIVE
        notes = result.summary["pathways"][name]["notes"]
        assert any("IRC failed" in n for n in notes)


def test_tunneling_failure_keeps_wigner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_minimum_at_endpoint", _raise_scf)
    result = run_campaign(CAMPAIGN, tmp_path)
    tun = result.summary["pathways"]["to_c"]["tunneling"]
    assert tun["wigner"] > 1.0
    assert "eckart" not in tun and "SCF did not converge" in tun["eckart_note"]
    assert result.verdicts["to_c"] is Verdict.SUPPORTED


def test_robustness_variant_failure_fails_g8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pipeline._run_variant

    def failing_precision(c: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "precision":
            raise PCMNotConverged(1e-3, 50)
        return original(c, name, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_run_variant", failing_precision)
    result = run_campaign(CAMPAIGN, tmp_path)
    for name in ("to_c", "to_d"):
        g8 = _gate(result, name, "G8")
        assert not g8["passed"] and "precision" in g8["reason"]
        variant = result.summary["pathways"][name]["robustness"]["precision"]
        assert variant["verdict"] == "FAILED"
        assert "PCM charges did not converge" in variant["reason"]
        assert variant["barrier_kcal_mol"] is None


def test_stencil_store_is_closed_when_the_baseline_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    original_close = pipeline.StencilStore.close

    def recording_close(self: Any) -> None:
        closed.append(True)
        original_close(self)

    monkeypatch.setattr(pipeline.StencilStore, "close", recording_close)
    monkeypatch.setattr(pipeline, "_run_variant", _raise_scf)
    with pytest.raises(SCFError):
        run_campaign(CAMPAIGN, tmp_path)
    assert closed == [True]


def _provider(run: Any) -> tuple[CuestProvider, Any]:
    provider = object.__new__(CuestProvider)
    provider.level = SimpleNamespace(solvent=None)  # type: ignore[assignment]
    return provider, SimpleNamespace(run=run)


def test_g2_reports_converged_stable_scf() -> None:
    result = SimpleNamespace(
        converged=True, orbitals_stable=True, stability_eigenvalues=np.array([0.12])
    )
    provider, model = _provider(lambda *a, **k: result)
    g2 = provider.g2_evidence(model, np.zeros(6))
    assert g2["scf_converged"] is True and g2["orbitals_stable"] is True
    assert G.g2_electronic(G.Evidence(**_g2_fields(g2))).passed


def test_g2_fails_when_scf_does_not_converge_at_the_ts() -> None:
    provider, model = _provider(_raise_scf)
    g2 = provider.g2_evidence(model, np.zeros(6))
    assert g2["scf_converged"] is False
    gate = G.g2_electronic(G.Evidence(**_g2_fields(g2)))
    assert not gate.passed and "SCF not converged" in gate.reason


def test_g2_fails_as_missing_evidence_when_stability_solver_fails() -> None:
    def unstable_solver(*args: Any, **kwargs: Any) -> Any:
        raise StabilityNotConverged("stability analysis did not converge")

    provider, model = _provider(unstable_solver)
    g2 = provider.g2_evidence(model, np.zeros(6))
    assert g2["scf_converged"] is True and g2["orbitals_stable"] is None
    gate = G.g2_electronic(G.Evidence(**_g2_fields(g2)))
    assert not gate.passed and "missing evidence" in gate.reason


def _g2_fields(g2: dict[str, Any]) -> dict[str, Any]:
    keys = ("scf_converged", "orbitals_stable", "uses_pcm", "pcm_converged")
    return {k: g2[k] for k in keys}
