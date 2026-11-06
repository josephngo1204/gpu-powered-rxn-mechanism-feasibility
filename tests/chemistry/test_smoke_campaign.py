"""End-to-end real-DFT campaign: HCN -> HNC (full pipeline on cuEST)."""

from pathlib import Path

import pytest

pytest.importorskip("cupy")

from saddle.gates.verdict import Verdict
from saddle.workflows.pipeline import run_campaign

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

CAMPAIGN = Path(__file__).resolve().parents[2] / "campaigns" / "smoke" / "hcn_isomerization.yaml"


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    return run_campaign(CAMPAIGN, tmp_path_factory.mktemp("smoke"))


def test_every_gate_passes_on_real_evidence(result) -> None:  # type: ignore[no-untyped-def]
    s = result.summary["pathways"]["isomerization"]
    failed = [(g["gate"], g["reason"]) for g in s["gates"] if not g["passed"]]
    assert not failed, (failed, s["notes"])
    assert any("internal RHF->RHF stability" in n for n in s["notes"])


def test_rotational_residual_is_reported_and_rotation_variant_ran(result) -> None:  # type: ignore[no-untyped-def]
    s = result.summary["pathways"]["isomerization"]
    assert "rotation" in s["robustness"]
    grid = next(u for u in s["uncertainty"] if u["component"] == "grid")
    assert grid["source"] == "measured"


def test_bridged_ts_and_high_barrier(result) -> None:  # type: ignore[no-untyped-def]
    s = result.summary["pathways"]["isomerization"]
    assert s["imaginary_frequency_cm"] < -800
    assert 35.0 < s["electronic_barrier_kcal_mol"] < 60.0
    assert 30.0 < s["barrier_kcal_mol"] < 60.0
    assert s["tunneling"]["eckart"] > s["tunneling"]["wigner"] > 1.0


def test_verdict_is_infeasible(result) -> None:  # type: ignore[no-untyped-def]
    assert result.verdicts["isomerization"] is Verdict.INFEASIBLE


def test_provenance_records_the_level_of_theory(result) -> None:  # type: ignore[no-untyped-def]
    m = result.summary["model"]
    assert m["functional"] == "WB97X-V" and m["basis"] == "def2-SVP"
    assert m["grid"] == "(99,590)" and m["math_mode"] == "native-fp64"
    assert m["cuest_version"] == "0.2.2"


def test_robustness_variants_agree(result) -> None:  # type: ignore[no-untyped-def]
    rob = result.summary["pathways"]["isomerization"]["robustness"]
    assert set(rob) == {"grid", "precision", "functional", "rotation"}
    for name, v in rob.items():
        assert v["verdict"] == "INFEASIBLE", (name, v)


def test_robustness_variant_barriers_agree(result) -> None:  # type: ignore[no-untyped-def]
    """Variant barriers must stay close to baseline (the 216 kcal/mol wander is fixed)."""
    s = result.summary["pathways"]["isomerization"]
    for name, v in s["robustness"].items():
        assert abs(v["barrier_kcal_mol"] - s["barrier_kcal_mol"]) < 3.0, (name, v)
