import json
import shutil
from pathlib import Path

import pytest

from saddle.gates.verdict import Verdict
from saddle.store.results import read_result
from saddle.workflows.pipeline import CampaignError, load_campaign, run_campaign

CAMPAIGN = Path(__file__).resolve().parents[3] / "campaigns" / "fake" / "h_transfer.yaml"


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    return run_campaign(CAMPAIGN, tmp_path_factory.mktemp("run"))


def test_campaign_loads() -> None:
    c = load_campaign(CAMPAIGN)
    assert c.symbols == ("C", "H", "C", "C")
    assert set(c.ts_guesses) == {"to_c", "to_d"}


def test_every_gate_passes_for_both_saddles(result) -> None:  # type: ignore[no-untyped-def]
    for name in ("to_c", "to_d"):
        gates = result.summary["pathways"][name]["gates"]
        failed = [g for g in gates if not g["passed"]]
        assert not failed, (name, failed)


def test_barriers_follow_donor_acceptor_distance(result) -> None:  # type: ignore[no-untyped-def]
    c = result.summary["pathways"]["to_c"]["barrier_kcal_mol"]
    d = result.summary["pathways"]["to_d"]["barrier_kcal_mol"]
    assert 5.0 < c < d


def test_verdicts(result) -> None:  # type: ignore[no-untyped-def]
    assert result.verdicts["to_c"] is Verdict.SUPPORTED
    assert result.verdicts["to_d"] in (Verdict.INFEASIBLE, Verdict.OUTCOMPETED)
    ranking = result.summary["rankings"][0]
    assert ranking["resolved"] and ranking["lower"] == "to_c"


def test_report_ingredients_present(result) -> None:  # type: ignore[no-untyped-def]
    s = result.summary["pathways"]["to_c"]
    assert s["imaginary_frequency_cm"] < 0
    assert len(s["barrier_by_step"]) == 2
    assert set(s["robustness"]) == {"grid", "precision", "functional"}
    assert {u["component"] for u in s["uncertainty"]} == {
        "numerical",
        "grid",
        "basis",
        "method",
        "thermo",
        "solvent",
    }
    assert s["tunneling"]["wigner"] > 1.0
    assert s["tunneling"]["eckart"] > 1.0
    assert s["irc"]["forward"]["endpoint_converged"]


def test_results_written_with_provenance(result) -> None:  # type: ignore[no-untyped-def]
    stored = read_result(result.out_dir / "to_c" / "ts.h5")
    assert stored.provenance.model.startswith("fake:pair_model:")
    assert stored.summary["verdict"] == "SUPPORTED"
    doc = json.loads((result.out_dir / "campaign.json").read_text())
    assert doc["pathways"]["to_d"]["pathway"] == "to_d"


def test_rerun_refuses_to_overwrite_and_resumes_from_cache(result) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(CampaignError, match="exists"):
        run_campaign(CAMPAIGN, result.out_dir)
    again = run_campaign(CAMPAIGN, result.out_dir, overwrite=True)
    assert again.verdicts == result.verdicts


def test_noisy_precision_variant_breaks_robustness(tmp_path: Path) -> None:
    """At 1e-7 Eh/bohr gradient noise the variant's Hessian fails G3, so G8 fails."""
    work = tmp_path / "fake"
    shutil.copytree(CAMPAIGN.parent, work)
    harsh = work / CAMPAIGN.name
    harsh.write_text(harsh.read_text().replace("gradient_noise: 1.0e-9", "gradient_noise: 1.0e-7"))
    r = run_campaign(harsh, tmp_path / "run")
    s = r.summary["pathways"]["to_c"]
    assert s["robustness"]["precision"]["reason"] == "failed gates: G3"
    g8 = next(g for g in s["gates"] if g["gate"] == "G8")
    assert not g8["passed"] and "precision" in g8["reason"]
    assert r.verdicts["to_c"] is Verdict.INCONCLUSIVE


def test_pcm_failure_during_search_is_reported_as_g2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §11/§12: a PCM convergence failure is a G2 rejection, not a silent drop."""
    from saddle.errors import PCMNotConverged
    from saddle.workflows import pipeline

    real = pipeline._refined

    def fail_saddle_search(search: object, *args: object, **kwargs: object) -> object:
        if search is pipeline._core.saddle_search:
            raise PCMNotConverged(2e-2, 1)
        return real(search, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "_refined", fail_saddle_search)
    r = run_campaign(CAMPAIGN, tmp_path / "run")
    s = r.summary["pathways"]["to_c"]
    g2 = next(g for g in s["gates"] if g["gate"] == "G2")
    assert not g2["passed"] and "PCM not converged" in g2["reason"]
    assert any("PCM charges did not converge" in n for n in s["notes"])
    assert r.verdicts["to_c"] is not Verdict.SUPPORTED
