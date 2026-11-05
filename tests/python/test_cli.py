from pathlib import Path

import pytest

from saddle.cli import main

CAMPAIGN = Path(__file__).resolve().parents[2] / "campaigns" / "fake" / "h_transfer.yaml"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli") / "run"
    assert main(["run", str(CAMPAIGN), "--out", str(out)]) == 0
    return out


def test_run_prints_verdicts(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (run_dir / "campaign.json").exists()
    assert (run_dir / "to_c" / "ts.h5").exists()


def test_report_markdown(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", str(run_dir), "--format", "markdown"]) == 0
    text = capsys.readouterr().out
    assert "# Saddle report: fake-h-transfer" in text
    assert "| to_c | **SUPPORTED** |" in text
    assert "### Gates" in text and "G8 robustness" in text
    assert "### Uncertainty" in text and "| **total** |" in text
    assert "variant: precision" in text
    assert "Eckart" in text


def test_report_to_file(run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    assert main(["report", str(run_dir), "--out", str(out)]) == 0
    assert out.read_text().startswith("# Saddle report")


def test_noise_and_hessian_commands(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ts = str(run_dir / "to_c" / "ts.h5")
    assert main(["noise", ts, "--modes", "lowest:2"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 3
    assert main(["hessian", ts, "--step", "auto"]) == 0
    assert "frequencies" in capsys.readouterr().out


def test_rerun_without_overwrite_is_refused(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", str(CAMPAIGN), "--out", str(run_dir)]) == 2
    assert "exists" in capsys.readouterr().err


def test_gpu_flags_rejected_for_fake_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", str(CAMPAIGN), "--out", str(tmp_path), "--math", "native-fp64"]) == 2
    assert "cuEST models only" in capsys.readouterr().err


def test_bench_explains_phase_f(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "benchmarks/hessian_suite.yaml"]) == 2
    assert "Phase F" in capsys.readouterr().err
