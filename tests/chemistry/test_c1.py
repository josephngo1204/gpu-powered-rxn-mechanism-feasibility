"""C1: Cl- + CH3Cl gas-phase identity SN2 must show the double well.

Reference: data/references.yaml (CCSD(T)/aug-cc-pVQZ):
complex -10.42, TS +2.52 kcal/mol relative to separated reactants (electronic).
Pass criteria: the double well is reproduced, with errors within 2 kcal/mol.

Set SADDLE_C1_RUN to a finished `saddle run campaigns/c1/c1_sn2.yaml` output
directory to check it without rerunning (the campaign takes hours on one MIG slice).
"""

import json
import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("cupy")

from saddle.workflows.pipeline import run_campaign

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.ladder]

ROOT = Path(__file__).resolve().parents[2]
REF = yaml.safe_load((ROOT / "data" / "references.yaml").read_text())["c1_cl_ch3cl_gas"]
TOL = 2.0


@pytest.fixture(scope="module")
def sn2(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    run = os.environ.get("SADDLE_C1_RUN")
    if run:
        doc = json.loads((Path(run) / "campaign.json").read_text())
    else:
        doc = run_campaign(
            ROOT / "campaigns" / "c1" / "c1_sn2.yaml", tmp_path_factory.mktemp("c1")
        ).summary
    return doc["pathways"]["sn2"]  # type: ignore[no-any-return]


def test_ts_is_validated(sn2: dict[str, object]) -> None:
    failed = [(g["gate"], g["reason"]) for g in sn2["gates"] if not g["passed"]]  # type: ignore[index, union-attr]
    assert not failed, (failed, sn2["notes"])


def test_double_well(sn2: dict[str, object]) -> None:
    wells = sn2["endpoint_minima_kcal_mol"]
    complex_ = min(wells.values())  # type: ignore[union-attr]
    ts = sn2["electronic_barrier_kcal_mol"]
    assert complex_ < -5.0
    assert ts > complex_ + 5.0


def test_energies_match_ccsdt_within_tolerance(sn2: dict[str, object]) -> None:
    complex_ = min(sn2["endpoint_minima_kcal_mol"].values())  # type: ignore[union-attr]
    assert complex_ == pytest.approx(REF["complex_kcal_mol"], abs=TOL)
    assert sn2["electronic_barrier_kcal_mol"] == pytest.approx(REF["ts_kcal_mol"], abs=TOL)


def test_identity_reaction_is_symmetric(sn2: dict[str, object]) -> None:
    wells = list(sn2["endpoint_minima_kcal_mol"].values())  # type: ignore[union-attr]
    assert len(wells) == 2 and abs(wells[0] - wells[1]) < 0.1
