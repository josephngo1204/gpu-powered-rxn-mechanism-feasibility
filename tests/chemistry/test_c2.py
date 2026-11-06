"""C2: Cl- + CH3Cl SN2 in PCM water must show the solvent-driven change.

Expected solvent-driven profile change: the gas-phase double
well (C1: complex about -11 kcal/mol) becomes close to unimodal, and the barrier
rises toward the estimated 26 kcal/mol (data/references.yaml).

Tolerances are fixed before the run:
- "close to unimodal": any ion-dipole well is shallower than 3 kcal/mol;
- "higher barrier": electronic barrier at least 10 kcal/mol, against +2.5 in the gas phase;
- dG++ within 4 kcal/mol of 26. That is twice the 2 kcal/mol MAD target,
  because the reference is an estimate rather than a benchmark.

Set SADDLE_C2_RUN to a finished `saddle run campaigns/c2/c2_sn2_water.yaml`
output directory to check it without rerunning.
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
REF = yaml.safe_load((ROOT / "data" / "references.yaml").read_text())["c2_cl_ch3cl_water"]
BARRIER_TOL = 4.0
MAX_WELL_DEPTH = 3.0
MIN_ELECTRONIC_BARRIER = 10.0


@pytest.fixture(scope="module")
def sn2(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    run = os.environ.get("SADDLE_C2_RUN")
    if run:
        doc = json.loads((Path(run) / "campaign.json").read_text())
    else:
        doc = run_campaign(
            ROOT / "campaigns" / "c2" / "c2_sn2_water.yaml", tmp_path_factory.mktemp("c2")
        ).summary
    return doc["pathways"]["sn2"]  # type: ignore[no-any-return]


def test_ts_is_validated(sn2: dict[str, object]) -> None:
    failed = [(g["gate"], g["reason"]) for g in sn2["gates"] if not g["passed"]]  # type: ignore[index, union-attr]
    assert not failed, (failed, sn2["notes"])


def test_solvent_flattens_the_double_well(sn2: dict[str, object]) -> None:
    wells = sn2["endpoint_minima_kcal_mol"]
    assert wells, "IRC endpoints must be optimized"
    assert min(wells.values()) > -MAX_WELL_DEPTH  # type: ignore[union-attr]


def test_solvent_raises_the_barrier(sn2: dict[str, object]) -> None:
    assert sn2["electronic_barrier_kcal_mol"] > MIN_ELECTRONIC_BARRIER  # type: ignore[operator]
    assert sn2["barrier_kcal_mol"] == pytest.approx(REF["barrier_kcal_mol"], abs=BARRIER_TOL)
