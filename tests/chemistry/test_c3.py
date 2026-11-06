"""C3: X- + CH3CH2X (X = F, Cl), SN2 vs anti-E2 vs syn-E2 in the gas phase.

Pass criteria: both halide preference orders and anti-over-syn E2 are reproduced.
CCSD(T) TS energies (data/references.yaml c3_ccsdt_ts, kcal/mol from separated reactants):
  F-:  anti-E2 -1.27 < SN2 2.20 < syn-E2 5.68
  Cl-: SN2 5.81 < anti-E2 18.18 < syn-E2 30.92
The order matters as much as the error size; MAD <= 2 kcal/mol
on the high-quality reference subset.

Set SADDLE_C3_F_RUN / SADDLE_C3_CL_RUN to finished run directories to check
them without rerunning (each campaign takes hours on one MIG slice).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("cupy")

from saddle.workflows.pipeline import run_campaign

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.ladder]

ROOT = Path(__file__).resolve().parents[2]
REF = yaml.safe_load((ROOT / "data" / "references.yaml").read_text())["c3_ccsdt_ts"]
PATHWAYS = ("sn2", "anti_e2", "syn_e2")
MAD_LIMIT = 2.0
CASES = {"f": ("F_minus", "SADDLE_C3_F_RUN"), "cl": ("Cl_minus", "SADDLE_C3_CL_RUN")}


def _pathways(halide: str, tmp: pytest.TempPathFactory) -> dict[str, dict[str, object]]:
    run = os.environ.get(CASES[halide][1])
    if run:
        doc = json.loads((Path(run) / "campaign.json").read_text())
    else:
        doc = run_campaign(
            ROOT / "campaigns" / "c3" / f"c3_{halide}.yaml", tmp.mktemp(f"c3_{halide}")
        ).summary
    return doc["pathways"]  # type: ignore[no-any-return]


@pytest.fixture(scope="module", params=list(CASES))
def case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, dict[str, dict[str, object]]]:
    return request.param, _pathways(request.param, tmp_path_factory)


def _energies(pathways: dict[str, dict[str, object]]) -> dict[str, float]:
    return {p: float(pathways[p]["electronic_barrier_kcal_mol"]) for p in PATHWAYS}  # type: ignore[arg-type]


def test_every_ts_is_validated(case: tuple[str, dict[str, dict[str, object]]]) -> None:
    _, pathways = case
    for p in PATHWAYS:
        failed = [(g["gate"], g["reason"]) for g in pathways[p]["gates"] if not g["passed"]]  # type: ignore[index, union-attr]
        assert not failed, (p, failed, pathways[p]["notes"])


def test_preference_order_matches_ccsdt(case: tuple[str, dict[str, dict[str, object]]]) -> None:
    halide, pathways = case
    e = _energies(pathways)
    ref = REF[CASES[halide][0]]
    assert sorted(PATHWAYS, key=e.__getitem__) == sorted(PATHWAYS, key=ref.__getitem__), (e, ref)


def test_anti_e2_below_syn_e2(case: tuple[str, dict[str, dict[str, object]]]) -> None:
    _, pathways = case
    e = _energies(pathways)
    assert e["anti_e2"] < e["syn_e2"]


def test_mean_absolute_deviation_within_spec(tmp_path_factory: pytest.TempPathFactory) -> None:
    dev = []
    for halide, (key, _) in CASES.items():
        e = _energies(_pathways(halide, tmp_path_factory))
        dev += [e[p] - REF[key][p] for p in PATHWAYS]
    assert float(np.mean(np.abs(dev))) <= MAD_LIMIT, dev
