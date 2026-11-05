"""saddle command-line interface (spec §10 "Commands")."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from saddle import _core
from saddle.analysis.hessian_quality import assess_hessians
from saddle.store.results import read_result
from saddle.workflows.models import Model, build_model
from saddle.workflows.pipeline import (
    FOURTH_DERIVATIVE_ESTIMATE,
    MAX_STEP,
    MIN_STEP,
    NOISE_RESOLUTION_FLOOR,
    CampaignError,
    load_campaign,
    run_campaign,
)
from saddle.workflows.report import render_markdown


class CliError(Exception):
    pass


def _fake_only_checks(model_type: str, args: argparse.Namespace) -> None:
    if not model_type.startswith("fake:"):
        return
    for flag, value, default in (("--math", args.math, None), ("--gpus", args.gpus, None)):
        if value != default:
            raise CliError(f"{flag} applies to cuEST models only; {model_type} has no such setting")
    if getattr(args, "profile", "reference") != "reference":
        raise CliError("profiles select cuEST levels of theory; fake models have one level")


@dataclass(frozen=True)
class _StoredTS:
    summary: dict[str, Any]
    model_id: str
    model: Model
    coords: npt.NDArray[np.float64]
    hessian: npt.NDArray[np.float64]
    masses: npt.NDArray[np.float64]


def _load_stored(path: str) -> _StoredTS:
    stored = read_result(path)
    campaign = load_campaign(stored.summary["campaign"])
    return _StoredTS(
        summary=stored.summary,
        model_id=stored.provenance.model,
        model=build_model(campaign.model_spec),
        coords=np.asarray(stored.arrays["coords"], dtype=np.float64),
        hessian=np.asarray(stored.arrays["hessian_raw"], dtype=np.float64),
        masses=campaign.masses,
    )


def cmd_run(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.campaign)
    _fake_only_checks(str(campaign.model_spec["type"]), args)
    result = run_campaign(args.campaign, args.out, overwrite=args.overwrite)
    for name, verdict in result.verdicts.items():
        p = result.summary["pathways"][name]
        barrier = p["barrier_kcal_mol"]
        shown = "—" if barrier is None else f"{barrier:.2f} kcal/mol"
        print(f"{name}: {verdict.value} ({shown}; {p['reason']})")
    print(f"results: {Path(args.out) / 'campaign.json'}")
    return 0


def cmd_noise(args: argparse.Namespace) -> int:
    ts = _load_stored(args.ts)
    _fake_only_checks(ts.model_id, args)
    kind, _, count = args.modes.partition(":")
    if kind != "lowest" or not count.isdigit():
        raise CliError("--modes must look like lowest:N")
    vib = _core.analyze_vibrations(0.5 * (ts.hessian + ts.hessian.T), ts.coords, ts.masses)
    modes = np.asarray(vib.modes_cartesian)
    print("mode  frequency_cm  epsilon_Eh_bohr")
    for k in range(min(int(count), modes.shape[1])):
        eps, _ = _core.probe_noise(ts.model, ts.coords, modes[:, k])
        print(f"{k:4d}  {vib.frequencies_cm[k]:12.1f}  {eps:.2e}")
    return 0


def cmd_hessian(args: argparse.Namespace) -> int:
    ts = _load_stored(args.ts)
    _fake_only_checks(ts.model_id, args)
    if args.step == "auto":
        eps, _ = _core.probe_noise(ts.model, ts.coords, np.ones_like(ts.coords))
        h_opt = _core.optimal_step(max(eps, NOISE_RESOLUTION_FLOOR), FOURTH_DERIVATIVE_ESTIMATE)
        step = float(np.clip(h_opt, MIN_STEP, MAX_STEP))
    else:
        step = float(args.step)
    raws: dict[float, npt.NDArray[np.float64]] = {}
    for h in (step, 2 * step):
        points = _core.central_stencil(ts.coords, h)
        raw, _, _ = _core.assemble_central([ts.model.evaluate(c)[1] for _, _, c in points], h)
        raws[h] = np.asarray(raw)
    quality = assess_hessians(
        raws, ts.coords, ts.masses, noise_floor_cm=ts.summary.get("noise_floor_cm")
    )
    vib = _core.analyze_vibrations(0.5 * (raws[step] + raws[step].T), ts.coords, ts.masses)
    print(f"step {step:.3e} bohr")
    print("frequencies (cm^-1):", " ".join(f"{f:.1f}" for f in vib.frequencies_cm))
    print(quality.summary())
    return 0 if quality.passed else 1


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.run) / "campaign.json"
    if not path.exists():
        raise CliError(f"no campaign.json in {args.run}")
    campaign = json.loads(path.read_text())
    if args.format == "json":
        text = json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    else:
        text = render_markdown(campaign)
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    raise CliError("benchmarks measure cuEST on a GPU and are not available until Phase F")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saddle", description="Mechanism feasibility engine")
    sub = parser.add_subparsers(dest="command", required=True)

    def gpu_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--math", default=None, help="cuEST math mode (cuEST models only)")
        p.add_argument("--gpus", default=None, help="comma-separated GPU ids (cuEST models only)")

    run = sub.add_parser("run", help="run a campaign")
    run.add_argument("campaign")
    run.add_argument("--out", required=True)
    run.add_argument("--profile", default="reference", choices=["search", "reference"])
    run.add_argument("--overwrite", action="store_true", help="replace existing results")
    gpu_flags(run)
    run.set_defaults(func=cmd_run)

    noise = sub.add_parser("noise", help="probe gradient noise along TS modes")
    noise.add_argument("ts")
    noise.add_argument("--modes", default="lowest:6")
    gpu_flags(noise)
    noise.set_defaults(func=cmd_noise)

    hess = sub.add_parser("hessian", help="recompute a TS Hessian with quality checks")
    hess.add_argument("ts")
    hess.add_argument("--step", default="auto")
    gpu_flags(hess)
    hess.set_defaults(func=cmd_hessian)

    report = sub.add_parser("report", help="render a campaign report")
    report.add_argument("run")
    report.add_argument("--format", default="markdown", choices=["markdown", "json"])
    report.add_argument("--out", default=None)
    report.set_defaults(func=cmd_report)

    bench = sub.add_parser("bench", help="cuEST performance benchmarks (Phase F)")
    bench.add_argument("suite")
    bench.add_argument("--warm-jit", action="store_true")
    bench.add_argument("--out", default=None)
    bench.set_defaults(func=cmd_bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (CliError, CampaignError) as exc:
        print(f"saddle: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
