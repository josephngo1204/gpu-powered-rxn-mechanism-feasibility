"""Markdown report (spec §5, §6, §12).

Every barrier summary shows its uncertainty and any failed gates, and the
report shows how step size and the robustness variants move each barrier.
"""

from collections.abc import Mapping
from typing import Any


def _fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_markdown(campaign: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Saddle report: {campaign['hypothesis']}")
    add("")
    add(
        f"T = {campaign['temperature_K']} K, "
        f"target half-life {campaign['target_half_life_s']:g} s, "
        f"feasibility limit ΔG‡ = {campaign['limit_kcal_mol']:.2f} kcal/mol."
    )
    env = campaign.get("environment", {})
    add(
        f"Code {env.get('git_commit', 'unknown')[:12]} (dirty: {env.get('git_dirty', 'unknown')}), "
        f"GPU {env.get('gpu_model', 'unknown')}, cuEST {env.get('cuest_version', 'unknown')}."
    )
    add("")
    add("## Verdicts")
    add("")
    add("| Pathway | Verdict | ΔG‡ (kcal/mol) | 2σ | Failed gates | Reason |")
    add("|---|---|---|---|---|---|")
    for name, p in campaign["pathways"].items():
        failed = [g["gate"] for g in p["gates"] if not g["passed"]]
        sigma = p.get("sigma_kcal_mol")
        two_sigma = _fmt(None if sigma is None else 2 * sigma)
        add(
            f"| {name} | **{p['verdict']}** | {_fmt(p['barrier_kcal_mol'])} | {two_sigma} | "
            f"{', '.join(failed) or 'none'} | {p['reason']} |"
        )
    add("")
    if campaign.get("rankings"):
        add("## Rankings")
        add("")
        for r in campaign["rankings"]:
            a, b = r["pair"]
            outcome = f"{r['lower']} is lower" if r["resolved"] else "not resolved"
            add(f"- {a} vs {b}: {outcome} ({r['reason']})")
        add("")

    for name, p in campaign["pathways"].items():
        add(f"## Pathway `{name}`")
        add("")
        add(
            f"Imaginary frequency {_fmt(p['imaginary_frequency_cm'], 1)} cm⁻¹; gradient noise "
            f"{p['noise_epsilon_eh_bohr']:.1e} Eh/bohr; FD step {p['fd_step_bohr']:.2e} bohr; "
            f"noise floor {p['noise_floor_cm']:.1f} cm⁻¹; {p['n_gradients']} gradients."
        )
        add("")
        add("### Gates")
        add("")
        add("| Gate | Result | Reason |")
        add("|---|---|---|")
        for g in p["gates"]:
            mark = "pass" if g["passed"] else f"**FAIL** → {g['failure_action']}"
            add(f"| {g['gate']} {g['name']} | {mark} | {g['reason']} |")
        add("")
        add("### Uncertainty (kcal/mol)")
        add("")
        add("| Component | σ | Source | Note |")
        add("|---|---|---|---|")
        for u in p["uncertainty"]:
            value = "qualitative" if u["source"] == "qualitative" else _fmt(u["value"], 3)
            add(f"| {u['component']} | {value} | {u['source']} | {u['note']} |")
        add(f"| **total** | {_fmt(p.get('sigma_kcal_mol'), 3)} | | |")
        add("")
        add("### Sensitivity of the barrier")
        add("")
        add("| Setting | ΔG‡ (kcal/mol) | Verdict |")
        add("|---|---|---|")
        for step, value in p["barrier_by_step"].items():
            add(f"| FD step {step} bohr | {_fmt(value, 3)} | |")
        for k, v in p["robustness"].items():
            barrier = _fmt(v["barrier_kcal_mol"], 3)
            add(f"| variant: {k} | {barrier} | {v['verdict']} ({v['reason']}) |")
        add("")
        tun = p.get("tunneling") or {}
        if tun:
            add("### Tunneling (reported separately, not applied)")
            add("")
            if "wigner" in tun:
                add(f"- Wigner κ = {tun['wigner']:.3f}")
            if "eckart" in tun:
                add(
                    f"- Eckart κ = {tun['eckart']:.3f} (reverse barrier "
                    f"{tun['reverse_barrier_kcal_mol']:.2f} kcal/mol, ZPE-corrected)"
                )
            if "eckart_note" in tun:
                add(f"- {tun['eckart_note']}")
            add("")
        if p.get("irc"):
            add("### IRC")
            add("")
            for branch, info in p["irc"].items():
                kind = (
                    "reactants"
                    if info["matches_reactant"]
                    else "intended products"
                    if info["matches_product"]
                    else "other"
                )
                add(
                    f"- {branch}: {info['points']} points, stopped on {info['stop_reason']}, "
                    f"endpoint {kind} (converged: {info['endpoint_converged']})"
                )
            add("")
        if p.get("notes"):
            add("### Notes")
            add("")
            for n in p["notes"]:
                add(f"- {n}")
            add("")
    return "\n".join(lines).rstrip() + "\n"
