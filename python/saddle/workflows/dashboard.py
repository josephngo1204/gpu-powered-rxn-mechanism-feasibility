"""Self-contained HTML results dashboard over one or more finished campaigns.

Reads only ``campaign.json`` documents (the same input as ``saddle report``) and
writes one HTML file with inline SVG charts: no JavaScript libraries, no network
access and no plotting dependency. It shows, for every pathway:

- the free-energy barrier against the feasibility limit, with its 2σ interval;
- the electronic energy profile (separated reactants, IRC endpoint minima, TS),
  with reference values overlaid where ``data/references.yaml`` has them;
- the uncertainty budget, the robustness variants, the gates, and key numbers.

Colors follow a validated categorical palette (slots 1-3 checked all-pairs in
light and dark); every series is also direct-labelled and every chart has a
table view, so color never carries meaning alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

SERIES = 3  # validated categorical slots; more pathways fold onto the table only
GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
VERDICT_STATUS = {  # saddle.gates.verdict.Verdict values -> (status, icon)
    "SUPPORTED": ("good", "✓"),
    "OUTCOMPETED": ("warning", "↓"),
    "INFEASIBLE": ("critical", "×"),
    "NO_SADDLE_FOUND": ("critical", "×"),
    "BARRIERLESS_CANDIDATE": ("warning", "?"),
    "INCONCLUSIVE": ("warning", "?"),
}


@dataclass(frozen=True)
class Run:
    """One finished campaign: a display label and its ``campaign.json`` document."""

    label: str
    doc: Mapping[str, Any]


def _e(x: Any) -> str:
    return escape(str(x), quote=True)


def _num(x: Any, digits: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return _e(x)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _f(x: Any) -> float | None:
    """``x`` as a float when it is a finite number, else None."""
    return float(x) if _finite(x) else None


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round tick values covering [lo, hi] with about ``target`` intervals."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks: list[float] = []
    t = start
    while t <= hi + 1e-9 * step:
        ticks.append(round(t, 10))
        t += step
    if ticks[-1] < hi:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def _norm_key(s: str) -> str:
    return s.replace("-", "_").lower()


def _reference_for(doc: Mapping[str, Any], refs: Mapping[str, Any]) -> Mapping[str, Any]:
    """References keyed by the hypothesis id (dashes and underscores equivalent)."""
    by_key = {_norm_key(k): v for k, v in refs.items() if isinstance(v, Mapping)}
    found = by_key.get(_norm_key(str(doc.get("hypothesis", ""))))
    return found if isinstance(found, Mapping) else {}


# --------------------------------------------------------------------------- SVG


class _Svg:
    def __init__(self, width: int, height: int, label: str) -> None:
        self.w, self.h = width, height
        self.parts: list[str] = []
        self.label = label

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x: float, y: float, s: str, cls: str = "lbl", anchor: str = "start") -> None:
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{_e(s)}</text>'
        )

    def render(self) -> str:
        return (
            f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{_e(self.label)}" '
            f'preserveAspectRatio="xMinYMin meet">' + "".join(self.parts) + "</svg>"
        )


def _x_axis(svg: _Svg, ticks: list[float], sx: Any, y0: float, y1: float, unit: str) -> None:
    for t in ticks:
        x = sx(t)
        svg.add(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{y0:.1f}" y2="{y1:.1f}"/>')
        svg.text(x, y1 + 16, f"{t:g}", "tick", "middle")
    svg.text(sx(ticks[-1]), y1 + 32, unit, "tick", "end")


def barrier_chart(rows: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    """Dot + 2σ whisker per pathway, with each campaign's feasibility limit as a tick."""
    usable = [(n, p) for n, p in rows if _finite(p.get("barrier_kcal_mol"))]
    if not usable:
        return '<p class="muted">No pathway produced a barrier.</p>'
    left, right, row_h, top = 200, 130, 34, 12
    width = 760
    height = top + row_h * len(usable) + 60
    vals: list[float] = [0.0]
    for _, p in usable:
        b = float(p["barrier_kcal_mol"])
        s = float(p.get("sigma_kcal_mol") or 0.0)
        vals += [b - 2 * s, b + 2 * s]
        if _finite(p.get("limit_kcal_mol")):
            vals.append(float(p["limit_kcal_mol"]))
    ticks = nice_ticks(min(vals), max(vals))
    lo, hi = ticks[0], ticks[-1]

    def sx(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (width - left - right)

    svg = _Svg(width, height, "Free-energy barrier per pathway with 2 sigma interval and limit")
    y_end = top + row_h * len(usable)
    _x_axis(svg, ticks, sx, top - 4, y_end, "ΔG‡ (kcal/mol)")
    for i, (name, p) in enumerate(usable):
        y = top + row_h * i + row_h / 2
        b = float(p["barrier_kcal_mol"])
        s = float(p.get("sigma_kcal_mol") or 0.0)
        verdict = str(p.get("verdict", ""))
        status, icon = VERDICT_STATUS.get(verdict, ("warning", "?"))
        tip = f"{name}: ΔG‡ {b:.2f} ± {2 * s:.2f} (2σ) kcal/mol; {verdict}; {p.get('reason', '')}"
        svg.add(f'<g class="hit" data-tip="{_e(tip)}"><title>{_e(tip)}</title>')
        svg.add(
            f'<rect x="0" y="{y - row_h / 2:.1f}" width="{width}" height="{row_h}" class="row"/>'
        )
        svg.text(left - 10, y + 4, name, "lbl", "end")
        if _finite(p.get("limit_kcal_mol")):
            lx = sx(float(p["limit_kcal_mol"]))
            svg.add(
                f'<line class="limit" x1="{lx:.1f}" x2="{lx:.1f}" y1="{y - 12:.1f}" y2="{y + 12:.1f}"/>'
            )
        svg.add(
            f'<line class="whisker" x1="{sx(b - 2 * s):.1f}" x2="{sx(b + 2 * s):.1f}" '
            f'y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        svg.add(f'<circle class="dot s1" cx="{sx(b):.1f}" cy="{y:.1f}" r="5"/>')
        svg.add(f'<text x="{width - right + 12}" y="{y + 4:.1f}" class="lbl">')
        svg.add(f'<tspan class="st-{status}">{icon}</tspan> {_e(verdict)} {b:.1f}</text>')
        svg.add("</g>")
    svg.text(left, height - 2, "| = feasibility limit for the campaign's T and half-life", "tick")
    return svg.render()


def _profile_points(p: Mapping[str, Any]) -> list[tuple[str, float | None]]:
    minima = p.get("endpoint_minima_kcal_mol") or {}
    return [
        ("Separated reactants", 0.0),
        ("Reactant-side minimum", minima.get("reactant")),
        ("TS", p.get("electronic_barrier_kcal_mol")),
        ("Product-side minimum", minima.get("product")),
    ]


def profile_chart(
    pathways: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Any] | None = None
) -> str:
    """Electronic energy profile, one 2px line per pathway (first three slots)."""
    names = list(pathways)[:SERIES]
    cats = [c for c, _ in _profile_points({})]
    series = {n: _profile_points(pathways[n]) for n in names}
    ref_pts: dict[int, float] = {}
    if reference:
        if _finite(reference.get("complex_kcal_mol")):
            ref_pts[1] = float(reference["complex_kcal_mol"])
        if _finite(reference.get("ts_kcal_mol")):
            ref_pts[2] = float(reference["ts_kcal_mol"])
    clean = {
        n: [(i, f) for i, (_, v) in enumerate(pts) if (f := _f(v)) is not None]
        for n, pts in series.items()
    }
    vals = [v for pts in clean.values() for _, v in pts] + list(ref_pts.values())
    if len(vals) < 2:
        return '<p class="muted">No energy profile available.</p>'
    ticks = nice_ticks(min(vals), max(vals), 5)
    lo, hi = ticks[0], ticks[-1]
    width, height, left, right, top, bottom = 760, 320, 60, 150, 36, 40

    def sx(i: int) -> float:
        return left + i * (width - left - right) / (len(cats) - 1)

    def sy(v: float) -> float:
        return top + (hi - v) / (hi - lo) * (height - top - bottom)

    svg = _Svg(width, height, "Electronic energy profile relative to separated reactants")
    for t in ticks:
        svg.add(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
        )
        svg.text(left - 8, sy(t) + 4, f"{t:g}", "tick", "end")
    svg.add(
        f'<line class="base" x1="{left}" x2="{width - right}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>'
    )
    for i, c in enumerate(cats):
        svg.text(sx(i), height - bottom + 18, c, "tick", "middle")
    svg.text(left - 50, 12, "ΔE (kcal/mol)", "tick")
    label_y: list[float] = []
    for k, n in enumerate(names):
        pts = clean[n]
        d = " ".join(
            f"{'M' if j == 0 else 'L'}{sx(i):.1f},{sy(float(v)):.1f}"
            for j, (i, v) in enumerate(pts)
        )
        svg.add(f'<path class="line s{k + 1}" d="{d}"/>')
        for i, v in pts:
            tip = f"{n} · {cats[i]}: {float(v):.2f} kcal/mol"
            svg.add(
                f'<g class="hit" data-tip="{_e(tip)}"><title>{_e(tip)}</title>'
                f'<circle class="hitbox" cx="{sx(i):.1f}" cy="{sy(float(v)):.1f}" r="12"/>'
                f'<circle class="dot s{k + 1}" cx="{sx(i):.1f}" cy="{sy(float(v)):.1f}" r="4.5"/></g>'
            )
        i_last, v_last = pts[-1]
        y = sy(float(v_last)) + 4
        while any(abs(y - u) < 14 for u in label_y):
            y += 14
        label_y.append(y)
        svg.add(f'<circle class="dot s{k + 1}" cx="{sx(i_last) + 18:.1f}" cy="{y - 4:.1f}" r="4"/>')
        svg.text(sx(i_last) + 28, y, f"{n} ({float(v_last):.1f})")
    for i, v in ref_pts.items():
        tip = f"Reference · {cats[i]}: {v:.2f} kcal/mol"
        svg.add(
            f'<g class="hit" data-tip="{_e(tip)}"><title>{_e(tip)}</title>'
            f'<circle class="hitbox" cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="12"/>'
            f'<rect class="ref" x="{sx(i) - 5:.1f}" y="{sy(v) - 5:.1f}" width="10" height="10"/></g>'
        )
    return svg.render()


def hbar_chart(items: Sequence[tuple[str, Any, str]], unit: str, label: str) -> str:
    """Horizontal bars from a zero baseline; one series, value at the tip."""
    items = [(n, f, note) for n, v, note in items if (f := _f(v)) is not None]
    if not items:
        return '<p class="muted">Nothing to plot.</p>'
    left, right, row_h, top, width = 110, 60, 28, 8, 560
    height = top + row_h * len(items) + 40
    ticks = nice_ticks(0.0, max(max(v for _, v, _ in items), 1e-9), 5)
    hi = ticks[-1]

    def sx(v: float) -> float:
        return left + v / hi * (width - left - right)

    svg = _Svg(width, height, label)
    _x_axis(svg, ticks, sx, top - 4, top + row_h * len(items), unit)
    for i, (n, v, note) in enumerate(items):
        y = top + row_h * i + row_h / 2
        w = max(sx(v) - left, 0.0)
        tip = f"{n}: {v:.3g} {unit}" + (f" ({note})" if note else "")
        svg.add(f'<g class="hit" data-tip="{_e(tip)}"><title>{_e(tip)}</title>')
        svg.add(
            f'<rect x="0" y="{y - row_h / 2:.1f}" width="{width}" height="{row_h}" class="row"/>'
        )
        svg.text(left - 10, y + 4, n, "lbl", "end")
        r = min(4.0, w / 2)
        svg.add(
            f'<path class="bar s1" d="M{left},{y - 8:.1f} h{max(w - r, 0):.2f} '
            f"a{r:.2f},{r:.2f} 0 0 1 {r:.2f},{r:.2f} v{16 - 2 * r:.2f} "
            f'a{r:.2f},{r:.2f} 0 0 1 -{r:.2f},{r:.2f} h-{max(w - r, 0):.2f} z"/>'
        )
        svg.text(left + w + 6, y + 4, f"{v:.3g}", "val")
        svg.add("</g>")
    return svg.render()


def robustness_chart(p: Mapping[str, Any]) -> str:
    """Barrier under each FD step and robustness variant, against the baseline."""
    rows: list[tuple[str, float, str]] = []
    for step, v in (p.get("barrier_by_step") or {}).items():
        if (fv := _f(v)) is not None:
            rows.append((f"FD step {step} bohr", fv, ""))
    for k, v in (p.get("robustness") or {}).items():
        if (fv := _f(v.get("barrier_kcal_mol"))) is not None:
            rows.append((f"variant: {k}", fv, str(v.get("verdict", ""))))
    base = _f(p.get("barrier_kcal_mol"))
    if not rows or base is None:
        return '<p class="muted">No variants were run.</p>'
    s = float(p.get("sigma_kcal_mol") or 0.0)
    vals = [float(v) for _, v, _ in rows] + [float(base) - 2 * s, float(base) + 2 * s]
    ticks = nice_ticks(min(vals), max(vals), 5)
    lo, hi = ticks[0], ticks[-1]
    left, right, row_h, top, width = 160, 60, 26, 8, 560
    height = top + row_h * len(rows) + 58

    def sx(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (width - left - right)

    svg = _Svg(width, height, "Barrier sensitivity to step size and robustness variants")
    y_end = top + row_h * len(rows)
    _x_axis(svg, ticks, sx, top - 4, y_end, "ΔG‡ (kcal/mol)")
    bx = sx(float(base))
    svg.add(
        f'<rect class="band" x="{sx(float(base) - 2 * s):.1f}" y="{top - 4}" '
        f'width="{sx(float(base) + 2 * s) - sx(float(base) - 2 * s):.1f}" height="{y_end - top + 4}"/>'
    )
    svg.add(f'<line class="limit" x1="{bx:.1f}" x2="{bx:.1f}" y1="{top - 4}" y2="{y_end}"/>')
    for i, (n, v, verdict) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        tip = f"{n}: {float(v):.3f} kcal/mol (Δ {float(v) - float(base):+.3f}) {verdict}"
        svg.add(f'<g class="hit" data-tip="{_e(tip)}"><title>{_e(tip)}</title>')
        svg.add(
            f'<rect x="0" y="{y - row_h / 2:.1f}" width="{width}" height="{row_h}" class="row"/>'
        )
        svg.text(left - 10, y + 4, n, "lbl", "end")
        svg.add(f'<circle class="dot s1" cx="{sx(float(v)):.1f}" cy="{y:.1f}" r="5"/>')
        shift = float(v) - float(base)
        svg.text(
            width - right + 10, y + 4, "0.00" if abs(shift) < 0.005 else f"{shift:+.2f}", "val"
        )
        svg.add("</g>")
    svg.text(left, height - 2, "| baseline · shaded ±2σ · right: shift", "tick")
    return svg.render()


# --------------------------------------------------------------------------- HTML


def _verdict_badge(verdict: str) -> str:
    status, icon = VERDICT_STATUS.get(verdict, ("warning", "?"))
    return f'<span class="badge b-{status}"><span aria-hidden="true">{icon}</span> {_e(verdict)}</span>'


def _gate_table(rows: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    head = "".join(f"<th>{g}</th>" for g in GATES)
    body: list[str] = []
    for name, p in rows:
        by_gate = {g.get("gate"): g for g in p.get("gates", [])}
        cells: list[str] = []
        for g in GATES:
            info = by_gate.get(g)
            if info is None:
                cells.append('<td class="muted" title="not run">·</td>')
                continue
            ok = bool(info.get("passed"))
            reason = f"{info.get('name', '')}: {info.get('reason', '')}"
            if not ok and info.get("failure_action"):
                reason += f" → {info['failure_action']}"
            mark = '<span class="st-good">✓</span>' if ok else '<span class="st-critical">×</span>'
            cells.append(
                f'<td title="{_e(reason)}" data-tip="{_e(reason)}" class="hit">{mark}</td>'
            )
        body.append(f'<tr><th scope="row">{_e(name)}</th>{"".join(cells)}</tr>')
    return (
        f'<table class="gates"><thead><tr><th>Pathway</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _pathway_section(run: Run, name: str, p: Mapping[str, Any]) -> str:
    unc = [
        (str(u.get("component")), u.get("value"), f"{u.get('source')}: {u.get('note', '')}")
        for u in p.get("uncertainty", [])
        if u.get("source") != "qualitative"
    ]
    unc_rows = "".join(
        f"<tr><td>{_e(u.get('component'))}</td><td class='n'>"
        f"{'qualitative' if u.get('source') == 'qualitative' else _num(u.get('value'), 3)}</td>"
        f"<td>{_e(u.get('source'))}</td><td>{_e(u.get('note', ''))}</td></tr>"
        for u in p.get("uncertainty", [])
    )
    tun = p.get("tunneling") or {}
    facts = [
        ("Imaginary frequency", f"{_num(p.get('imaginary_frequency_cm'), 1)} cm⁻¹"),
        ("Electronic barrier", f"{_num(p.get('electronic_barrier_kcal_mol'))} kcal/mol"),
        (
            "ΔG‡ ± 2σ",
            f"{_num(p.get('barrier_kcal_mol'))} ± "
            f"{_num(2 * p['sigma_kcal_mol'] if _finite(p.get('sigma_kcal_mol')) else None)} kcal/mol",
        ),
        (
            "Gradient noise",
            f"{p['noise_epsilon_eh_bohr']:.1e} Eh/bohr"
            if _finite(p.get("noise_epsilon_eh_bohr"))
            else "—",
        ),
        ("Noise floor", f"{_num(p.get('noise_floor_cm'), 1)} cm⁻¹"),
        ("Gradients evaluated", _num(p.get("n_gradients"), 0)),
        ("Wigner / Eckart κ", f"{_num(tun.get('wigner'), 3)} / {_num(tun.get('eckart'), 3)}"),
    ]
    facts_html = "".join(f"<div><dt>{_e(k)}</dt><dd>{v}</dd></div>" for k, v in facts)
    notes = "".join(f"<li>{_e(n)}</li>" for n in p.get("notes", []))
    freqs = ", ".join(f"{f:.1f}" for f in p.get("frequencies_cm", []) if _finite(f))
    return f"""
<section class="pathway">
  <h3>{_e(run.label)} / <code>{_e(name)}</code> {_verdict_badge(str(p.get("verdict", "")))}</h3>
  <p class="muted">{_e(p.get("reason", ""))}</p>
  <dl class="facts">{facts_html}</dl>
  <div class="grid2">
    <figure><figcaption>Uncertainty budget (σ, kcal/mol) — total {_num(p.get("sigma_kcal_mol"), 2)}</figcaption>
      {hbar_chart(unc, "kcal/mol", "Uncertainty budget components")}
      <details><summary>Table</summary><table><thead><tr><th>Component</th><th>σ</th><th>Source</th><th>Note</th></tr></thead><tbody>{unc_rows}</tbody></table></details>
    </figure>
    <figure><figcaption>Barrier sensitivity (step size and robustness variants)</figcaption>
      {robustness_chart(p)}
    </figure>
  </div>
  <details><summary>Harmonic frequencies (cm⁻¹)</summary><p class="mono">{freqs or "—"}</p></details>
  {f"<details><summary>Notes</summary><ul>{notes}</ul></details>" if notes else ""}
</section>"""


def _campaign_section(run: Run, refs: Mapping[str, Any]) -> str:
    doc = run.doc
    pathways: Mapping[str, Mapping[str, Any]] = doc.get("pathways", {})
    ref = _reference_for(doc, refs)
    env = doc.get("environment", {})
    model = doc.get("model", {})
    model_bits = " · ".join(
        str(model[k])
        for k in ("functional", "basis", "grid", "solvent", "math_mode")
        if model.get(k)
    ) or str(model.get("type", ""))
    rank = "".join(
        f"<li>{_e(r['pair'][0])} vs {_e(r['pair'][1])}: "
        f"{(_e(r['lower']) + ' is lower') if r.get('resolved') else 'not resolved'} "
        f"<span class='muted'>({_e(r.get('reason', ''))})</span></li>"
        for r in doc.get("rankings", [])
    )
    ref_rows = ""
    if ref:
        ref_cells = []
        if _finite(ref.get("complex_kcal_mol")):
            got = [p.get("endpoint_minima_kcal_mol", {}).get("reactant") for p in pathways.values()]
            ref_cells.append(("Complex ΔE", got[0] if got else None, ref["complex_kcal_mol"]))
        if _finite(ref.get("ts_kcal_mol")):
            got = [p.get("electronic_barrier_kcal_mol") for p in pathways.values()]
            ref_cells.append(("TS ΔE", got[0] if got else None, ref["ts_kcal_mol"]))
        if _finite(ref.get("barrier_kcal_mol")):
            got = [p.get("barrier_kcal_mol") for p in pathways.values()]
            ref_cells.append(("ΔG‡", got[0] if got else None, ref["barrier_kcal_mol"]))
        trs = "".join(
            f"<tr><td>{_e(k)}</td><td class='n'>{_num(c)}</td><td class='n'>{_num(r)}</td>"
            f"<td class='n'>{_num(c - r) if _finite(c) else '—'}</td></tr>"
            for k, c, r in ref_cells
        )
        ref_rows = (
            "<h4>Against reference</h4><table><thead><tr><th>Quantity</th><th>Computed</th>"
            f"<th>Reference</th><th>Δ</th></tr></thead><tbody>{trs}</tbody></table>"
            f"<p class='muted small'>{_e(str(ref.get('citation', '')).strip())}</p>"
        )
    prof_rows = "".join(
        f"<tr><td>{_e(n)}</td>"
        + "".join(f"<td class='n'>{_num(v)}</td>" for _, v in _profile_points(p))
        + "</tr>"
        for n, p in pathways.items()
    )
    prof_head = "".join(f"<th>{_e(c)}</th>" for c, _ in _profile_points({}))
    legend = "".join(
        f'<span class="key"><svg width="22" height="10"><line class="line s{i + 1}" x1="0" x2="22" y1="5" y2="5"/></svg>{_e(n)}</span>'
        for i, n in enumerate(list(pathways)[:SERIES])
    )
    if ref:
        legend += '<span class="key"><svg width="12" height="12"><rect class="ref" x="1" y="1" width="10" height="10"/></svg>reference</span>'
    folded = (
        f"<p class='muted small'>{len(pathways) - SERIES} more pathways are in the table only.</p>"
        if len(pathways) > SERIES
        else ""
    )
    sections = "".join(_pathway_section(run, n, p) for n, p in pathways.items())
    return f"""
<section class="campaign" id="{_e(run.label)}">
  <h2>{_e(run.label)} <span class="muted">— {_e(doc.get("hypothesis", ""))}</span></h2>
  <p class="muted">{_e(model_bits)} · T = {_num(doc.get("temperature_K"), 2)} K · half-life
     {_num(doc.get("target_half_life_s"), 0)} s · limit {_num(doc.get("limit_kcal_mol"))} kcal/mol ·
     code {_e(str(env.get("git_commit", "unknown"))[:10])} · {_e(env.get("gpu_model", "CPU / fake"))}
     · cuEST {_e(env.get("cuest_version", "—"))}</p>
  {f'<ul class="rank">{rank}</ul>' if rank else ""}
  <figure><figcaption>Electronic energy profile relative to separated reactants (kcal/mol)</figcaption>
    <div class="legend">{legend}</div>
    {profile_chart(pathways, ref)}
    {folded}
    <details><summary>Table</summary><table><thead><tr><th>Pathway</th>{prof_head}</tr></thead><tbody>{prof_rows}</tbody></table></details>
  </figure>
  {ref_rows}
  {sections}
</section>"""


_CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;
--text-secondary:#52514e;--muted:#898781;--gridline:#e1e0d9;--baseline:#c3c2b7;
--border:rgba(11,11,11,.10);--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;
--good:#0ca30c;--warning:#fab219;--critical:#d03b3b;--ref:#0b0b0b;--wash:rgba(42,120,214,.10)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;--gridline:#2c2c2a;
--baseline:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;
--ref:#f0efec;--wash:rgba(57,135,229,.14)}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--gridline:#2c2c2a;--baseline:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--ref:#f0efec;--wash:rgba(57,135,229,.14)}
body{margin:0}
.viz-root{background:var(--page);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"DejaVu Sans",sans-serif;
min-height:100vh;padding:24px}
main{max-width:1100px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 4px}h3{font-size:15px;margin:20px 0 4px}
h4{font-size:14px;margin:16px 0 4px}
.muted{color:var(--text-secondary)}.small{font-size:12px}.mono,code{font-family:ui-monospace,monospace;font-size:12px}
section.campaign,section.overview{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;
padding:16px 20px;margin:16px 0}
section.pathway{border-top:1px solid var(--gridline);margin-top:16px}
figure{margin:12px 0}figcaption{font-weight:600;margin-bottom:6px}
svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--gridline);stroke-width:1}.base{stroke:var(--baseline);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.lbl{fill:var(--text-primary);font-size:12px}.val{fill:var(--text-secondary);font-size:11px;font-variant-numeric:tabular-nums}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.s1{stroke:var(--series-1)}.line.s2{stroke:var(--series-2)}.line.s3{stroke:var(--series-3)}
.dot{stroke:var(--surface-1);stroke-width:2}.dot.s1{fill:var(--series-1)}.dot.s2{fill:var(--series-2)}.dot.s3{fill:var(--series-3)}
.bar.s1{fill:var(--series-1)}
.whisker{stroke:var(--series-1);stroke-width:2;stroke-linecap:round;opacity:.55}
.limit{stroke:var(--text-primary);stroke-width:2}
.band{fill:var(--wash)}
.ref{fill:none;stroke:var(--ref);stroke-width:2}
.row{fill:transparent}.hit:hover .row{fill:var(--wash)}.hitbox{fill:transparent}
.st-good{fill:var(--good);color:var(--good);font-weight:700}.st-critical{fill:var(--critical);color:var(--critical);font-weight:700}
.st-warning{fill:var(--text-primary);color:var(--text-primary);font-weight:700}
.badge{display:inline-block;font-size:12px;font-weight:600;border-radius:999px;padding:1px 10px;margin-left:6px;
border:1px solid currentColor}
.b-good{color:var(--good)}.b-critical{color:var(--critical)}.b-warning{color:var(--text-primary);background:var(--warning)}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:10px 16px;min-width:120px}
.tile .label{color:var(--text-secondary);font-size:12px}.tile .value{font-size:24px;font-weight:600}
dl.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:6px 16px;margin:8px 0}
dl.facts dt{color:var(--text-secondary);font-size:12px}dl.facts dd{margin:0;font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:1fr;gap:8px}
@media (min-width:1000px){.grid2{grid-template-columns:1fr 1fr}}
table{border-collapse:collapse;font-size:12px;margin:6px 0}
th,td{border-bottom:1px solid var(--gridline);padding:4px 8px;text-align:left}
td.n{text-align:right;font-variant-numeric:tabular-nums}
table.gates td{text-align:center}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--text-secondary);margin-bottom:4px}
.key{display:inline-flex;align-items:center;gap:6px}
details{margin:6px 0}summary{cursor:pointer;color:var(--text-secondary);font-size:12px}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);
border-radius:8px;padding:6px 10px;font-size:12px;max-width:380px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9}
.theme{float:right;font:inherit;font-size:12px;background:var(--surface-1);color:var(--text-primary);
border:1px solid var(--border);border-radius:6px;padding:4px 10px;cursor:pointer}
"""

_JS = """
(()=>{const t=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const h=e.target.closest('[data-tip]');
if(!h){t.style.display='none';return}t.textContent=h.dataset.tip;t.style.display='block';
const x=Math.min(e.clientX+14,innerWidth-t.offsetWidth-8);t.style.left=x+'px';t.style.top=(e.clientY+14)+'px'});
document.querySelectorAll('[data-tip] title').forEach(n=>n.remove());
document.querySelectorAll('td[data-tip]').forEach(n=>n.removeAttribute('title'));
const b=document.getElementById('theme');b.addEventListener('click',()=>{const r=document.documentElement;
const dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
r.dataset.theme=dark?'light':'dark'})})();
"""


def render_dashboard(runs: Sequence[Run], references: Mapping[str, Any] | None = None) -> str:
    """Render one HTML page over every run. ``references`` is ``data/references.yaml``."""
    refs = references or {}
    all_rows: list[tuple[str, Mapping[str, Any]]] = []
    for run in runs:
        for name, p in run.doc.get("pathways", {}).items():
            row = dict(p)
            row.setdefault("limit_kcal_mol", run.doc.get("limit_kcal_mol"))
            all_rows.append((f"{run.label} / {name}", row))
    counts: dict[str, int] = {}
    for _, p in all_rows:
        counts[str(p.get("verdict"))] = counts.get(str(p.get("verdict")), 0) + 1
    tiles = [("Campaigns", len(runs)), ("Pathways", len(all_rows)), *sorted(counts.items())]
    tiles_html = "".join(
        f'<div class="tile"><div class="label">{_e(k.capitalize() if k.isupper() else k)}</div>'
        f'<div class="value">{v}</div></div>'
        for k, v in tiles
    )
    overview_rows = "".join(
        f"<tr><td>{_e(n)}</td><td>{_verdict_badge(str(p.get('verdict', '')))}</td>"
        f"<td class='n'>{_num(p.get('barrier_kcal_mol'))}</td>"
        f"<td class='n'>{_num(2 * p['sigma_kcal_mol'] if _finite(p.get('sigma_kcal_mol')) else None)}</td>"
        f"<td class='n'>{_num(p.get('limit_kcal_mol'))}</td>"
        f"<td class='n'>{_num(p.get('electronic_barrier_kcal_mol'))}</td>"
        f"<td class='n'>{_num(p.get('imaginary_frequency_cm'), 1)}</td><td>{_e(p.get('reason', ''))}</td></tr>"
        for n, p in all_rows
    )
    campaigns = "".join(_campaign_section(r, refs) for r in runs)
    toc = " · ".join(f'<a href="#{_e(r.label)}">{_e(r.label)}</a>' for r in runs)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Saddle results dashboard</title><style>{_CSS}</style></head>
<body><div class="viz-root"><main>
<button class="theme" id="theme" type="button">Toggle theme</button>
<h1>Saddle results dashboard</h1>
<p class="muted">Mechanism feasibility verdicts with uncertainty. Campaigns: {toc or "—"}</p>
<div class="tiles">{tiles_html}</div>
<section class="overview">
<h2 style="margin-top:0">Barriers against the feasibility limit</h2>
<p class="muted small">Dot = ΔG‡, whisker = ±2σ. A pathway is SUPPORTED when ΔG‡ + 2σ is below the limit and INFEASIBLE when ΔG‡ − 2σ is above it.</p>
{barrier_chart(all_rows)}
<details><summary>Table</summary><table><thead><tr><th>Pathway</th><th>Verdict</th><th>ΔG‡</th><th>2σ</th>
<th>Limit</th><th>ΔE‡ (elec.)</th><th>ν‡ (cm⁻¹)</th><th>Reason</th></tr></thead><tbody>{overview_rows}</tbody></table></details>
<h3>TS validation gates</h3>
<p class="muted small">Hover a cell for the gate's reason.</p>
{_gate_table(all_rows)}
</section>
{campaigns}
</main><div id="tip" role="tooltip"></div></div><script>{_JS}</script></body></html>
"""
