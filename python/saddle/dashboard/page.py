"""Assembles the dashboard page, from plain-language summary down to technical detail.

Page order:
1. In plain words: one sentence per pathway, and its half-life on a time axis.
2. Technical overview: barriers against the limit, and the G1-G8 gate matrix.
3. One section per campaign: energy profile, reference comparison, and per
   pathway the gate checklist, uncertainty budget and sensitivity.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from saddle.dashboard import assets, charts
from saddle.dashboard.explain import GATE_QUESTIONS, headline, verdict_status
from saddle.dashboard.svg import esc
from saddle.dashboard.views import Campaign, Pathway, unique_anchors

GATE_CODES = tuple(GATE_QUESTIONS)


@dataclass(frozen=True)
class Run:
    """One finished campaign: a display label and its ``campaign.json`` document."""

    label: str
    doc: Mapping[str, Any]


def _num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]], cls: str = "") -> str:
    """An HTML table; cells are pre-rendered (escaped) HTML fragments."""
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def _details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary>{body}</details>"


def verdict_badge(verdict: str) -> str:
    status, icon = verdict_status(verdict)
    return (
        f'<span class="badge b-{status}"><span aria-hidden="true">{icon}</span> '
        f"{esc(verdict)}</span>"
    )


def _plain_words(campaigns: Sequence[Campaign]) -> str:
    cards = "".join(
        f'<div class="headline"><strong>{esc(c.label)} / {esc(p.name)}</strong>'
        f"{verdict_badge(p.verdict)}"
        f"<p>{esc(headline(p, c.temperature_k, c.target_half_life_s))}</p></div>"
        for c in campaigns
        for p in c.pathways
    )
    rows = [
        charts.HalfLifeRow(f"{c.label} / {p.name}", p, c.temperature_k, c.target_half_life_s)
        for c in campaigns
        if c.temperature_k is not None
        for p in c.pathways
    ]
    return f"""
<section class="card">
  <h2 style="margin-top:0">In plain words</h2>
  <p class="muted small">A reaction is judged feasible when half of the starting material would
  react within the target time. The half-life follows from the free-energy barrier through
  transition-state theory; 1 kcal/mol of barrier changes it about 5×.</p>
  <div class="headlines">{cards}</div>
  <figure><figcaption>How fast is it? Predicted half-life on a log time axis</figcaption>
    {charts.half_life_chart(rows)}
  </figure>
</section>"""


def _gate_cell(p: Pathway, code: str) -> str:
    gate = next((g for g in p.gates if g.code == code), None)
    if gate is None:
        return '<span class="muted" title="not run">·</span>'
    reason = f"{gate.name}: {gate.reason}"
    if not gate.passed and gate.failure_action:
        reason += f" → {gate.failure_action}"
    mark = '<span class="st-good">✓</span>' if gate.passed else '<span class="st-critical">×</span>'
    return f'<span class="hit" title="{esc(reason)}" data-tip="{esc(reason)}">{mark}</span>'


def _technical_overview(campaigns: Sequence[Campaign]) -> str:
    rows = [(f"{c.label} / {p.name}", p) for c in campaigns for p in c.pathways]
    summary = _table(
        ["Pathway", "Verdict", "ΔG‡", "2σ", "Limit", "ΔE‡ (electronic)", "ν‡ (cm⁻¹)", "Reason"],
        [
            [
                esc(label),
                verdict_badge(p.verdict),
                _num(p.barrier),
                _num(p.two_sigma),
                _num(p.limit),
                _num(p.electronic_barrier),
                _num(p.imaginary_cm, 1),
                esc(p.reason),
            ]
            for label, p in rows
        ],
    )
    gates = _table(
        ["Pathway", *GATE_CODES],
        [[esc(label), *(_gate_cell(p, code) for code in GATE_CODES)] for label, p in rows],
        cls="gates",
    )
    legend = "".join(
        f"<li><strong>{code}</strong> {esc(q)}</li>" for code, q in GATE_QUESTIONS.items()
    )
    return f"""
<section class="card">
  <h2 style="margin-top:0">Barriers against the feasibility limit</h2>
  <p class="muted small">Dot = ΔG‡, whisker = ±2σ. SUPPORTED when ΔG‡ + 2σ is below the limit,
  INFEASIBLE when ΔG‡ − 2σ is above it, otherwise INCONCLUSIVE.</p>
  {charts.barrier_chart(rows)}
  {_details("Table", summary)}
  <h3>Transition-state checks (G1–G8)</h3>
  <p class="muted small">A barrier only counts when all eight checks pass. Hover a mark for
  its reason.</p>
  {gates}
  {_details("What each check asks", f'<ul class="small">{legend}</ul>')}
</section>"""


def _reference_table(c: Campaign) -> str:
    ref, p = c.reference, c.reference_pathway()
    if ref is None:
        return ""
    if p is None:
        return (
            "<p class='muted small'>Reference values exist but name no pathway of this "
            "multi-pathway campaign, so they are not compared.</p>"
        )
    pairs = [
        ("Complex ΔE", p.reactant_minimum, ref.complex_energy),
        ("TS ΔE", p.electronic_barrier, ref.ts_energy),
        ("ΔG‡", p.barrier, ref.barrier),
    ]
    rows = [
        [esc(name), _num(got), _num(want), "—" if got is None else _num(got - want)]
        for name, got, want in pairs
        if want is not None
    ]
    return (
        f"<h4>Against reference ({esc(p.name)})</h4>"
        + _table(["Quantity", "Computed", "Reference", "Δ"], rows)
        + f"<p class='muted small'>{esc(ref.citation)}</p>"
    )


def _profile_figure(c: Campaign) -> str:
    keys = "".join(
        f'<span class="key"><svg width="22" height="10"><line class="line s{i + 1}" x1="0" '
        f'x2="22" y1="5" y2="5"/></svg>{esc(p.name)}</span>'
        for i, p in enumerate(c.pathways[: charts.MAX_SERIES])
    )
    if c.reference_pathway() is not None:
        keys += (
            '<span class="key"><svg width="12" height="12"><rect class="ref" x="1" y="1" '
            'width="10" height="10"/></svg>reference</span>'
        )
    folded = len(c.pathways) - charts.MAX_SERIES
    table = _table(
        ["Pathway", *charts.PROFILE_STAGES],
        [[esc(p.name), *(_num(v) for v in charts.profile_points(p))] for p in c.pathways],
    )
    return f"""
<figure><figcaption>The energy hill: electronic energy relative to separated reactants (kcal/mol)</figcaption>
  <p class="muted small">Reactants climb to the transition state (the top of the pass) and
  settle into the product side. For two molecules that attract, the minima on either side are
  loosely bound complexes.</p>
  <div class="legend">{keys}</div>
  {charts.energy_profile_chart(c)}
  {f"<p class='muted small'>{folded} more pathways are in the table only.</p>" if folded > 0 else ""}
  {_details("Table", table)}
</figure>"""


def _checklist(p: Pathway) -> str:
    items = []
    for g in p.gates:
        mark = (
            '<span class="st-good">✓</span>' if g.passed else '<span class="st-critical">×</span>'
        )
        action = f" → {esc(g.failure_action)}" if not g.passed and g.failure_action else ""
        question = GATE_QUESTIONS.get(g.code, g.name)
        items.append(
            f"<li>{mark} <strong>{esc(g.code)}</strong> {esc(question)}"
            f'<span class="why">{esc(g.reason)}{action}</span></li>'
        )
    return f'<ul class="checklist">{"".join(items)}</ul>'


def _pathway_section(c: Campaign, p: Pathway) -> str:
    facts = [
        ("Imaginary frequency", f"{_num(p.imaginary_cm, 1)} cm⁻¹"),
        ("Electronic barrier", f"{_num(p.electronic_barrier)} kcal/mol"),
        ("ΔG‡ ± 2σ", f"{_num(p.barrier)} ± {_num(p.two_sigma)} kcal/mol"),
        ("Gradient noise", "—" if p.noise_epsilon is None else f"{p.noise_epsilon:.1e} Eh/bohr"),
        ("Noise floor", f"{_num(p.noise_floor_cm, 1)} cm⁻¹"),
        ("Gradients evaluated", _num(p.n_gradients, 0)),
        ("Tunneling κ (Wigner / Eckart)", f"{_num(p.wigner, 3)} / {_num(p.eckart, 3)}"),
    ]
    facts_html = "".join(f"<div><dt>{esc(k)}</dt><dd>{v}</dd></div>" for k, v in facts)
    budget_table = _table(
        ["Component", "σ", "Source", "Note"],
        [
            [
                esc(t.component),
                "qualitative" if t.source == "qualitative" else _num(t.value, 3),
                esc(t.source),
                esc(t.note),
            ]
            for t in p.uncertainty
        ],
    )
    freqs = ", ".join(f"{f:.1f}" for f in p.frequencies_cm) or "—"
    notes = "".join(f"<li>{esc(n)}</li>" for n in p.notes)
    return f"""
<section class="pathway">
  <h3>{esc(c.label)} / <code>{esc(p.name)}</code> {verdict_badge(p.verdict)}</h3>
  <p>{esc(headline(p, c.temperature_k, c.target_half_life_s))}</p>
  <p class="muted small">{esc(p.reason)}</p>
  <dl class="facts">{facts_html}</dl>
  <h4>Transition-state checks</h4>
  {_checklist(p)}
  <div class="grid2">
    <figure><figcaption>Where the error bar comes from (σ, kcal/mol; total {_num(p.sigma)})</figcaption>
      {charts.budget_chart(p)}
      {_details("Table", budget_table)}
    </figure>
    <figure><figcaption>Does the barrier move when settings change?</figcaption>
      {charts.sensitivity_chart(p)}
    </figure>
  </div>
  {_details("Harmonic frequencies (cm⁻¹)", f'<p class="mono">{freqs}</p>')}
  {_details("Notes", f"<ul>{notes}</ul>") if notes else ""}
</section>"""


def _campaign_section(c: Campaign) -> str:
    rankings = "".join(
        f"<li>{esc(r.first)} vs {esc(r.second)}: "
        f"{esc(r.lower) + ' is lower' if r.resolved else 'not resolved'} "
        f"<span class='muted'>({esc(r.reason)})</span></li>"
        for r in c.rankings
    )
    conditions = (
        f"{esc(c.model)} · T = {_num(c.temperature_k)} K · target half-life "
        f"{_num(c.target_half_life_s, 0)} s · limit {_num(c.limit)} kcal/mol · code "
        f"{esc(c.commit)} · {esc(c.gpu)} · cuEST {esc(c.cuest)}"
    )
    return f"""
<section class="card" id="{esc(c.anchor)}">
  <h2 style="margin-top:0">{esc(c.label)} <span class="muted">— {esc(c.hypothesis)}</span></h2>
  <p class="muted small">{conditions}</p>
  {f"<ul>{rankings}</ul>" if rankings else ""}
  {_profile_figure(c)}
  {_reference_table(c)}
  {"".join(_pathway_section(c, p) for p in c.pathways)}
</section>"""


def render_dashboard(runs: Sequence[Run], references: Mapping[str, Any] | None = None) -> str:
    """One self-contained HTML page over every run. ``references`` is references.yaml."""
    anchors = unique_anchors([r.label for r in runs])
    campaigns = [
        Campaign.from_json(r.label, a, r.doc, references or {})
        for r, a in zip(runs, anchors, strict=True)
    ]
    verdicts = Counter(p.verdict for c in campaigns for p in c.pathways)
    tiles = [("Campaigns", len(campaigns)), ("Pathways", sum(verdicts.values()))] + [
        (v.replace("_", " ").capitalize(), n) for v, n in sorted(verdicts.items())
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="label">{esc(k)}</div><div class="value">{v}</div></div>'
        for k, v in tiles
    )
    toc = " · ".join(f'<a href="#{esc(c.anchor)}">{esc(c.label)}</a>' for c in campaigns)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Saddle results dashboard</title><style>{assets.CSS}</style></head>
<body><div class="viz-root"><main>
<button class="theme" id="theme" type="button">Toggle theme</button>
<h1>Saddle results dashboard</h1>
<p class="muted">Is each proposed reaction step fast enough? Campaigns: {toc or "—"}</p>
<div class="tiles">{tiles_html}</div>
{_plain_words(campaigns)}
{_technical_overview(campaigns)}
{"".join(_campaign_section(c) for c in campaigns)}
</main><div id="tip" role="tooltip"></div></div><script>{assets.JS}</script></body></html>
"""
