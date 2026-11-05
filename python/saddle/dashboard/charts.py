"""The dashboard's charts. Each takes typed views and returns one ``<svg>`` string.

Every chart is also available as a table on the page, and every series is
direct-labelled, so no chart relies on color alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from saddle.dashboard.explain import duration, safe_half_life_s, verdict_status
from saddle.dashboard.svg import RowChart, Svg, esc, linear_scale, nice_ticks
from saddle.dashboard.views import Campaign, Pathway

MAX_SERIES = 3  # categorical color slots validated all-pairs in light and dark


def _empty(message: str) -> str:
    return f'<p class="muted">{message}</p>'


# --------------------------------------------------------------------------- barrier


def barrier_chart(rows: Sequence[tuple[str, Pathway]]) -> str:
    """Free-energy barrier per pathway: dot, ±2σ whisker, and the feasibility limit."""
    rows = [(label, p) for label, p in rows if p.barrier is not None]
    if not rows:
        return _empty("No pathway produced a barrier.")
    chart = RowChart(len(rows))
    extent = [0.0]
    for _, p in rows:
        assert p.barrier is not None
        half_width = p.two_sigma or 0.0
        extent += [p.barrier - half_width, p.barrier + half_width]
        if p.limit is not None:
            extent.append(p.limit)
    ticks = nice_ticks(min(extent), max(extent))
    sx = chart.x_scale((ticks[0], ticks[-1]))
    svg = Svg(chart.width, chart.height, "Free-energy barrier per pathway with 2 sigma and limit")
    chart.x_axis(svg, ticks, sx, "ΔG‡ (kcal/mol)")
    for i, (label, p) in enumerate(rows):
        assert p.barrier is not None
        half_width = p.two_sigma or 0.0
        tip = f"{label}: ΔG‡ {p.barrier:.2f} ± {half_width:.2f} (2σ) kcal/mol; {p.verdict}"
        y = chart.begin_row(svg, i, label, tip)
        if p.limit is not None:
            svg.line(sx(p.limit), y - 12, sx(p.limit), y + 12, "limit")
        svg.line(sx(p.barrier - half_width), y, sx(p.barrier + half_width), y, "whisker")
        svg.circle(sx(p.barrier), y, 5, "dot s1")
        status, icon = verdict_status(p.verdict)
        svg.raw(
            f'<text class="lbl" x="{chart.width - chart.right + 12}" y="{y + 4:.1f}">'
            f'<tspan class="st-{status}">{icon}</tspan> {esc(p.verdict)} {p.barrier:.1f}</text>'
        )
        svg.end_hover()
    chart.footnote(svg, "| = feasibility limit (the barrier that gives the target half-life)")
    return svg.render()


# --------------------------------------------------------------------------- half-life

# Landmarks for the log-time axis: (seconds, label).
_TIME_LANDMARKS = (
    (1e-12, "1 ps"),
    (1e-9, "1 ns"),
    (1e-6, "1 µs"),
    (1e-3, "1 ms"),
    (1.0, "1 s"),
    (3600.0, "1 h"),
    (86400.0, "1 day"),
    (3.15576e7, "1 yr"),
    (3.15576e10, "1000 yr"),
    (3.15576e13, "1 Myr"),
    (3.15576e16, "1 Gyr"),
)
_LOG_T_MIN, _LOG_T_MAX = -13.0, 18.0  # clamp: 0.1 ps .. ~3 Gyr


@dataclass(frozen=True)
class HalfLifeRow:
    label: str
    pathway: Pathway
    temperature_k: float
    target_s: float | None


def _log_seconds(barrier: float, temperature_k: float) -> float:
    t = safe_half_life_s(barrier, temperature_k)
    return _LOG_T_MAX if math.isinf(t) else min(max(math.log10(t), _LOG_T_MIN), _LOG_T_MAX)


def _spaced_landmarks(
    sx: Callable[[float], float], lo: float, hi: float, min_gap_px: float = 48
) -> list[tuple[float, str]]:
    """Time landmarks inside [lo, hi] (log10 s), dropping any that would crowd the previous."""
    kept: list[tuple[float, str]] = []
    for seconds, name in _TIME_LANDMARKS:
        x = math.log10(seconds)
        if lo <= x <= hi and (not kept or sx(x) - sx(kept[-1][0]) >= min_gap_px):
            kept.append((x, name))
    return kept


def half_life_chart(rows: Sequence[HalfLifeRow]) -> str:
    """How long half the reactant takes to react, on a log time axis.

    The band spans the ±2σ barrier interval (a higher barrier is slower); the
    vertical mark is the target half-life the verdict is judged against.
    """
    rows = [r for r in rows if r.pathway.barrier is not None]
    if not rows:
        return _empty("No pathway produced a barrier.")
    chart = RowChart(len(rows), right=170, row_height=36)
    spans = []
    for r in rows:
        b, w = r.pathway.barrier, r.pathway.two_sigma or 0.0
        assert b is not None
        spans.append(
            (
                _log_seconds(b - w, r.temperature_k),
                _log_seconds(b, r.temperature_k),
                _log_seconds(b + w, r.temperature_k),
            )
        )
    extent = [x for s in spans for x in s] + [
        math.log10(r.target_s) for r in rows if r.target_s is not None
    ]
    lo, hi = math.floor(min(extent)) - 1, math.ceil(max(extent)) + 1
    sx = chart.x_scale((lo, hi))
    svg = Svg(
        chart.width, chart.height, "Predicted half-life per pathway on a logarithmic time axis"
    )
    for x, name in _spaced_landmarks(sx, lo, hi):
        svg.line(sx(x), chart.top - 4, sx(x), chart.plot_bottom, "grid")
        svg.text(sx(x), chart.plot_bottom + 16, name, "tick", "middle")
    svg.text(
        chart.width - chart.right, chart.plot_bottom + 32, "half-life (log scale)", "tick", "end"
    )
    for i, (r, (fast, central, slow)) in enumerate(zip(rows, spans, strict=True)):
        assert r.pathway.barrier is not None
        t_central = safe_half_life_s(r.pathway.barrier, r.temperature_k)
        tip = (
            f"{r.label}: half-life about {duration(t_central)}; ±2σ range "
            f"{duration(10**fast)} to {duration(10**slow)}"
        )
        y = chart.begin_row(svg, i, r.label, tip)
        svg.rect(sx(fast), y - 6, max(sx(slow) - sx(fast), 2.0), 12, "band-s1")
        if r.target_s is not None:
            tx = sx(math.log10(r.target_s))
            svg.line(tx, y - 14, tx, y + 14, "limit")
        svg.circle(sx(central), y, 5, "dot s1")
        svg.text(chart.width - chart.right + 12, y + 4, duration(t_central), "lbl")
        svg.end_hover()
    chart.footnote(svg, "| = target half-life · bar = range across the ±2σ error bar")
    return svg.render()


# --------------------------------------------------------------------------- energy profile

PROFILE_STAGES = (
    "Separated reactants",
    "Minimum before the TS",
    "Transition state",
    "Minimum after the TS",
)


def profile_points(p: Pathway) -> tuple[float | None, ...]:
    """Electronic energies (kcal/mol vs separated reactants) at each PROFILE_STAGES entry."""
    return (0.0, p.reactant_minimum, p.electronic_barrier, p.product_minimum)


def energy_profile_chart(c: Campaign) -> str:
    """The energy 'hill' each pathway climbs, with reference values where known.

    A single pathway also gets a bracket showing its electronic barrier.
    """
    shown = c.pathways[:MAX_SERIES]
    series = [[(i, v) for i, v in enumerate(profile_points(p)) if v is not None] for p in shown]
    ref_path = c.reference_pathway()
    ref_points: dict[int, float] = {}
    if c.reference is not None and ref_path is not None:
        if c.reference.complex_energy is not None:
            ref_points[1] = c.reference.complex_energy
        if c.reference.ts_energy is not None:
            ref_points[2] = c.reference.ts_energy
    values = [v for pts in series for _, v in pts] + list(ref_points.values())
    if len(values) < 2:
        return _empty("No energy profile available.")
    ticks = nice_ticks(min(values), max(values), 5)
    width, height, left, right, top, bottom = 760, 330, 60, 170, 36, 44
    sx = linear_scale((0, len(PROFILE_STAGES) - 1), (left + 20, width - right))
    sy = linear_scale((ticks[0], ticks[-1]), (height - bottom, top))
    svg = Svg(width, height, "Electronic energy profile relative to separated reactants")
    for t in ticks:
        svg.line(left, sy(t), width - right + 20, sy(t), "grid")
        svg.text(left - 8, sy(t) + 4, f"{t:g}", "tick", "end")
    svg.line(left, sy(0), width - right + 20, sy(0), "base")
    for i, stage in enumerate(PROFILE_STAGES):
        svg.text(sx(i), height - bottom + 20, stage, "tick", "middle")
    svg.text(left - 50, 14, "energy (kcal/mol)", "tick")

    if len(shown) == 1 and shown[0].electronic_barrier is not None:
        # Bracket from the reactant level up to the TS: "how high is the hill".
        bx, top_y = sx(2) + 16, sy(shown[0].electronic_barrier)
        svg.line(bx, sy(0), bx, top_y, "bracket")
        svg.line(bx - 4, sy(0), bx + 4, sy(0), "bracket")
        svg.line(bx - 4, top_y, bx + 4, top_y, "bracket")
        svg.text(bx + 8, (sy(0) + top_y) / 2 + 4, f"ΔE‡ {shown[0].electronic_barrier:+.1f}")

    label_ys: list[float] = []
    for k, (p, pts) in enumerate(zip(shown, series, strict=True)):
        slot = f"s{k + 1}"
        path = " ".join(
            f"{'M' if j == 0 else 'L'}{sx(i):.1f},{sy(v):.1f}" for j, (i, v) in enumerate(pts)
        )
        svg.raw(f'<path class="line {slot}" d="{path}"/>')
        for i, v in pts:
            svg.begin_hover(f"{p.name} · {PROFILE_STAGES[i]}: {v:+.2f} kcal/mol")
            svg.circle(sx(i), sy(v), 12, "hitbox")
            svg.circle(sx(i), sy(v), 4.5, f"dot {slot}")
            svg.end_hover()
        # Direct label at the line's end, nudged down past earlier labels.
        i_end, v_end = pts[-1]
        y = sy(v_end) + 4
        while any(abs(y - other) < 14 for other in label_ys):
            y += 14
        label_ys.append(y)
        svg.circle(sx(i_end) + 22, y - 4, 4, f"dot {slot}")
        svg.text(sx(i_end) + 32, y, f"{p.name} ({v_end:+.1f})")
    for i, v in ref_points.items():
        svg.begin_hover(f"Reference · {PROFILE_STAGES[i]}: {v:+.2f} kcal/mol")
        svg.circle(sx(i), sy(v), 12, "hitbox")
        svg.rect(sx(i) - 5, sy(v) - 5, 10, 10, "ref")
        svg.end_hover()
    return svg.render()


# --------------------------------------------------------------------------- per pathway


def budget_chart(p: Pathway) -> str:
    """Uncertainty components as bars from zero; one series, value at the tip."""
    terms = [t for t in p.uncertainty if t.value is not None]
    if not terms:
        return _empty("No quantified uncertainty components.")
    chart = RowChart(len(terms), width=560, left=120, right=60, top=8, row_height=28, footer=40)
    ticks = nice_ticks(0.0, max(max(t.value or 0.0 for t in terms), 1e-9), 5)
    sx = chart.x_scale((0.0, ticks[-1]))
    svg = Svg(chart.width, chart.height, "Uncertainty budget components")
    chart.x_axis(svg, ticks, sx, "σ (kcal/mol)")
    for i, t in enumerate(terms):
        assert t.value is not None
        y = chart.begin_row(
            svg, i, t.component, f"{t.component}: {t.value:.3g} ({t.source}) {t.note}"
        )
        w = max(sx(t.value) - chart.left, 0.0)
        r = min(4.0, w / 2)  # rounded data end, square at the baseline
        svg.raw(
            f'<path class="bar s1" d="M{chart.left},{y - 8:.1f} h{w - r:.2f} '
            f"a{r:.2f},{r:.2f} 0 0 1 {r:.2f},{r:.2f} v{16 - 2 * r:.2f} "
            f'a{r:.2f},{r:.2f} 0 0 1 -{r:.2f},{r:.2f} h-{w - r:.2f} z"/>'
        )
        svg.text(chart.left + w + 6, y + 4, f"{t.value:.3g} · {t.source}", "val")
        svg.end_hover()
    return svg.render()


def sensitivity_chart(p: Pathway) -> str:
    """The barrier under each FD step and robustness variant, against the baseline ±2σ."""
    if not p.sensitivity or p.barrier is None:
        return _empty("No variants were run.")
    base, half_width = p.barrier, p.two_sigma or 0.0
    rows = p.sensitivity
    chart = RowChart(len(rows), width=560, left=160, right=60, top=8, row_height=26, footer=56)
    ticks = nice_ticks(
        min([v.barrier for v in rows] + [base - half_width]),
        max([v.barrier for v in rows] + [base + half_width]),
        5,
    )
    sx = chart.x_scale((ticks[0], ticks[-1]))
    svg = Svg(chart.width, chart.height, "Barrier sensitivity to step size and robustness variants")
    chart.x_axis(svg, ticks, sx, "ΔG‡ (kcal/mol)")
    svg.rect(
        sx(base - half_width),
        chart.top - 4,
        sx(base + half_width) - sx(base - half_width),
        chart.plot_bottom - chart.top + 4,
        "band",
    )
    svg.line(sx(base), chart.top - 4, sx(base), chart.plot_bottom, "limit")
    for i, v in enumerate(rows):
        shift = v.barrier - base
        y = chart.begin_row(
            svg, i, v.label, f"{v.label}: {v.barrier:.3f} kcal/mol ({shift:+.3f}) {v.verdict}"
        )
        svg.circle(sx(v.barrier), y, 5, "dot s1")
        svg.text(
            chart.width - chart.right + 10,
            y + 4,
            "0.00" if abs(shift) < 0.005 else f"{shift:+.2f}",
            "val",
        )
        svg.end_hover()
    chart.footnote(svg, "| baseline · shaded = baseline ±2σ · right: shift")
    return svg.render()
