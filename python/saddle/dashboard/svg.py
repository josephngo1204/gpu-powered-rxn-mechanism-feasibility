"""Minimal SVG building blocks: an element buffer, axis ticks and a row-chart frame.

All text goes through ``esc``; numbers are formatted here, so no caller ever
puts untrusted text into markup unescaped.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from html import escape

LABEL_ROW_HEIGHT = 30


def esc(x: object) -> str:
    return escape(str(x), quote=True)


class Svg:
    """Accumulates SVG elements and renders a responsive ``<svg>``."""

    def __init__(self, width: float, height: float, label: str) -> None:
        self.width, self.height, self.label = width, height, label
        self._parts: list[str] = []

    def raw(self, markup: str) -> None:
        """Append markup built from numbers and ``esc``-ed text only."""
        self._parts.append(markup)

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str) -> None:
        self.raw(f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')

    def circle(self, x: float, y: float, r: float, cls: str) -> None:
        self.raw(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r:g}"/>')

    def rect(self, x: float, y: float, w: float, h: float, cls: str) -> None:
        self.raw(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"/>')

    def text(self, x: float, y: float, s: str, cls: str = "lbl", anchor: str = "start") -> None:
        self.raw(
            f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{esc(s)}</text>'
        )

    def begin_hover(self, tip: str) -> None:
        """Start a group with a tooltip (native ``<title>`` plus ``data-tip`` for the page JS)."""
        self.raw(f'<g class="hit" data-tip="{esc(tip)}"><title>{esc(tip)}</title>')

    def end_hover(self) -> None:
        self.raw("</g>")

    def render(self) -> str:
        return (
            f'<svg viewBox="0 0 {self.width:g} {self.height:g}" role="img" '
            f'aria-label="{esc(self.label)}" preserveAspectRatio="xMinYMin meet">'
            + "".join(self._parts)
            + "</svg>"
        )


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round tick values (1, 2, 2.5 or 5 times a power of ten) covering [lo, hi]."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("nice_ticks needs finite bounds")
    if hi <= lo:
        hi = lo + 1.0
    raw_step = (hi - lo) / target
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw_step)
    first = math.floor(lo / step)
    last = math.ceil(hi / step - 1e-9)
    return [round(k * step, 10) for k in range(first, last + 1)]


def linear_scale(
    domain: tuple[float, float], pixels: tuple[float, float]
) -> Callable[[float], float]:
    """Map data values in ``domain`` linearly onto the pixel range."""
    (d0, d1), (p0, p1) = domain, pixels
    span = d1 - d0 or 1.0
    return lambda v: p0 + (v - d0) / span * (p1 - p0)


@dataclass(frozen=True)
class RowChart:
    """Horizontal one-row-per-item layout: labels left, plot middle, values right."""

    n_rows: int
    width: float = 760
    left: float = 200
    right: float = 130
    top: float = 12
    row_height: float = 32
    footer: float = 56

    @property
    def height(self) -> float:
        return self.top + self.row_height * self.n_rows + self.footer

    @property
    def plot_bottom(self) -> float:
        return self.top + self.row_height * self.n_rows

    def x_scale(self, domain: tuple[float, float]) -> Callable[[float], float]:
        return linear_scale(domain, (self.left, self.width - self.right))

    def row_center(self, i: int) -> float:
        return self.top + self.row_height * (i + 0.5)

    def x_axis(
        self,
        svg: Svg,
        ticks: list[float],
        sx: Callable[[float], float],
        unit: str,
        tick_label: Callable[[float], str] = lambda t: f"{t:g}",
    ) -> None:
        for t in ticks:
            svg.line(sx(t), self.top - 4, sx(t), self.plot_bottom, "grid")
            svg.text(sx(t), self.plot_bottom + 16, tick_label(t), "tick", "middle")
        svg.text(self.width - self.right, self.plot_bottom + 32, unit, "tick", "end")

    def begin_row(self, svg: Svg, i: int, label: str, tip: str) -> float:
        """Open a hoverable row with its left label; returns the row's center y."""
        y = self.row_center(i)
        svg.begin_hover(tip)
        svg.rect(0, y - self.row_height / 2, self.width, self.row_height, "row")
        svg.text(self.left - 10, y + 4, label, "lbl", "end")
        return y

    def footnote(self, svg: Svg, text: str) -> None:
        svg.text(self.left, self.height - 4, text, "tick")
