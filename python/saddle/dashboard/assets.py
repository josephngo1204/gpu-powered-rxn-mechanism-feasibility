"""Stylesheet and script for the dashboard page.

Colors are design tokens: one dict per theme, rendered into CSS custom
properties. Series slots 1-3 were checked with the dataviz palette validator
(lightness band, chroma, CVD separation all-pairs) on both surfaces. Light
slot 3 is below 3:1 contrast, so every series is direct-labelled and every
chart has a table view. Status colors always ship with an icon and a word.
"""

LIGHT = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "text": "#0b0b0b",
    "text-secondary": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
    "border": "rgba(11,11,11,.10)",
    "series-1": "#2a78d6",
    "series-2": "#eb6834",
    "series-3": "#1baf7a",
    "wash": "rgba(42,120,214,.10)",
    "band-1": "rgba(42,120,214,.28)",
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
}

DARK = {
    **LIGHT,
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "text": "#f0efec",
    "text-secondary": "#c3c2b7",
    "gridline": "#2c2c2a",
    "baseline": "#383835",
    "border": "rgba(255,255,255,.10)",
    "series-1": "#3987e5",
    "series-2": "#d95926",
    "series-3": "#199e70",
    "wash": "rgba(57,135,229,.14)",
    "band-1": "rgba(57,135,229,.34)",
}


def _tokens(theme: dict[str, str]) -> str:
    return "".join(f"--{name}:{value};" for name, value in theme.items())


_RULES = """
body { margin: 0 }
.viz-root { background: var(--page); color: var(--text); min-height: 100vh; padding: 24px;
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, Arial, "DejaVu Sans", sans-serif }
main { max-width: 1100px; margin: 0 auto }
h1 { font-size: 22px; margin: 0 0 4px }
h2 { font-size: 18px; margin: 28px 0 4px }
h3 { font-size: 15px; margin: 20px 0 4px }
h4 { font-size: 14px; margin: 16px 0 4px }
.muted { color: var(--text-secondary) }
.small { font-size: 12px }
.mono, code { font-family: ui-monospace, monospace; font-size: 12px }
section.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 20px; margin: 16px 0 }
section.pathway { border-top: 1px solid var(--gridline); margin-top: 16px }
figure { margin: 12px 0 }
figcaption { font-weight: 600; margin-bottom: 6px }
svg { width: 100%; height: auto; display: block; overflow: visible }

/* chart marks */
.grid { stroke: var(--gridline); stroke-width: 1 }
.base { stroke: var(--baseline); stroke-width: 1 }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums }
.lbl { fill: var(--text); font-size: 12px }
.val { fill: var(--text-secondary); font-size: 11px; font-variant-numeric: tabular-nums }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round }
.line.s1 { stroke: var(--series-1) } .line.s2 { stroke: var(--series-2) }
.line.s3 { stroke: var(--series-3) }
.dot { stroke: var(--surface); stroke-width: 2 }
.dot.s1 { fill: var(--series-1) } .dot.s2 { fill: var(--series-2) } .dot.s3 { fill: var(--series-3) }
.bar.s1 { fill: var(--series-1) }
.whisker { stroke: var(--series-1); stroke-width: 2; stroke-linecap: round; opacity: .55 }
.limit, .bracket { stroke: var(--text); stroke-width: 2 }
.band { fill: var(--wash) }
.band-s1 { fill: var(--band-1); rx: 6 }
.ref { fill: none; stroke: var(--text); stroke-width: 2 }
.row { fill: transparent } .hit:hover .row { fill: var(--wash) } .hitbox { fill: transparent }

/* status: always icon + word, never color alone */
.st-good { fill: var(--good); color: var(--good); font-weight: 700 }
.st-critical { fill: var(--critical); color: var(--critical); font-weight: 700 }
.st-warning { fill: var(--text); color: var(--text); font-weight: 700 }
.badge { display: inline-block; font-size: 12px; font-weight: 600; border-radius: 999px;
  padding: 1px 10px; margin-left: 6px; border: 1px solid currentColor }
.b-good { color: var(--good) } .b-critical { color: var(--critical) }
.b-warning { color: var(--text); background: var(--warning) }

/* layout pieces */
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0 }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 16px; min-width: 120px }
.tile .label { color: var(--text-secondary); font-size: 12px }
.tile .value { font-size: 24px; font-weight: 600 }
.headlines { display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)) }
.headline { border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px }
.headline p { margin: 6px 0 0 }
dl.facts { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
  gap: 6px 16px; margin: 8px 0 }
dl.facts dt { color: var(--text-secondary); font-size: 12px }
dl.facts dd { margin: 0; font-variant-numeric: tabular-nums }
.grid2 { display: grid; grid-template-columns: 1fr; gap: 8px }
@media (min-width: 1000px) { .grid2 { grid-template-columns: 1fr 1fr } }
ul.checklist { list-style: none; padding: 0; margin: 6px 0 }
ul.checklist li { padding: 3px 0; border-bottom: 1px solid var(--gridline) }
ul.checklist .why { color: var(--text-secondary); font-size: 12px; display: block; margin-left: 22px }
table { border-collapse: collapse; font-size: 12px; margin: 6px 0 }
th, td { border-bottom: 1px solid var(--gridline); padding: 4px 8px; text-align: left }
td.n { text-align: right; font-variant-numeric: tabular-nums }
table.gates td { text-align: center }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: var(--text-secondary);
  margin-bottom: 4px }
.key { display: inline-flex; align-items: center; gap: 6px }
details { margin: 6px 0 }
summary { cursor: pointer; color: var(--text-secondary); font-size: 12px }
#tip { position: fixed; pointer-events: none; display: none; z-index: 9; max-width: 380px;
  background: var(--surface); color: var(--text); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12) }
.theme { float: right; font: inherit; font-size: 12px; background: var(--surface); color: var(--text);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; cursor: pointer }
"""

CSS = (
    f".viz-root{{color-scheme:light;{_tokens(LIGHT)}}}\n"
    '@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root'
    f"{{color-scheme:dark;{_tokens(DARK)}}}}}\n"
    f':root[data-theme="dark"] .viz-root{{color-scheme:dark;{_tokens(DARK)}}}\n' + _RULES
)

JS = """
(() => {
  const tip = document.getElementById('tip');
  document.addEventListener('mousemove', (event) => {
    const target = event.target.closest('[data-tip]');
    if (!target) { tip.style.display = 'none'; return; }
    tip.textContent = target.dataset.tip;
    tip.style.display = 'block';
    tip.style.left = Math.min(event.clientX + 14, innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = (event.clientY + 14) + 'px';
  });
  document.querySelectorAll('[data-tip] title').forEach((n) => n.remove());
  document.querySelectorAll('td[data-tip]').forEach((n) => n.removeAttribute('title'));
  document.getElementById('theme').addEventListener('click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme
      ? root.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
  });
})();
"""
