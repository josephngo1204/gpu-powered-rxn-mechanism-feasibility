"""Self-contained HTML results dashboard over finished campaigns.

Reads only ``campaign.json`` documents and writes one HTML file with inline
SVG: no plotting dependency, no JavaScript library and no network access.
Modules: ``views`` (typed input), ``explain`` (plain-language text),
``svg`` and ``charts`` (drawing), ``assets`` (CSS/JS), ``page`` (layout).
"""

from saddle.dashboard.page import Run, render_dashboard

__all__ = ["Run", "render_dashboard"]
