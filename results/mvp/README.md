# MVP results baseline (2026-11-05)

Result summaries (`campaign.json`) copied from finished `runs/`, with their markdown reports
and a combined dashboard. HDF5 and SQLite stores are not committed (spec §12).
Each `campaign.json` records the code commit, cuEST version, GPU and math mode it was produced with.

| Run | Produced by | Model |
|---|---|---|
| `c1/` | commit 6429ab7, cuEST 0.2.2, H200 NVL MIG 1g.18gb, native FP64 | wB97X-V/def2-SVPD, gas |
| `fake/` | commit b1c0483 (CPU) | fake pair-model surface |

Regenerate: `saddle dashboard results/mvp --out results/mvp/dashboard.html`
