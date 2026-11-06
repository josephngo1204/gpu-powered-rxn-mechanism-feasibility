# Results: C1, Cl⁻ + CH₃Cl identity SN2 (gas phase)

- [`c1/report.md`](c1/report.md): full report with gates, uncertainty budget, sensitivity, tunneling and IRC
- [`c1/campaign.json`](c1/campaign.json): machine-readable summary with provenance (code commit, cuEST version, GPU, math mode)
- [`dashboard.html`](dashboard.html): self-contained interactive dashboard; open it in a browser

| Setting | Value |
|---|---|
| Model | ωB97X-V/def2-SVPD, density-fitted, (75,302) grid |
| Hardware | NVIDIA H200 NVL, one MIG 1g.18gb slice, native FP64 |
| Library | cuEST 0.2.2 |

Regenerate the dashboard: `saddle dashboard results/mvp --out results/mvp/dashboard.html`
