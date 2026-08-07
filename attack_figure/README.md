# Attack figures

PNG figures generated from `tables/*.csv` (which are in turn sourced from `attack_logs/`).
Palette and mark conventions follow the dataviz skill's validated default: fixed categorical
hue order (blue/green/magenta/yellow/aqua/orange/violet/red), single hue for single-series
bars, one axis, thin gridlines, direct value labels instead of a hover layer (these are
static images for the thesis, not interactive charts).

| Figure | Attack | What it shows |
|---|---|---|
| `kb_extraction_rate_by_source.png` | KB Extraction | Extraction rate per source x access level (Unauthorized / Authorized / Authorized-Blind). Unauthorized bars at 0% confirm the API-key gate holds. |
| `kb_llm_interface_leakage.png` | KB Extraction | Black-box `/query` leakage: exact-match rate, semantic similarity, edit similarity, chunk recovery rate (CRR), from 20 keyword/paraphrase probes. |
| `mia_production_auc_by_seed.png` | MIA | Production gated-composite AUC-ROC for thesis seeds 0/42/123 (mean 0.567), with the earlier n=5/5 seed-999 quick-check shown in grey for reference only. |
| `mia_ablation_auc_by_seed.png` | MIA | Primary vs. ablation composite AUC-ROC across the 10-seed held-out set — tests whether the production composite generalizes beyond its tuning seeds. |
| `mia_consistency_auc_by_seed.png` | MIA | Repeated-probe consistency-score AUC-ROC across 6 pilot seeds — a weak, inconsistent signal. |
| `sfa_hit_rate_vs_ratio.png` | Selective Forwarding | Hit-rate vs. compromise ratio for `random`/`high_connectivity` strategies (mock sweep) plus the live-network validation point, annotated for its redundant-probe-reroute anomaly. |
| `ddos_hit_rate_vs_ratio.png` | DDoS | Hit-rate vs. attack ratio for `random`/`sequential`/`targeted` node-selection strategies (mock sweep, mean over 3 seeds x 5 waves). |
| `ddos_live_quality_degradation.png` | DDoS | Baseline vs. post-attack F1 at low/mid/high live flood intensity — direct analogue of the poison-ratio-vs-F1 table format. |

Regenerate by re-running the generator script against updated `tables/*.csv` (script lives in
the session scratchpad, not checked into this repo — ask to have it added under `scripts/` if
you want it version-controlled).
