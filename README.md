# UQ-GNN Anomaly Typing — Phase-1 scoped workspace

Self-contained workspace for the prospectus in **`PROSPECTUS.md`**
(*Uncertainty-Augmented GNN Anomaly Detection with Rule-Based Anomaly Typing*).
Copied from `CF_Uncertainity_for_STGNN/` on 2026-05-31, scoped to **only** what
this chapter needs: the 285 GB raw-SWaT dumps, the deferred DualSTGF model, the
Phase-2 counterfactual artifacts, and regenerable competitor checkpoints were
**excluded**. Total ≈ 1.9 GB.

**Standalone for retraining too** — model-ready CSVs + competitor data dirs +
conda-env `pip freeze` specs are bundled. The 285 GB raw dump is *not* needed to
retrain (it feeds preprocessing only, not training). See **`RETRAIN.md`**.

## What's here

| Path | What | Role in the plan |
|---|---|---|
| `models/` | GDN, GDN_GDeltaUQ (anchoring), graph_layer, aleatoric_head, causal_mask | the backbone + UQ machinery (§3.1–3.2) |
| `train_gdeltauq_main.py`, `inference_gdeltauq.py` | train + K-anchor inference → 4 channels | produces μ̄, U_par, U_str, U_dist, σ²_ale |
| `scripts/` | calibrate_gdeltauq, eval_paper_protocol, fusion_sweep_K100_full, pa_k_metric, sweep_postproc_threshold, cf_engine | calibration, fusion (M10), PA%K, Fix-A |
| `competitors/common/` | eval_from_arrays, emit_arrays_generic, emit_arrays_cstgl | **model-agnostic harness** — every method flows through `arrays.npz` (§3.1) |
| `competitors/{CST-GL,GTA,TopoGDN}/` | **code only** (no checkpoints/data) | the 3 transfer backbones (§3.1, RQ2) |
| `data/swat/` | train/test CSV, gdeltauq_split.json, list.txt, attack_targets.json, attack_list.csv | model-ready SWaT + attack-type ground truth (§4, §5.4) |
| `pretrained/swat_gdeltauq_70_sw60_seed{1,2,3,100}/`, `swat_gdeltauq_sw60/` | GDN checkpoints + K=100 calibration bundles | 5-seed GDN (seed42 = `swat_gdeltauq_sw60`) |
| `pretrained/swat_ensemble/calibration_bundle/` | `calibration_set_indices.json` (the C/val split) | the canonical eval split every method uses |
| `results/gdn/seed{1,2,3,100,ref_seed42}/arrays.npz` | GDN 5-seed cached UQ arrays | inputs for fusion + typing + calibration |
| `results/competitors/{cstgl,gta,topogdn}/` | 30 `arrays.npz` + 37 eval JSONs | competitor cached UQ arrays + per-seed metrics |
| `docs/` | RESULTS, RESEARCH_NOTE_UQ_Fusion_Metrics, GDN_SWaT_END_TO_END_METHOD, SESSION_NOTES, inference_pipeline_outline, CLAUDE | provenance |

## The `arrays.npz` contract (what makes the framework model-agnostic)

Every backbone emits one file with: `test_mu_bar (T,V)`, `test_ground_truth`,
`test_attack_label (T,)`, `test_U_par (T,V)`, `test_U_str (T,E)` *(attention
backbones only)*, `test_U_dist (T,)`, `test_sigma2_ale (T,V)`, `val_mu_bar`,
`val_ground_truth`. Everything downstream (fusion, PA%K, calibration, typing)
reads only this file → adding a backbone = emitting this contract.

## Honest status (corrected 2026-05-31)

**Done — the scaffolding (~30% of the chapter):**
- GDN + GBM-fusion (M10) detection, now properly **5-seed K=100** (see below).
- Model-agnostic harness + 3 competitor integrations (cached arrays present).
- Negative Phase-2 CF/hub-dominance result (motivates the detection+typing focus).

**GDN 5-seed K=100 (the corrected headline sanity check):**

| seed | M0 F1 | M10 F1 | ΔF1 | M0 PA%K | M10 PA%K |
|---|---|---|---|---|---|
| 1 | 0.7724 | 0.7419 | −0.030 | 0.8235 | 0.7941 |
| 2 | 0.7678 | 0.8056 | +0.038 | 0.8258 | 0.8316 |
| 3 | 0.7373 | 0.7398 | +0.003 | 0.7922 | 0.7943 |
| 100 | 0.7647 | 0.7568 | −0.008 | 0.8207 | 0.7944 |
| 42 (ref) | 0.8109 | 0.8391 | +0.028 | 0.8633 | 0.8714 |
| **mean±std** | **0.771±0.026** | **0.777±0.044** | **+0.006** | **0.825** | **0.817** |

→ **Detection gain is +0.006 F1, range −0.030…+0.038 — straddles zero; PA%K
slightly *down* (0.825→0.817).** seed42 (the old 0.811/0.839 headline) is the
luckiest seed, not the center. Confirms §5.5/§11: lead with **typing**, report
detection as parity. The M10 (GBM) numbers also can't do faithful typing (§3.4).

**NOT yet built (the actual chapter — three load-bearing gaps found in review):**
1. **The distributional channel Ω is not distinct from epistemic.** In
   `inference_gdeltauq.py` it is literally `U_dist = U_par.mean(dim=-1)` — the
   spatial mean of the epistemic channel, *not* a density/OOD quantity. No
   genuine OOD estimator exists in the repo. The typing table (§3.5) and the
   §3.6 stealthy-attack tiebreaker both require Ω ≠ epistemic → **build a real
   distributional channel first** (input/embedding-space density, kNN, or flow).
2. **The primary likelihood-score fusion (§3.4) is not implemented.** All
   current numbers are the GBM (M10) that the plan explicitly demotes. The
   standardized-residual / Gaussian-NLL / Mahalanobis score that drives *both*
   detection and typing does not exist yet (cheap to add — arithmetic on the
   cached arrays).
3. **The typing confusion-matrix experiment (§5.4, the headline) is not
   started.** Build early as a kill-test.

Plus: β-NLL not implemented (`aleatoric_head.py` is vanilla Gaussian NLL);
calibration pass (milestone 1) not started; U_str exists for GDN/TopoGDN only.

**Caveat on all F1/P/R:** they use a post-proc-aware threshold swept on the test
labels (oracle). **PA%K-AUC is the threshold-robust column.** Honest val-fit
thresholding is a TODO before any F1 goes in the paper.

## Reproduce / next steps (suggested order, per §10)
1. Build a real Ω channel → re-emit arrays with a genuine distributional signal.
2. Implement likelihood-score fusion (primary method) → re-run GDN 5-seed under it.
3. Calibration pass on SWaT (reliability + AUSE) — the floor.
4. Typing confusion matrix vs `data/swat/attack_targets.json` categories.
5. Competitors + WADI + rigor (5-seed CIs, ablations, sweeps, overhead).

Eval any cached arrays through the shared harness:
```bash
python competitors/common/eval_from_arrays.py \
  --arrays results/gdn/ref_seed42/arrays.npz \
  --split  pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json \
  --bundle pretrained/swat_ensemble/calibration_bundle \
  --slide_win 60 --label GDN-ref --out /tmp/check.json
# expect M0 F1≈0.811, M10 F1≈0.839 (the seed42 reference)
```
Run in the `rashindaNew-torch-env` conda env (torch 2.x).
