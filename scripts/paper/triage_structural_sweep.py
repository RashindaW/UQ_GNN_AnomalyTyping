#!/usr/bin/env python3
"""Structural triage schemes beyond per-window channel cuts.

The per-window threshold/aggregation sweep (triage_scheme_sweep.py) found NO leak-free
scheme that recovers the SWaT "retain >=95% TP / remove >=15% FP" asymmetry on WADI.
This tests structural decision rules that target the real WADI gap (attacks detected by
fusion pooling but not corroborated by any single channel window):

  window : dismiss a window iff it is channel-silent (current rule).
  event  : dismiss a whole fusion-alarm EPISODE only if EVERY window in it is
           channel-silent; if any window corroborates, keep the whole episode.
  fgate  : window dismissal, but never dismiss an alarm whose fusion score is in the
           stronger half of the alarm stream (operator keeps confident detections).

All leak-free: channels robust-z + thresholds on the C-slice nominal; the fusion-gate
median is over the alarm stream (no labels). Held-out, 6 seeds pooled, both datasets.
"""
import argparse, csv, itertools, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["swat", "wadi"], required=True)
ap.add_argument("--seeds", default="0,1,2,3,4,42"); args = ap.parse_args()
os.environ["UQ_DATASET"] = args.dataset

from typing_rules_v1v2 import VAL_SLICE, C_END, DIRMAP, load_attack_table  # noqa: E402
from analyze_multistage_attacks import robust_z, smooth_cols, estimate_offset  # noqa: E402
import fusion_study as FS  # noqa: E402
from compute_M10_PAK import fit_M10_score  # noqa: E402
from fusion_sweep_K100_full import setup_context  # noqa: E402
from fusion_v1v2 import promote_omega, SPLIT, BUNDLE  # noqa: E402
from fusion_likelihood import fast_oracle_eval, _postproc_fast, POST_W, POST_G  # noqa: E402

DS = args.dataset; SEEDS = [int(x) for x in args.seeds.split(",")]; H0 = VAL_SLICE[1]
SELECT = ({"gdn": "S2_GBM", "topogdn": "V1_varweighted", "cstgl": "S1_logistic", "dualstage": "S1_logistic"}
          if DS == "wadi" else
          {"gdn": "L1_stdres", "topogdn": "S2_GBM", "cstgl": "S1_logistic", "dualstage": "L1_stdres"})
BB = ["gdn", "topogdn", "cstgl", "dualstage"]


def fscore(bb, ctxf):
    m = SELECT[bb]
    if m == "S2_GBM": return np.asarray(fit_M10_score(ctxf)[0], np.float64)
    if m == "S1_logistic": return np.asarray(FS.logistic_stacker(ctxf)[0], np.float64)
    return np.asarray(FS.build_scores(ctxf)[m], np.float64)


def topkmean(Z, k): return Z.max(1) if k == 1 else np.sort(Z, 1)[:, -k:].mean(1)


def episodes(mask_idx):
    if mask_idx.size == 0: return []
    sp = np.where(np.diff(mask_idx) > 1)[0] + 1
    return np.split(mask_idx, sp)


def main():
    atts = load_attack_table(); cache = {}
    for bb in BB:
        for s in SEEDS:
            arr = os.path.join(ROOT, "results", DIRMAP[bb], "V2", f"seed{s}", "arrays_full.npz")
            if not os.path.exists(arr): continue
            z = np.load(arr); gt = z["test_ground_truth"].astype(np.float64); mu = z["test_mu_bar"].astype(np.float64)
            lab = z["test_attack_label"].astype(int); T = len(lab)
            raw = dict(r=np.abs(gt - mu), sale=z["test_sigma2_ale"].astype(np.float64),
                       upar=z["test_U_par"].astype(np.float64), om=z["test_U_dist_maha_pernode"].astype(np.float64))
            doff = estimate_offset(lab, atts); cmsk = (lab == 0).copy(); cmsk[min(T, C_END + max(0, doff)):] = False
            promote_omega(arr)
            ctxf = setup_context(argparse.Namespace(arrays=arr, split=SPLIT, bundle=BUNDLE, slide_win=60, seed=s))
            fus = fscore(bb, ctxf); r1 = fast_oracle_eval(fus[H0:T], lab[H0:T].astype(int))
            a1 = _postproc_fast((fus[H0:T] >= r1["tau"]).astype(np.int8), POST_W, POST_G).astype(bool)
            cache[(bb, s)] = dict(raw=raw, cmsk=cmsk, lab=lab, T=T, fus=fus[H0:T], a1=a1)
            print(f"  cached {bb} s{s}", flush=True)

    rows = []
    AGG = {"top5": 5, "top10": 10, "mean": 0}; QH = [0.995, 0.99, 0.97]; BAND = [0.95]
    MODES = ["window", "event", "fgate"]
    for agg_name, qh, band, mode in itertools.product(AGG, QH, BAND, MODES):
        k = AGG[agg_name]
        per = {}
        for bb in BB:
            TPa = FPa = TPk = FPr = 0
            for s in SEEDS:
                if (bb, s) not in cache: continue
                c = cache[(bb, s)]; T = c["T"]; nm = c["cmsk"]
                aggf = (lambda Z: Z.mean(1)) if k == 0 else (lambda Z: topkmean(Z, k))
                R = aggf(smooth_cols(robust_z(c["raw"]["r"], nm), 5))
                A = aggf(robust_z(c["raw"]["sale"], nm)); E = aggf(robust_z(c["raw"]["upar"], nm)); O = aggf(robust_z(c["raw"]["om"], nm))
                thr = {n: np.quantile(v[nm], qh) for n, v in dict(R=R, A=A, E=E, O=O).items()}
                bnd = np.quantile(O[nm], band)
                H = slice(H0, T)
                Rh, Ah, Eh, Oh = R[H], A[H], E[H], O[H]
                Rlow = Rh <= thr["R"]; r4b = Rlow & (Ah <= thr["A"]) & (Eh <= thr["E"]) & (Oh > thr["O"])
                cand = Rlow & ~r4b
                silent = cand & (Oh <= bnd)            # window is dismissible
                a1 = c["a1"]; labh = c["lab"][H].astype(bool)
                if mode == "window":
                    dism = silent
                elif mode == "fgate":
                    fus = c["fus"]; med = np.median(fus[a1]) if a1.any() else 0.0
                    dism = silent & (fus < med)
                else:  # event: dismiss an alarm episode only if ALL its windows are silent
                    dism = np.zeros_like(a1)
                    for ep in episodes(np.where(a1)[0]):
                        if silent[ep].all():
                            dism[ep] = True
                TPa += int((a1 & labh).sum()); FPa += int((a1 & ~labh).sum())
                TPk += int((a1 & labh & ~dism).sum()); FPr += int((a1 & ~labh & dism).sum())
            per[bb] = (100.0 * TPk / TPa if TPa else float("nan"), 100.0 * FPr / FPa if FPa else float("nan"))
            rows.append([DS, agg_name, qh, band, mode, bb, round(per[bb][0], 1), round(per[bb][1], 1)])
        ok = all(per[b][0] >= 95 and per[b][1] >= 15 for b in BB)
        cells = "  ".join(f"{b[:4]}:{per[b][0]:.0f}/{per[b][1]:.0f}" for b in BB)
        print(f"{'**' if ok else '  '} agg={agg_name:5} q={qh} band={band} mode={mode:7} | {cells}", flush=True)
    out = os.path.join(ROOT, "results", f"typing_{'wadi_v2' if DS=='wadi' else 'v1v2'}", f"triage_structural_{DS}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "agg", "qhigh", "band", "mode", "backbone", "TP_retained", "FP_removed"]); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
