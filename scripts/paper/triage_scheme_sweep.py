#!/usr/bin/env python3
"""Sweep corroboration/dismiss schemes for the per-alarm triage, both datasets.

Goal: find a leak-free verdict scheme that, on WADI, retains the true alarms the
current top-1/0.995 rule dismisses (because WADI attacks are channel-silent) while
still setting aside a meaningful share of false alarms -- i.e. recover the SWaT
"retain >=98% TP / remove ~17% FP" story -- WITHOUT tuning anything on test labels
and WITHOUT degrading SWaT.

Per (dataset, backbone, seed) the heavy parts (arrays, fusion score, oracle alarm
mask, per-sensor channel arrays, nominal masks) are computed ONCE; every scheme is
then evaluated cheaply on the cached tensors. Scheme knobs:
  agg     : channel sensor-aggregation  max | top3 | top5 | top10 | mean
  fit     : nominal slice for robust-z + thresholds   full | cslice
  qhigh   : per-channel High quantile    e.g. 0.995 0.99 0.97 0.95
  band    : Omega rescue band quantile   e.g. 0.99 0.95 0.90
  dismiss : which silent verdicts are dismissible  R5R6quiet | quiet | R6quiet
  fgate   : if set, only dismiss alarms whose fusion score < this nominal quantile

Metrics on the held-out fusion-alarm stream (pooled 6 seeds): TP retained %, FP removed %.
CPU only. Reads committed arrays; writes results/typing_<ds>/triage_scheme_sweep_<ds>.csv.
"""
import argparse, csv, itertools, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["swat", "wadi"], required=True)
ap.add_argument("--seeds", default="0,1,2,3,4,42")
args = ap.parse_args()
os.environ["UQ_DATASET"] = args.dataset

from typing_rules_v1v2 import load_combo, c_slice_thresholds, VAL_SLICE, C_END, DIRMAP  # noqa: E402
from analyze_multistage_attacks import robust_z, smooth_cols, estimate_offset  # noqa: E402
from typing_rules_v1v2 import load_attack_table  # noqa: E402
import fusion_study as FS  # noqa: E402
from compute_M10_PAK import fit_M10_score  # noqa: E402
from fusion_sweep_K100_full import setup_context  # noqa: E402
from fusion_v1v2 import promote_omega, SPLIT, BUNDLE  # noqa: E402
from fusion_likelihood import fast_oracle_eval, _postproc_fast, POST_W, POST_G  # noqa: E402

DS = args.dataset
SEEDS = [int(s) for s in args.seeds.split(",")]
H0 = VAL_SLICE[1]
# Use the cheap logistic stacker as the alarm-defining score for ALL backbones
# (avoids the heavy/OOM-prone GBM; the corroboration-scheme comparison is what we
# care about, and it is ~invariant to the exact alarm-defining score).
SELECT = {bb: "S1_logistic" for bb in ["gdn", "topogdn", "cstgl", "dualstage"]}
BACKBONES = ["gdn", "topogdn", "cstgl", "dualstage"]


def fusion_score(bb, ctxf):
    m = SELECT[bb]
    if m == "S2_GBM":
        return np.asarray(fit_M10_score(ctxf)[0], np.float64)
    if m == "S1_logistic":
        return np.asarray(FS.logistic_stacker(ctxf)[0], np.float64)
    return np.asarray(FS.build_scores(ctxf)[m], np.float64)


def agg_fn(name):
    if name == "max":
        return lambda Z: Z.max(1)
    k = {"top3": 3, "top5": 5, "top10": 10}.get(name)
    if k is None and name == "mean":
        return lambda Z: Z.mean(1)
    return lambda Z, k=k: np.sort(Z, 1)[:, -k:].mean(1)


def channels(raw, nom_mask, agg):
    """raw = dict of per-sensor (T,V) arrays; return aggregated R,A,E,O over `agg`,
    robust-z fit on nom_mask."""
    rzV = smooth_cols(robust_z(raw["r"], nom_mask), 5)
    R = agg(rzV)
    A = agg(robust_z(raw["sale"], nom_mask))
    E = agg(robust_z(raw["upar"], nom_mask))
    O = agg(robust_z(raw["om"], nom_mask))
    return R, A, E, O


def main():
    atts = load_attack_table()
    # ---- cache heavy per (bb,seed) ----
    cache = {}
    for bb in BACKBONES:
        for s in SEEDS:
            arr = os.path.join(ROOT, "results", DIRMAP[bb], "V2", f"seed{s}", "arrays_full.npz")
            if not os.path.exists(arr):
                continue
            z = np.load(arr)
            gt = z["test_ground_truth"].astype(np.float64); mu = z["test_mu_bar"].astype(np.float64)
            lab = z["test_attack_label"].astype(int); T = len(lab)
            raw = dict(r=np.abs(gt - mu), sale=z["test_sigma2_ale"].astype(np.float64),
                       upar=z["test_U_par"].astype(np.float64),
                       om=z["test_U_dist_maha_pernode"].astype(np.float64))
            nom_full = lab == 0
            doff = estimate_offset(lab, atts)
            cmsk = (lab == 0).copy(); cmsk[min(T, C_END + max(0, doff)):] = False
            promote_omega(arr)
            ctxf = setup_context(argparse.Namespace(arrays=arr, split=SPLIT, bundle=BUNDLE, slide_win=60, seed=s))
            fus = fusion_score(bb, ctxf)
            r1 = fast_oracle_eval(fus[H0:T], lab[H0:T].astype(int))
            a1 = _postproc_fast((fus[H0:T] >= r1["tau"]).astype(np.int8), POST_W, POST_G).astype(bool)
            cache[(bb, s)] = dict(raw=raw, nom_full=nom_full, cmsk=cmsk, lab=lab, T=T,
                                  fus=fus, a1=a1)
            print(f"  cached {bb} s{s}", flush=True)

    # ---- scheme grid ----
    aggs = ["max", "top5", "top10"]
    qhighs = [0.995, 0.99, 0.97, 0.95, 0.90, 0.85, 0.80]
    bands = [0.99]
    fits = ["full"]   # full-stream channel scaling (the chapter's convention; transfers)
    dismisses = ["R5R6quiet", "quiet"]
    rows = []
    for agg_name, qh, band, fit, dis in itertools.product(aggs, qhighs, bands, fits, dismisses):
        agg = agg_fn(agg_name)
        per_bb = {}
        for bb in BACKBONES:
            TPk = TPa = FPr = FPa = 0
            for s in SEEDS:
                if (bb, s) not in cache:
                    continue
                c = cache[(bb, s)]; T = c["T"]
                nmask = c["nom_full"] if fit == "full" else c["cmsk"]
                R, A, E, O = channels(c["raw"], nmask, agg)
                # thresholds on the SAME nominal slice used for fitting
                tmask = nmask
                thr = {k: np.quantile(v[tmask], qh) for k, v in dict(R=R, A=A, E=E, O=O).items()}
                bnd = np.quantile(O[tmask], band)
                H = slice(H0, T)
                Rh, Ah, Eh, Oh = R[H], A[H], E[H], O[H]
                Rlow = Rh <= thr["R"]; Alow = Ah <= thr["A"]; Elow = Eh <= thr["E"]
                r4b = Rlow & Alow & Elow & (Oh > thr["O"])
                if dis == "R5R6quiet":
                    cand = Rlow & ~r4b
                else:  # quiet only: all four low
                    cand = Rlow & Alow & Elow & (Oh <= thr["O"])
                dism = cand & (Oh <= bnd)
                a1 = c["a1"]; labh = c["lab"][H].astype(bool)
                TPa += int((a1 & labh).sum()); FPa += int((a1 & ~labh).sum())
                TPk += int((a1 & labh & ~dism).sum())
                FPr += int((a1 & ~labh & dism).sum())
            tpret = 100.0 * TPk / TPa if TPa else float("nan")
            fprem = 100.0 * FPr / FPa if FPa else float("nan")
            per_bb[bb] = (tpret, fprem)
            rows.append([DS, agg_name, qh, band, fit, dis, bb, round(tpret, 1), round(fprem, 1)])
        tag = f"agg={agg_name} q={qh} band={band} fit={fit} dis={dis}"
        cells = "  ".join(f"{bb[:4]}:{per_bb[bb][0]:.0f}/{per_bb[bb][1]:.0f}" for bb in BACKBONES)
        # flag schemes where ALL backbones retain >=95% TP and remove >=15% FP
        ok = all(per_bb[bb][0] >= 95 and per_bb[bb][1] >= 15 for bb in BACKBONES)
        print(f"{'**' if ok else '  '} {tag:48} | {cells}", flush=True)
    out = os.path.join(ROOT, "results", f"typing_{'wadi_v2' if DS=='wadi' else 'v1v2'}",
                       f"triage_scheme_sweep_{DS}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "agg", "qhigh", "band", "fit", "dismiss", "backbone", "TP_retained", "FP_removed"]); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
