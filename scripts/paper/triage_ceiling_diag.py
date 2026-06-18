#!/usr/bin/env python3
"""Threshold-free CEILING for the per-alarm typing story, SWaT vs WADI.

The typing rule is, at bottom, a classifier of true-vs-false among the detector's
ALARMS, using the channel signature (R, A, E, Omega) on each alarm window. Its best
possible clean-story quality is bounded by how well those channels SEPARATE true
alarms (TP) from false alarms (FP) -- a threshold-free quantity (AUROC). If that
AUROC is high (~0.85) a good rule must exist and we keep searching aggregations; if
it is ~0.6 no per-alarm rule can reproduce the SWaT story and that is the honest
finding.

Computed leak-free: channels robust-z on C-slice nominal, several aggregations; the
alarm set is the held-out best-fusion alarm stream (oracle threshold, as the chapter).
Per backbone we report, among held-out alarms, AUROC(channel score, TP-vs-FP) for the
best single channel and a logistic combo of all four (fit per-seed on that seed's own
alarms -- optimistic in-sample ceiling, which is the point: it is an upper bound).
"""
import argparse, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["swat", "wadi"], required=True)
args = ap.parse_args(); os.environ["UQ_DATASET"] = args.dataset

from typing_rules_v1v2 import VAL_SLICE, C_END, DIRMAP, load_attack_table  # noqa: E402
from analyze_multistage_attacks import robust_z, smooth_cols, estimate_offset  # noqa: E402
import fusion_study as FS  # noqa: E402
from compute_M10_PAK import fit_M10_score  # noqa: E402
from fusion_sweep_K100_full import setup_context  # noqa: E402
from fusion_v1v2 import promote_omega, SPLIT, BUNDLE  # noqa: E402
from fusion_likelihood import fast_oracle_eval, _postproc_fast, POST_W, POST_G  # noqa: E402
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

DS = args.dataset; SEEDS = [0, 1, 2, 3, 4, 42]; H0 = VAL_SLICE[1]
SELECT = ({"gdn": "S2_GBM", "topogdn": "V1_varweighted", "cstgl": "S1_logistic", "dualstage": "S1_logistic"}
          if DS == "wadi" else
          {"gdn": "L1_stdres", "topogdn": "S2_GBM", "cstgl": "S1_logistic", "dualstage": "L1_stdres"})
BB = ["gdn", "topogdn", "cstgl", "dualstage"]


def fscore(bb, ctxf):
    m = SELECT[bb]
    if m == "S2_GBM": return np.asarray(fit_M10_score(ctxf)[0], np.float64)
    if m == "S1_logistic": return np.asarray(FS.logistic_stacker(ctxf)[0], np.float64)
    return np.asarray(FS.build_scores(ctxf)[m], np.float64)


def auroc(s, y):
    try: return roc_auc_score(y, s)
    except Exception: return float("nan")


def main():
    atts = load_attack_table()
    print(f"==== {DS}: among held-out ALARMS, channel separation of TP vs FP (AUROC; higher=cleaner story) ====")
    print(f"{'backbone':9} | nA(TP/FP) | bestR-agg | A | E | Omega | logistic(all4) | base story (max/0.995)")
    for bb in BB:
        feats_all, y_all = [], []
        bestR = {"max": [], "top5": [], "mean": []}
        for s in SEEDS:
            arr = os.path.join(ROOT, "results", DIRMAP[bb], "V2", f"seed{s}", "arrays_full.npz")
            if not os.path.exists(arr): continue
            z = np.load(arr); gt = z["test_ground_truth"].astype(np.float64); mu = z["test_mu_bar"].astype(np.float64)
            lab = z["test_attack_label"].astype(int); T = len(lab)
            r = np.abs(gt - mu); sale = z["test_sigma2_ale"].astype(np.float64)
            upar = z["test_U_par"].astype(np.float64); om = z["test_U_dist_maha_pernode"].astype(np.float64)
            doff = estimate_offset(lab, atts); nm = (lab == 0).copy(); nm[min(T, C_END + max(0, doff)):] = False
            promote_omega(arr)
            ctxf = setup_context(argparse.Namespace(arrays=arr, split=SPLIT, bundle=BUNDLE, slide_win=60, seed=s))
            fus = fscore(bb, ctxf); r1 = fast_oracle_eval(fus[H0:T], lab[H0:T].astype(int))
            a1 = _postproc_fast((fus[H0:T] >= r1["tau"]).astype(np.int8), POST_W, POST_G).astype(bool)
            def aggset(x):
                rz = robust_z(x, nm)
                return dict(max=rz.max(1), top5=np.sort(rz, 1)[:, -5:].mean(1), mean=rz.mean(1))
            Rz = aggset(r); Az = aggset(sale); Ez = aggset(upar); Oz = aggset(om)
            H = slice(H0, T); lh = lab[H].astype(bool)
            al = a1; idx = np.where(al)[0]
            if idx.size < 20 or lh[idx].sum() == 0 or (~lh[idx]).sum() == 0:
                continue
            y = lh[idx].astype(int)
            for agg in bestR:
                bestR[agg].append(auroc(Rz[agg][H][idx], y))
            feats = np.column_stack([smooth_cols(robust_z(r, nm), 5).max(1)[H][idx],
                                     Az["max"][H][idx], Ez["max"][H][idx], Oz["max"][H][idx]])
            feats_all.append((feats, y))
        if not feats_all:
            print(f"{bb:9} | (insufficient alarms)"); continue
        # aggregate AUROCs (mean over seeds)
        mR = {k: np.nanmean(v) for k, v in bestR.items()}
        # per-seed logistic in-sample ceiling
        logs = []
        Amax, Emax, Omax = [], [], []
        for feats, y in feats_all:
            if len(np.unique(y)) < 2: continue
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(feats, y)
            logs.append(auroc(clf.decision_function(feats), y))
            Amax.append(auroc(feats[:, 1], y)); Emax.append(auroc(feats[:, 2], y)); Omax.append(auroc(feats[:, 3], y))
        nTP = sum(int(y.sum()) for _, y in feats_all); nFP = sum(int((1 - y).sum()) for _, y in feats_all)
        print(f"{bb:9} | {nTP}/{nFP} | R:max{mR['max']:.2f} t5{mR['top5']:.2f} mn{mR['mean']:.2f} | "
              f"A{np.nanmean(Amax):.2f} | E{np.nanmean(Emax):.2f} | O{np.nanmean(Omax):.2f} | "
              f"LOGIT(insample)={np.nanmean(logs):.2f}")


if __name__ == "__main__":
    main()
