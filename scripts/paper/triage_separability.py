#!/usr/bin/env python3
"""DECISIVE diagnostic: among the held-out FUSION ALARMS, can the uncertainty
channels separate TRUE alarms (TP) from FALSE alarms (FP)? This is exactly the
triage's job. If no channel (or the fusion score, or any combination) separates
TP from FP on WADI, then NO verdict scheme can recover the SWaT story -- it is an
information limit, not a threshold choice.

Reports, per dataset/backbone, AUROC(channel ; TP-vs-FP) over the held-out alarm
stream, pooled 6 seeds, for R, A, E, Omega (robust top-5 aggregation, leak-free
C-slice robust-z) and for the fusion score. Contrast SWaT vs WADI.
CPU only. argv: dataset.
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["swat", "wadi"], required=True)
a = ap.parse_args(); os.environ["UQ_DATASET"] = a.dataset
from typing_rules_v1v2 import VAL_SLICE, C_END, DIRMAP, load_attack_table  # noqa
from analyze_multistage_attacks import robust_z, smooth_cols, estimate_offset  # noqa
import fusion_study as FS  # noqa
from compute_M10_PAK import fit_M10_score  # noqa
from fusion_sweep_K100_full import setup_context  # noqa
from fusion_v1v2 import promote_omega, SPLIT, BUNDLE  # noqa
from fusion_likelihood import fast_oracle_eval, _postproc_fast, POST_W, POST_G  # noqa
from sklearn.metrics import roc_auc_score
DS = a.dataset; H0 = VAL_SLICE[1]; SEEDS = [0, 1, 2, 3, 4, 42]
# Use the cheap logistic stacker for ALL backbones to define the alarm set (avoids the
# heavy/OOM-prone GBM). The channels' TP-vs-FP separability is ~invariant to the exact
# alarm-defining score, so this is fine for the decisive diagnostic.
SEL = {bb: "S1_logistic" for bb in ["gdn", "topogdn", "cstgl", "dualstage"]}
BB = ["gdn", "topogdn", "cstgl", "dualstage"]
def fscore(bb, ctxf):
    m = SEL[bb]
    if m == "S2_GBM": return np.asarray(fit_M10_score(ctxf)[0], np.float64)
    if m == "S1_logistic": return np.asarray(FS.logistic_stacker(ctxf)[0], np.float64)
    return np.asarray(FS.build_scores(ctxf)[m], np.float64)
top5 = lambda Z: np.sort(Z, 1)[:, -5:].mean(1)
def auroc(s, y):
    try: return roc_auc_score(y, s)
    except Exception: return float("nan")
atts = load_attack_table()
print(f"=== {DS}: AUROC(channel ; TRUE-vs-FALSE alarm) among held-out fusion alarms, pooled 6 seeds ===")
print(f"{'backbone':9} | n_alarm TP/FP | R | A | E | Omega | best-chan-max | fusion-score")
for bb in BB:
    S = {k: [] for k in ["R", "A", "E", "O", "best", "fus"]}; Y = []
    for s in SEEDS:
        arr = os.path.join(ROOT, "results", DIRMAP[bb], "V2", f"seed{s}", "arrays_full.npz")
        if not os.path.exists(arr): continue
        z = np.load(arr); gt = z["test_ground_truth"].astype(float); mu = z["test_mu_bar"].astype(float)
        lab = z["test_attack_label"].astype(int); T = len(lab)
        doff = estimate_offset(lab, atts)
        cm = (lab == 0).copy(); cm[min(T, C_END + max(0, doff)):] = False
        R = top5(smooth_cols(robust_z(np.abs(gt - mu), cm), 5))
        A = top5(robust_z(z["test_sigma2_ale"].astype(float), cm))
        E = top5(robust_z(z["test_U_par"].astype(float), cm))
        O = top5(robust_z(z["test_U_dist_maha_pernode"].astype(float), cm))
        promote_omega(arr)
        ctxf = setup_context(argparse.Namespace(arrays=arr, split=SPLIT, bundle=BUNDLE, slide_win=60, seed=s))
        fus = fscore(bb, ctxf)
        r1 = fast_oracle_eval(fus[H0:T], lab[H0:T].astype(int))
        a1 = _postproc_fast((fus[H0:T] >= r1["tau"]).astype(np.int8), POST_W, POST_G).astype(bool)
        H = slice(H0, T); labh = lab[H]
        m = a1  # alarms only
        best = np.maximum.reduce([R[H], A[H], E[H], O[H]])
        S["R"] += list(R[H][m]); S["A"] += list(A[H][m]); S["E"] += list(E[H][m])
        S["O"] += list(O[H][m]); S["best"] += list(best[m]); S["fus"] += list(fus[H][m])
        Y += list(labh[m])
        print(f"  [{bb} s{s}] done", flush=True)
    Y = np.array(Y); ntp = int(Y.sum()); nfp = int((Y == 0).sum())
    au = {k: auroc(np.array(S[k]), Y) for k in S}
    print(f"{bb:9} | {ntp}/{nfp} | {au['R']:.2f} | {au['A']:.2f} | {au['E']:.2f} | {au['O']:.2f} | "
          f"{au['best']:.2f} | {au['fus']:.2f}", flush=True)
