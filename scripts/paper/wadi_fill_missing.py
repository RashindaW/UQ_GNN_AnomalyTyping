#!/usr/bin/env python3
"""Fill WADI (and one SWaT) pending cells using the EXACT typing-engine channels.

Reuses typing_rules_v1v2.load_combo (robust z, test-nominal fit, max over
sensors) so channel AUROCs, rho, corr(R,Omega) and misses-recovered match the
chapter's tab:uq / tab:uneven convention by construction. Dataset is chosen by
UQ_DATASET (set in env before running). CPU only.

  UQ_DATASET=swat python scripts/paper/wadi_fill_missing.py   # validation
  UQ_DATASET=wadi python scripts/paper/wadi_fill_missing.py   # deliverable
"""
import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
from typing_rules_v1v2 import load_combo
import typing_rules_v1v2 as TR
from analyze_multistage_attacks import robust_z

DS = TR.DATASET
HELD0 = 9445 if DS == "wadi" else 24530
SEEDS = [0, 1, 2, 3, 4, 42]
BACKBONES = ["gdn", "cstgl"] if DS == "wadi" else ["gdn", "topogdn", "cstgl"]


def runs(label):
    out, i, n = [], 0, len(label)
    while i < n:
        if label[i] == 1:
            j = i
            while j < n and label[j] == 1:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def structural_auroc(ctx):
    z = ctx["z"]
    if "test_U_str" not in z.files:
        return None
    nominal = ctx["nominal"]
    us = robust_z(z["test_U_str"].astype(np.float64), nominal).max(1)
    return roc_auc_score(ctx["lab"], us)


def analyse(bb):
    acc = {k: [] for k in ["R", "A", "E", "O", "U_str", "rho", "corrRO", "miss"]}
    for s in SEEDS:
        try:
            ctx = load_combo(bb, "V2", s)
        except Exception:
            continue
        lab = ctx["lab"]
        for ch in ["R", "A", "E", "O"]:
            acc[ch].append(roc_auc_score(lab, ctx[ch]))
        sa = structural_auroc(ctx)
        if sa is not None:
            acc["U_str"].append(sa)
        acc["rho"].append(spearmanr(ctx["O"], ctx["E"]).correlation)
        # held-out corr(R, O)
        H = HELD0
        acc["corrRO"].append(spearmanr(ctx["R"][H:], ctx["O"][H:]).correlation)
        # misses-recovered (held-out, event level): residual-missed attack
        # events in which O crosses its Q0.995 nominal threshold
        R, O, labh = ctx["R"][H:], ctx["O"][H:], lab[H:]
        thrR, thrO = ctx["thr"]["R"], ctx["thr"]["O"]
        ev = runs(labh)
        missed = [(a, b) for (a, b) in ev if not (R[a:b] >= thrR).any()]
        if missed:
            rec = sum(1 for (a, b) in missed if (O[a:b] >= thrO).any())
            acc["miss"].append(rec / len(missed))
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in acc.items() if v}


def fmt(t):
    return f"{t[0]:.3f} +/- {t[1]:.3f} (n={t[2]})"


print(f"===== DATASET = {DS} =====")
for bb in BACKBONES:
    r = analyse(bb)
    print(f"\n[{DS} {bb}]")
    for ch, lbl in [("R", "residual"), ("A", "aleatoric"), ("E", "epistemic"),
                    ("O", "Omega"), ("U_str", "structural"), ("rho", "rho(O,E)"),
                    ("corrRO", "corr(R,O) heldout"), ("miss", "misses-recovered")]:
        if ch in r:
            print(f"   {lbl:20s} {fmt(r[ch])}")
