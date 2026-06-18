#!/usr/bin/env python
"""
Track H -- Step 3: Typing analysis EXTENDED with the REAL Omega channel.

Prospectus 3.5 / 3.6. The prior experiment (typing_separation.py) used a
PLACEHOLDER distributional channel test_U_dist == U_par.mean(1). A real
distributional-uncertainty (Omega) channel now exists in arrays_omega.npz:
  test_U_dist_maha_mean  (T,)   <- validated best variant
  test_U_dist_maha_max   (T,)
  test_U_dist_maha_pernode (T,V) <- for TARGETED views
  test_U_dist_knn        (T,)
  test_U_dist            (T,)   <- the OLD placeholder (== U_par.mean), kept
                                   for the head-to-head comparison.

This script answers four questions, per seed (42,1,2,3,100) and mean:
  Q1 OOD / normal-vs-category: AUROC of Omega (maha_mean) vs placeholder vs
     residual, for each attack category (SSSP/SSMP/MSMP) from normal.
  Q2 STEALTH TIEBREAKER (the key new 3.6 test): restrict to STEALTHY
     anomalies = anomalous timesteps in the bottom quartile of the top-1
     residual detector score; AUROC of {residual, U_par, placeholder U_dist,
     real Omega maha_mean, U_str} for stealth-anomaly-vs-normal; report lift
     of Omega over residual and over epistemic.
  Q3 SPOOF-vs-PHYSICAL: recompute the targeted-channel reject rule
     (spoofness = z(tg_U_par)+z(tg_z_resid)) WITH vs WITHOUT a targeted real
     Omega channel tg_omega = maha_pernode at the attacked sensor(s).
  Q4 REJECT-OPTION TABLE: among detected anomalies, bin by
     {residual hi/lo} x {Omega hi/lo} x {epistemic hi/lo} and show the
     dominant true attack category in each cell.

Reuses scripts/paper/typing_separation.py for channel construction. ASCII only,
numpy + sklearn + scipy. CPU-only, no training.
"""
import os, sys, json
import numpy as np

ROOT = "/mnt/datassd3/rashinda/UQ_GNN_AnomalyTyping"
sys.path.insert(0, os.path.join(ROOT, "scripts/paper"))
import typing_separation as TS  # reuse auroc, cohen_d, build_channels, etc.

LISTTXT = os.path.join(ROOT, "data/swat/list.txt")
TARGETS = os.path.join(ROOT, "data/swat/attack_targets.json")
TYPEFILE = os.path.join(ROOT, "results/paper/typing/type_labels.npz")
OUTDIR = os.path.join(ROOT, "results/paper/typing")

SEEDS = [
    ("seed42", "results/gdn/ref_seed42/arrays_omega.npz"),
    ("seed1",  "results/gdn/seed1/arrays_omega.npz"),
    ("seed2",  "results/gdn/seed2/arrays_omega.npz"),
    ("seed3",  "results/gdn/seed3/arrays_omega.npz"),
    ("seed100","results/gdn/seed100/arrays_omega.npz"),
]

auroc = TS.auroc
cohen_d = TS.cohen_d
_z = TS._z
EPS = TS.EPS


def add_omega_channels(arrays_path, ch, extras):
    """Load the real Omega arrays and attach global + per-node-capable views."""
    d = np.load(arrays_path)
    maha_mean = d["test_U_dist_maha_mean"].astype(np.float64)   # (T,)
    maha_max = d["test_U_dist_maha_max"].astype(np.float64)     # (T,)
    maha_pernode = d["test_U_dist_maha_pernode"].astype(np.float64)  # (T,V)
    knn = d["test_U_dist_knn"].astype(np.float64)               # (T,)
    placeholder = d["test_U_dist"].astype(np.float64)           # (T,) == U_par.mean
    ch["omega_maha_mean"] = maha_mean
    ch["omega_maha_max"] = maha_max
    ch["omega_knn"] = knn
    ch["U_dist_placeholder"] = placeholder
    extras["maha_pernode"] = maha_pernode
    extras["maha_mean"] = maha_mean
    extras["placeholder"] = placeholder
    return ch, extras


def build_tg_omega(attacks, names, type_aid, extras):
    """Targeted real-Omega: mean over the active attack's targeted-sensor
    per-node maha values (NaN where no target maps). Mean (not max) per task."""
    T = extras["T"]
    name2idx = {n: i for i, n in enumerate(names)}
    aid2idx = {}
    for a in attacks:
        aid = int(a.get("attack_id", -1))
        tg = a.get("targets", []) or []
        idxs = [name2idx[t] for t in tg if t in name2idx]
        aid2idx[aid] = idxs
    pernode = extras["maha_pernode"]
    tg_om = np.full(T, np.nan)
    for t in range(T):
        aid = int(type_aid[t])
        if aid < 0:
            continue
        idxs = aid2idx.get(aid, [])
        if not idxs:
            continue
        tg_om[t] = pernode[t, idxs].mean()
    return tg_om


# ---------------------------------------------------------------------------
# Q2 helper: stealth subset = bottom-quartile of top-1 residual among anomalies
# ---------------------------------------------------------------------------
def stealth_mask(resid_top1, anom, q=0.25):
    """Boolean mask of STEALTHY anomalies: anomalous timesteps whose top-1
    residual score is below the q-quantile of the residual over anomalies."""
    r = np.asarray(resid_top1, float)
    anom_vals = r[anom & np.isfinite(r)]
    if anom_vals.size == 0:
        return np.zeros_like(anom, dtype=bool), float("nan")
    thr = np.quantile(anom_vals, q)
    sm = anom & np.isfinite(r) & (r <= thr)
    return sm, float(thr)


def run_seed(tag, arrays_path):
    typ = np.load(TYPEFILE)
    type_cat = typ["type_cat"]
    type_spoof = typ["type_spoof"]
    type_aid = typ["attack_id"]

    names = TS.load_point_names(LISTTXT)
    attacks, _meta = TS.load_attacks(TARGETS)

    # base channels (residual, z_resid, U_par, U_str, sigma_ale) + targeted
    ch, extras = TS.build_channels(arrays_path)
    tg, n_unmapped = TS.build_targeted_channels(attacks, names, type_aid, extras)
    ch.update(tg)
    ch, extras = add_omega_channels(arrays_path, ch, extras)
    ch["tg_omega"] = build_tg_omega(attacks, names, type_aid, extras)

    anom = type_cat != 0
    normal = ~anom
    spoof_mask = type_spoof == 1
    phys_mask = type_spoof == 2
    A = anom & spoof_mask           # sensor spoof
    B = anom & phys_mask            # physical/actuator

    out = {"tag": tag, "arrays": arrays_path,
           "n_anom": int(anom.sum()), "n_normal": int(normal.sum()),
           "n_spoof": int(A.sum()), "n_phys": int(B.sum()),
           "n_unmapped_targets": int(n_unmapped)}

    # ----------------------------------------------------------------
    # Q1: normal vs each category -- Omega vs placeholder vs residual etc.
    # ----------------------------------------------------------------
    inv_cat = {1: "SSSP", 2: "SSMP", 3: "MSMP"}
    q1_channels = ["resid_top1", "z_resid_top1", "U_par_mean", "U_str_mean",
                   "U_dist_placeholder", "omega_maha_mean", "omega_maha_max", "omega_knn"]
    q1_channels = [c for c in q1_channels if c in ch]
    q1 = {}
    for code, nm in inv_cat.items():
        catm = type_cat == code
        if catm.sum() == 0:
            continue
        per = {}
        for c in q1_channels:
            s = ch[c]
            yy = np.concatenate([np.ones(catm.sum()), np.zeros(normal.sum())])
            ss = np.concatenate([s[catm], s[normal]])
            per[c] = auroc(yy, ss)
        q1[nm] = per
    # all-anom-vs-normal too
    per = {}
    for c in q1_channels:
        s = ch[c]
        yy = np.concatenate([np.ones(anom.sum()), np.zeros(normal.sum())])
        ss = np.concatenate([s[anom], s[normal]])
        per[c] = auroc(yy, ss)
    q1["ALL"] = per
    out["q1_normal_vs_cat"] = q1
    out["q1_channels"] = q1_channels

    # ----------------------------------------------------------------
    # Q2: STEALTH TIEBREAKER
    #   stealth = anomalies in bottom quartile of resid_top1 detector.
    #   positives = stealth anomalies; negatives = ALL normal timesteps.
    #   AUROC and Cohen-d for {residual, U_par, placeholder, omega, U_str}.
    # ----------------------------------------------------------------
    sm, thr = stealth_mask(ch["resid_top1"], anom, q=0.25)
    q2_channels = {
        "residual": "resid_top1",
        "epistemic_U_par": "U_par_mean",
        "placeholder_U_dist": "U_dist_placeholder",
        "omega_maha_mean": "omega_maha_mean",
        "U_str": "U_str_mean",
    }
    q2 = {"n_stealth": int(sm.sum()), "resid_q25_thr": thr,
          "n_normal": int(normal.sum())}
    q2_auc = {}
    q2_d = {}
    for label, cname in q2_channels.items():
        if cname not in ch:
            q2_auc[label] = float("nan"); q2_d[label] = float("nan"); continue
        s = ch[cname]
        yy = np.concatenate([np.ones(sm.sum()), np.zeros(normal.sum())])
        ss = np.concatenate([s[sm], s[normal]])
        q2_auc[label] = auroc(yy, ss)
        q2_d[label] = cohen_d(s[sm], s[normal])
    q2["AUROC"] = q2_auc
    q2["cohen_d"] = q2_d
    # lifts
    q2["lift_omega_over_residual"] = (q2_auc["omega_maha_mean"] - q2_auc["residual"])
    q2["lift_omega_over_epistemic"] = (q2_auc["omega_maha_mean"] - q2_auc["epistemic_U_par"])
    q2["lift_omega_over_placeholder"] = (q2_auc["omega_maha_mean"] - q2_auc["placeholder_U_dist"])
    # category composition of the stealth subset (context)
    comp = {}
    for code, nm in {1: "SSSP", 2: "SSMP", 3: "MSMP"}.items():
        comp[nm] = int((sm & (type_cat == code)).sum())
    comp["spoof"] = int((sm & spoof_mask).sum())
    comp["phys"] = int((sm & phys_mask).sum())
    q2["stealth_composition"] = comp
    out["q2_stealth"] = q2

    # ----------------------------------------------------------------
    # Q3: SPOOF vs PHYSICAL reject rule, WITH vs WITHOUT tg_omega.
    #   base   : spoofness = z(tg_U_par) + z(tg_z_resid)
    #   +omega : spoofness = z(tg_U_par) + z(tg_z_resid) + z(tg_omega)
    # ----------------------------------------------------------------
    out["q3_spoof_vs_phys"] = q3_spoof_phys(ch, A, B, anom)

    # also store per-channel spoof-vs-phys AUROC incl tg_omega (context)
    svp = {}
    for c in ["tg_z_resid", "tg_U_par", "tg_omega", "tg_resid", "tg_sigma_ale"]:
        if c not in ch:
            continue
        s = ch[c]
        yy = np.concatenate([np.ones(A.sum()), np.zeros(B.sum())])
        ss = np.concatenate([s[A], s[B]])
        svp[c] = {"AUROC_spoof_gt_phys": auroc(yy, ss),
                  "cohen_d": cohen_d(s[A], s[B])}
    out["q3_channel_auroc"] = svp

    # ----------------------------------------------------------------
    # Q4: REJECT-OPTION TABLE on DETECTED anomalies.
    #   "detected" = anomalies with resid_top1 above its median over anomalies
    #     would exclude stealth; instead detected := anomalies flagged by the
    #     detector at the operating point = resid above the normal 95th pct
    #     (a standard detector threshold). We bin ALL anomalies for coverage
    #     but mark detected. Bin by hi/lo (median split over anomalies) of
    #     residual x Omega x epistemic; report dominant category per cell.
    # ----------------------------------------------------------------
    out["q4_reject_table"] = q4_reject_table(ch, anom, normal, type_cat, type_spoof)

    return out, ch, extras, (anom, normal, A, B, type_cat, type_spoof)


def q3_spoof_phys(ch, A, B, anom):
    pop = anom
    a1 = _z(np.where(pop, ch["tg_U_par"], np.nan))
    a2 = _z(np.where(pop, ch["tg_z_resid"], np.nan))
    a3 = _z(np.where(pop, ch["tg_omega"], np.nan))
    res = {}
    for name, score in [("base", a1 + a2), ("plus_omega", a1 + a2 + a3),
                        ("omega_only", a3)]:
        res[name] = confusion_from_score(score, A, B)
    res["desc_base"] = "spoofness = z(tg_U_par) + z(tg_z_resid)"
    res["desc_plus_omega"] = "spoofness = z(tg_U_par) + z(tg_z_resid) + z(tg_omega)"
    return res


def confusion_from_score(score, A, B):
    pred_spoof = score > 0.0
    AB = A | B
    y_spoof = A[AB]
    p_spoof = pred_spoof[AB]
    tp = int(np.sum(y_spoof & p_spoof))    # spoof->spoof
    fn = int(np.sum(y_spoof & ~p_spoof))   # spoof->phys
    fp = int(np.sum(~y_spoof & p_spoof))   # phys->spoof
    tn = int(np.sum(~y_spoof & ~p_spoof))  # phys->phys
    tot = tp + fn + fp + tn
    acc = float((tp + tn) / tot) if tot else float("nan")
    rec_spoof = tp / (tp + fn) if (tp + fn) else float("nan")
    rec_phys = tn / (tn + fp) if (tn + fp) else float("nan")
    bal = float(np.nanmean([rec_spoof, rec_phys]))
    maj = float(max(tp + fn, fp + tn) / tot) if tot else float("nan")
    return {"spoof_as_spoof": tp, "spoof_as_phys": fn,
            "phys_as_spoof": fp, "phys_as_phys": tn,
            "accuracy": acc, "recall_spoof": rec_spoof,
            "recall_phys": rec_phys, "balanced_accuracy": bal,
            "majority_baseline": maj}


def q4_reject_table(ch, anom, normal, type_cat, type_spoof):
    """Bin detected anomalies into the 8 cells of
    {residual hi/lo} x {Omega hi/lo} x {epistemic hi/lo}, splitting each axis
    at its median over the anomalous population. Report count + dominant true
    category (and dominant spoof axis) per cell."""
    inv_cat = {0: "normal", 1: "SSSP", 2: "SSMP", 3: "MSMP", 4: "NONE"}
    inv_spoof = {0: "normal", 1: "spoof", 2: "phys"}
    resid = ch["resid_top1"]
    omega = ch["omega_maha_mean"]
    epist = ch["U_par_mean"]

    def med_split(x):
        v = x[anom & np.isfinite(x)]
        return np.median(v) if v.size else np.nan
    tr, to, te = med_split(resid), med_split(omega), med_split(epist)

    # detected operating point: residual above normal-population 95th pct
    det_thr = np.quantile(resid[normal & np.isfinite(resid)], 0.95)
    detected = anom & np.isfinite(resid) & (resid >= det_thr)

    cells = {}
    idx = np.where(anom)[0]
    for rbit in (0, 1):
        for obit in (0, 1):
            for ebit in (0, 1):
                m = (anom &
                     ((resid >= tr) == bool(rbit)) &
                     ((omega >= to) == bool(obit)) &
                     ((epist >= te) == bool(ebit)))
                n = int(m.sum())
                ndet = int((m & detected).sum())
                if n == 0:
                    cells["R%d_O%d_E%d" % (rbit, obit, ebit)] = {
                        "n": 0, "n_detected": 0, "dom_cat": "-",
                        "dom_cat_frac": float("nan"), "dom_spoof": "-",
                        "cat_counts": {}}
                    continue
                cc = {}
                for code in (1, 2, 3, 4):
                    c = int((m & (type_cat == code)).sum())
                    if c:
                        cc[inv_cat[code]] = c
                # dominant category
                dom = max(cc.items(), key=lambda kv: kv[1]) if cc else ("-", 0)
                # dominant spoof axis
                sc = {}
                for code in (1, 2):
                    c = int((m & (type_spoof == code)).sum())
                    if c:
                        sc[inv_spoof[code]] = c
                doms = max(sc.items(), key=lambda kv: kv[1]) if sc else ("-", 0)
                cells["R%d_O%d_E%d" % (rbit, obit, ebit)] = {
                    "n": n, "n_detected": ndet,
                    "dom_cat": dom[0], "dom_cat_frac": dom[1] / n,
                    "dom_spoof": doms[0],
                    "spoof_frac": (sc.get("spoof", 0) / n),
                    "cat_counts": cc}
    return {"thr_resid_med": float(tr), "thr_omega_med": float(to),
            "thr_epist_med": float(te), "detect_thr_resid": float(det_thr),
            "n_detected_anom": int(detected.sum()), "n_anom": int(anom.sum()),
            "cells": cells,
            "axis_note": "bit=1 means channel >= its median over anomalies"}


def mean_nan(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    all_out = {}
    for tag, rel in SEEDS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print("[skip] missing", p)
            continue
        out, ch, extras, masks = run_seed(tag, p)
        all_out[tag] = out
        q2 = out["q2_stealth"]
        print("[%s] n_anom=%d n_stealth=%d  Q2 AUROC: resid=%.3f epist=%.3f place=%.3f OMEGA=%.3f Ustr=%.3f | lift(O-resid)=%+.3f lift(O-epi)=%+.3f"
              % (tag, out["n_anom"], q2["n_stealth"],
                 q2["AUROC"]["residual"], q2["AUROC"]["epistemic_U_par"],
                 q2["AUROC"]["placeholder_U_dist"], q2["AUROC"]["omega_maha_mean"],
                 q2["AUROC"]["U_str"], q2["lift_omega_over_residual"],
                 q2["lift_omega_over_epistemic"]))
        b = out["q3_spoof_vs_phys"]["base"]; po = out["q3_spoof_vs_phys"]["plus_omega"]
        print("        Q3 bal_acc base=%.3f  +omega=%.3f  (delta=%+.3f)"
              % (b["balanced_accuracy"], po["balanced_accuracy"],
                 po["balanced_accuracy"] - b["balanced_accuracy"]))

    # ---- aggregate means across seeds ----
    seeds_present = list(all_out.keys())
    agg = {"seeds": seeds_present}

    # Q1 means
    q1m = {}
    for cat in ["SSSP", "SSMP", "MSMP", "ALL"]:
        per = {}
        chans = all_out[seeds_present[0]]["q1_channels"]
        for c in chans:
            per[c] = mean_nan([all_out[s]["q1_normal_vs_cat"].get(cat, {}).get(c) for s in seeds_present])
        q1m[cat] = per
    agg["q1_mean"] = q1m

    # Q2 means
    q2_labels = ["residual", "epistemic_U_par", "placeholder_U_dist", "omega_maha_mean", "U_str"]
    q2m_auc = {L: mean_nan([all_out[s]["q2_stealth"]["AUROC"][L] for s in seeds_present]) for L in q2_labels}
    q2m_d = {L: mean_nan([all_out[s]["q2_stealth"]["cohen_d"][L] for s in seeds_present]) for L in q2_labels}
    agg["q2_mean"] = {
        "AUROC": q2m_auc, "cohen_d": q2m_d,
        "lift_omega_over_residual": mean_nan([all_out[s]["q2_stealth"]["lift_omega_over_residual"] for s in seeds_present]),
        "lift_omega_over_epistemic": mean_nan([all_out[s]["q2_stealth"]["lift_omega_over_epistemic"] for s in seeds_present]),
        "lift_omega_over_placeholder": mean_nan([all_out[s]["q2_stealth"]["lift_omega_over_placeholder"] for s in seeds_present]),
        "omega_beats_resid_n_seeds": int(sum(1 for s in seeds_present if all_out[s]["q2_stealth"]["lift_omega_over_residual"] > 0)),
        "omega_beats_epist_n_seeds": int(sum(1 for s in seeds_present if all_out[s]["q2_stealth"]["lift_omega_over_epistemic"] > 0)),
        "n_seeds": len(seeds_present),
    }

    # Q3 means
    q3m = {}
    for variant in ["base", "plus_omega", "omega_only"]:
        q3m[variant] = {
            "balanced_accuracy": mean_nan([all_out[s]["q3_spoof_vs_phys"][variant]["balanced_accuracy"] for s in seeds_present]),
            "accuracy": mean_nan([all_out[s]["q3_spoof_vs_phys"][variant]["accuracy"] for s in seeds_present]),
            "recall_spoof": mean_nan([all_out[s]["q3_spoof_vs_phys"][variant]["recall_spoof"] for s in seeds_present]),
            "recall_phys": mean_nan([all_out[s]["q3_spoof_vs_phys"][variant]["recall_phys"] for s in seeds_present]),
        }
    q3m["delta_balacc_plus_omega"] = q3m["plus_omega"]["balanced_accuracy"] - q3m["base"]["balanced_accuracy"]
    q3m["plus_omega_helps_n_seeds"] = int(sum(
        1 for s in seeds_present
        if all_out[s]["q3_spoof_vs_phys"]["plus_omega"]["balanced_accuracy"]
        > all_out[s]["q3_spoof_vs_phys"]["base"]["balanced_accuracy"]))
    q3m["n_seeds"] = len(seeds_present)
    agg["q3_mean"] = q3m

    all_out["_aggregate"] = agg

    jpath = os.path.join(OUTDIR, "typing_omega_metrics.json")
    with open(jpath, "w") as f:
        json.dump(all_out, f, indent=2,
                  default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else o)
    print("[saved]", jpath)
    print("__DONE__")
    return all_out


if __name__ == "__main__":
    main()
