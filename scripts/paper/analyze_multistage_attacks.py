#!/usr/bin/env python3
"""Per-attack-event analysis across backbones (focus: multi-stage attacks).

For every (backbone, variant, seed) arrays_full.npz and every physical attack in
data/swat/attack_list.csv, computes:
  - M0-style detection: per-sensor |residual| normalised by VAL median/IQR (the GDN
    error-score convention), aggregated as max over sensors; an event is detected if
    the score crosses the tau set at the Q_DET quantile of the NOMINAL (label==0)
    score distribution (fixed nominal FPR, threshold-light and comparable across
    backbones/seeds). Also: latency (s), in-event coverage, peak score percentile.
  - Channel responses: robust z (nominal median/IQR, per sensor / per edge) of
    U_par, sigma2_ale, Omega (maha pernode + mean) and U_str (edge-mean), reported
    as the in-event PEAK of the max-over-sensors series, plus a per-channel
    detection flag against the channel's own nominal Q_DET quantile.
  - Localisation: top-3 sensors by in-event per-sensor peak (residual-z, U_par-z,
    Omega-z) and whether any listed target/impact sensor is among them (hit@3).
Array alignment: attack start/end indices are in GDN-array coordinates; a global
offset per arrays is estimated by maximising in-window label mass over d in
[-120, 120] (handles CST-GL's different T).

Output: results/baseline_v1v2/attack_event_analysis.csv (one row per
backbone,variant,seed,attack) plus a printed multi-stage summary.
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# UQ_DATASET=wadi retargets the WADI campaign (same switch as typing_rules_v1v2)
if os.environ.get("UQ_DATASET", "swat") == "wadi":
    ATT_CSV = os.path.join(ROOT, "data/wadi/attack_list.csv")
    FEAT_TXT = os.path.join(ROOT, "data/wadi/list.txt")
    OUT_CSV = os.path.join(ROOT, "results/baseline_wadi_v2/attack_event_analysis.csv")
    SPECS = [("gdn", "results/baseline_wadi_v2/gdn"),
             ("topogdn", "results/uq_wadi_v2/topogdn"),
             ("cstgl", "results/uq_wadi_v2/cstgl")]
else:
    ATT_CSV = os.path.join(ROOT, "data/swat/attack_list.csv")
    FEAT_TXT = os.path.join(ROOT, "data/swat/list.txt")
    OUT_CSV = os.path.join(ROOT, "results/baseline_v1v2/attack_event_analysis.csv")
    SPECS = [("gdn", "results/baseline_v1v2/gdn"),
             ("topogdn", "results/uq_v1v2/topogdn"),
             ("cstgl", "results/uq_v1v2/cstgl"),
             ("dualstage", "results/uq_v1v2/dualstage")]  # V2-only; V1 skips
# Optional narrowing/redirect, e.g. a dualstage-only run that leaves the
# verified canonical CSV untouched:
#   UQ_ONLY_BACKBONE=dualstage \
#   UQ_EVENT_OUT=results/baseline_v1v2/attack_event_analysis_dualstage.csv
_only = os.environ.get("UQ_ONLY_BACKBONE")
if _only:
    SPECS = [(b, d) for b, d in SPECS if b == _only]
OUT_CSV = os.environ.get("UQ_EVENT_OUT", OUT_CSV)
SEEDS = [0, 1, 2, 3, 4, 42]
Q_DET = 0.995          # nominal quantile -> ~0.5% nominal FPR per series
EPS = 1e-9


def norm_name(x):
    return "".join(c for c in str(x).upper() if c.isalnum())


def load_attacks():
    feats = [l.strip() for l in open(FEAT_TXT) if l.strip()]
    fidx = {norm_name(f): i for i, f in enumerate(feats)}
    atts = []
    for r in csv.DictReader(open(ATT_CSV)):
        if r["category"] in ("", "NONE") or r["no_physical_impact"] == "True":
            continue
        tg = [norm_name(t) for t in r["targets"].split(";") if t]
        im = [norm_name(t) for t in (r["impact_sensors"] or "").split(";") if t]
        atts.append(dict(
            aid=int(r["attack_id"]), cat=r["category"], n_stages=int(r["n_stages"]),
            n_points=int(r["n_points"]), targets=tg, impacts=im,
            s=int(r["start_idx"]), e=int(r["end_idx"]),
        ))
    return atts, feats, fidx


def _robust_scale(ref):
    """Per-column robust scale: the MAX of IQR, central-95-range/3, and std (a max,
    not a cascade). Near-constant columns with rare nominal jumps (binary actuators)
    have ~zero IQR and quantile ranges but a meaningful std; taking the max keeps
    their z bounded (jump z ~ 10) instead of exploding and saturating the nominal
    tail. Truly constant columns fall back to 1."""
    s1 = np.percentile(ref, 75, axis=0) - np.percentile(ref, 25, axis=0)
    s2 = (np.percentile(ref, 97.5, axis=0) - np.percentile(ref, 2.5, axis=0)) / 3.0
    sd = ref.std(axis=0)
    scale = np.maximum(np.maximum(s1, s2), sd)
    return np.where(scale > EPS, scale, 1.0)


def robust_z(x, mask, zcap=1e4):
    """x (T,) or (T,K); robust z (median / cascading scale) fit on mask rows."""
    ref = x[mask]
    med = np.median(ref, axis=0)
    z = (x - med) / _robust_scale(ref)
    return np.clip(z, -zcap, zcap)


def smooth_cols(x, k=5):
    """Moving-average smoothing (window k) along axis 0, per column; mirrors the
    smoothing step of the paper scoring protocol."""
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, x)


def estimate_offset(lab, atts):
    best_d, best_m = 0, -1
    T = len(lab)
    for d in range(-120, 121):
        m = 0
        for a in atts:
            s, e = max(0, a["s"] + d), min(T, a["e"] + d)
            if e > s:
                m += int(lab[s:e].sum())
        if m > best_m:
            best_m, best_d = m, d
    return best_d


def first_cross(series, tau, s):
    idx = np.nonzero(series > tau)[0]
    return int(idx[0] - s) if idx.size else None


def top3(names, peraxis_peak):
    order = np.argsort(-peraxis_peak)[:3]
    return [names[i] for i in order]


def main():
    atts, feats, fidx = load_attacks()
    nfeat = [norm_name(f) for f in feats]
    rows = []
    for bb, d in SPECS:
        for V in ["V1", "V2"]:
            for sd in SEEDS:
                fp = os.path.join(ROOT, d, V, f"seed{sd}", "arrays_full.npz")
                if not os.path.exists(fp):
                    print(f"[skip] {fp}", flush=True)
                    continue
                z = np.load(fp)
                mu = z["test_mu_bar"].astype(np.float64)
                gt = z["test_ground_truth"].astype(np.float64)
                lab = z["test_attack_label"].astype(int)
                T = len(lab)
                nominal = lab == 0
                # Per-sensor error score, self-normalised on TEST-NOMINAL rows (the
                # repo's attack-association convention; robust to val->test drift
                # that otherwise saturates the max-over-sensors series).
                rz = robust_z(np.abs(gt - mu), nominal)   # (T,V)
                rz = smooth_cols(rz, 5)                   # protocol smoothing
                m0 = rz.max(1)
                tau_m0 = float(np.quantile(m0[nominal], Q_DET))
                nom_sorted = np.sort(m0[nominal])

                z_up = robust_z(z["test_U_par"].astype(np.float64), nominal)      # (T,V)
                z_sg = robust_z(z["test_sigma2_ale"].astype(np.float64), nominal)  # (T,V)
                z_om = robust_z(z["test_U_dist_maha_pernode"].astype(np.float64), nominal)  # (T,V)
                up_s = z_up.max(1); sg_s = z_sg.max(1); om_s = z_om.max(1)
                tau_up = float(np.quantile(up_s[nominal], Q_DET))
                tau_sg = float(np.quantile(sg_s[nominal], Q_DET))
                tau_om = float(np.quantile(om_s[nominal], Q_DET))
                if "test_U_str" in z.files:
                    st_s = robust_z(z["test_U_str"].astype(np.float64), nominal).mean(1)  # (T,)
                    tau_st = float(np.quantile(st_s[nominal], Q_DET))
                else:
                    st_s, tau_st = None, None

                doff = estimate_offset(lab, atts)
                for a in atts:
                    s, e = max(0, a["s"] + doff), min(T, a["e"] + doff)
                    if e <= s:
                        continue
                    W = slice(s, e)
                    det = bool((m0[W] > tau_m0).any())
                    lat = first_cross(m0[W], tau_m0, 0)
                    peak = float(m0[W].max())
                    pctl = float(100.0 * np.searchsorted(nom_sorted, peak) / len(nom_sorted))
                    t3_r = top3(nfeat, rz[W].max(0))
                    t3_u = top3(nfeat, z_up[W].max(0))
                    t3_o = top3(nfeat, z_om[W].max(0))
                    tgts = set(a["targets"]) | set(a["impacts"])
                    row = dict(
                        backbone=bb, variant=V, seed=sd, attack_id=a["aid"],
                        category=a["cat"], n_stages=a["n_stages"], n_points=a["n_points"],
                        targets=";".join(a["targets"]), impacts=";".join(a["impacts"]),
                        start=s, end=e, dur_s=e - s, offset=doff,
                        m0_detected=int(det),
                        m0_latency_s=lat if lat is not None else "",
                        m0_coverage=round(float((m0[W] > tau_m0).mean()), 4),
                        m0_peak_pctl=round(pctl, 3),
                        zpeak_upar=round(float(up_s[W].max()), 3),
                        zpeak_sig2=round(float(sg_s[W].max()), 3),
                        zpeak_omega=round(float(om_s[W].max()), 3),
                        zpeak_ustr=(round(float(st_s[W].max()), 3) if st_s is not None else ""),
                        det_upar=int(bool((up_s[W] > tau_up).any())),
                        det_sig2=int(bool((sg_s[W] > tau_sg).any())),
                        det_omega=int(bool((om_s[W] > tau_om).any())),
                        det_ustr=(int(bool((st_s[W] > tau_st).any())) if st_s is not None else ""),
                        top3_resid=";".join(t3_r), top3_upar=";".join(t3_u), top3_omega=";".join(t3_o),
                        hit_resid=int(bool(tgts & set(t3_r))),
                        hit_upar=int(bool(tgts & set(t3_u))),
                        hit_omega=int(bool(tgts & set(t3_o))),
                        uq_catch_miss=int((not det) and bool((om_s[W] > tau_om).any()
                                          or (up_s[W] > tau_up).any()
                                          or (st_s is not None and (st_s[W] > tau_st).any()))),
                    )
                    rows.append(row)
                print(f"[{bb} {V} s{sd}] T={T} offset={doff} tau_m0={tau_m0:.2f} events={sum(1 for r in rows if r['backbone']==bb and r['variant']==V and r['seed']==sd)}", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {OUT_CSV} ({len(rows)} rows)", flush=True)

    # ---- printed summary: multi-stage focus ----
    import collections
    import statistics as stt
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["backbone"], r["variant"], r["attack_id"])].append(r)
    print("\n===== MULTI-STAGE (n_stages>=2) per-event summary (over seeds) =====")
    for (bb, V, aid), rs in sorted(agg.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        if rs[0]["n_stages"] < 2:
            continue
        det = sum(r["m0_detected"] for r in rs)
        lats = [r["m0_latency_s"] for r in rs if r["m0_latency_s"] != ""]
        lat = (stt.median(lats) if lats else None)
        keys = ["zpeak_upar", "zpeak_sig2", "zpeak_omega"]
        zs = {k: stt.mean(float(r[k]) for r in rs) for k in keys}
        zstr = [float(r["zpeak_ustr"]) for r in rs if r["zpeak_ustr"] != ""]
        uq = sum(r["uq_catch_miss"] for r in rs)
        lat_s = ("%.0fs" % lat) if lat is not None else "n/a"
        print(f"A{aid:02d} {rs[0]['category']} [{rs[0]['targets']}] {bb} {V}: "
              f"det {det}/{len(rs)} lat={lat_s} cov={stt.mean(r['m0_coverage'] for r in rs):.2f} "
              f"zU={zs['zpeak_upar']:.1f} zS={zs['zpeak_sig2']:.1f} zO={zs['zpeak_omega']:.1f} "
              f"zStr={(stt.mean(zstr) if zstr else float('nan')):.1f} uq_only={uq}")
    print("\n===== single vs multi-stage M0 detection rate =====")
    for bb, _ in SPECS:
        for V in ["V1", "V2"]:
            ss = [r for r in rows if r["backbone"] == bb and r["variant"] == V and r["n_stages"] == 1]
            ms = [r for r in rows if r["backbone"] == bb and r["variant"] == V and r["n_stages"] >= 2]
            if not ss or not ms:
                continue
            print(f"{bb} {V}: single-stage {sum(r['m0_detected'] for r in ss)}/{len(ss)} "
                  f"({100*sum(r['m0_detected'] for r in ss)/len(ss):.0f}%) | "
                  f"multi-stage {sum(r['m0_detected'] for r in ms)}/{len(ms)} "
                  f"({100*sum(r['m0_detected'] for r in ms)/len(ms):.0f}%)")


if __name__ == "__main__":
    main()
