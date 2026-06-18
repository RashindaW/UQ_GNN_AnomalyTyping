#!/usr/bin/env python3
"""Alarm-level triage: type EVERY alarm episode, including the false alarms.

The event-conditioned typing (typing_rules_v1v2.py) starts from ground-truth
attack windows. This script starts from what an operator actually sees: all
alarm episodes raised by a score at the deployable Q0.995 threshold. P0
change C1 (2026-06-07, docs/PART2_PREREGISTRATION.md): thresholds for BOTH
sources are fit on C-SLICE nominal rows only (leakage-free primary); the
full-stream quantile is recorded in the summary as the legacy arm.
Episodes come from two alarm sources per combo:
  m0  : the residual channel R (smoothed max robust z), the anchored-M0 alarm;
  gbm : the S2 GBM fusion score (cached by explain_gbm_v1v2.py).
P0 change C3: every episode carries three verdict columns (majority-over-
window with the peak-step tie-break, PEAK-score-step, episode-ONSET step as
the fixed-position selection-bias control); the primary `verdict` is the
peak-step one for supervised-score sources and the majority one for m0.
Claims in P2 use the more conservative of peak/onset.
Each episode is classified against the offset-aligned attack windows as
  true      overlaps an attack window;
  transient starts within RECOVERY steps after an attack ends (post-attack
            process recovery, reported separately from clean false alarms);
  false     anywhere else.
Every episode is typed with the same per-step rule engine and majority verdict
as the event analysis, then we ask the operational question: do false alarms
carry a different signature (verdict, channel z, duration, peak margin) than
true alarms, and how much false-alarm load can verdict- or duration-based
suppression remove at what cost in true events?

Outputs: results/typing_v1v2/alarm_triage_episodes.csv,
alarm_triage_summary.json, and a printed report.
Runs in rashindaNew-torch-env, CPU, seconds per combo.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
from scipy.stats import chi2  # noqa: E402
from analyze_multistage_attacks import estimate_offset, norm_name  # noqa: E402
from typing_rules_v1v2 import (C_END, COMBOS, FEAT_TXT, OUTDIR, Q_HIGH,  # noqa: E402
                               RULE_TEXT, VAL_SLICE, c_slice_thresholds,
                               load_attack_table, load_combo, type_step)

GBM_DIR = os.path.join(OUTDIR, "gbm")     # follows UQ_DATASET via typing_rules OUTDIR
PER_DAY = 8640            # 10 s per array point (same cadence on SWaT and WADI)
RECOVERY = 30             # post-attack transient margin: 30 steps = 5 min
MERGE_GAP = 5             # alarm points closer than this merge into one episode


def episodes_from_mask(mask, merge_gap=MERGE_GAP):
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    eps, s, p = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        if i - p <= merge_gap:
            p = int(i)
            continue
        eps.append((s, p + 1))
        s = p = int(i)
    eps.append((s, p + 1))
    return eps


def classify_episode(s, e, wins):
    hits = sorted({aid for aid, (ws, we) in wins if s < we and e > ws})
    if hits:
        return "true", hits
    post = sorted({aid for aid, (ws, we) in wins if we <= s < we + RECOVERY})
    if post:
        return "transient", post
    return "false", []


def rank_auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1.0
    rp = ranks[: pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def main():
    feats = [l.strip() for l in open(FEAT_TXT) if l.strip()]
    nfeat = [norm_name(f) for f in feats]
    atts = load_attack_table()
    os.makedirs(OUTDIR, exist_ok=True)

    rows, summary = [], {}
    for bb, V, seed in COMBOS:
        ctx = load_combo(bb, V, seed)
        doff = estimate_offset(ctx["lab"], atts)
        T = ctx["T"]
        thr_full = ctx["thr"]                  # full-stream nominal (legacy arm)
        thr = c_slice_thresholds(ctx, doff)    # P0 C1: PRIMARY = C-slice nominal
        assert all(np.isfinite(list(thr.values()))), \
            f"C-slice thresholds degenerate for {bb} {V} s{seed}: {thr}"
        days = T / PER_DAY
        held0 = VAL_SLICE[1]
        days_heldout = (T - held0) / PER_DAY
        R, A, E, O = ctx["R"], ctx["A"], ctx["E"], ctx["O"]
        bits_all = np.stack([R > thr["R"], A > thr["A"], E > thr["E"], O > thr["O"]], 1)
        wins = []
        for a in atts:
            ws, we = max(0, a["s"] + doff), min(T, a["e"] + doff)
            if we > ws:
                wins.append((a["aid"], (ws, we)))

        gtag = f"{bb}_{V}_s{seed}"
        cache = np.load(os.path.join(GBM_DIR, f"{gtag}_cache.npz"))
        gscore = cache["score"].astype(np.float64)
        assert len(gscore) == T, (len(gscore), T)
        # P0 C1: gbm threshold also fit on C-slice nominal only (primary);
        # full-stream kept for the summary record.
        cm = ctx["nominal"].copy()
        cm[min(T, C_END + max(0, doff)):] = False
        gthr = float(np.quantile(gscore[cm], Q_HIGH))
        gthr_full = float(np.quantile(gscore[ctx["nominal"]], Q_HIGH))

        for source, score, mask in [
            ("m0", R, R > thr["R"]),
            ("gbm", gscore, gscore > gthr),
        ]:
            nom_score = score[ctx["nominal"]]
            eps = episodes_from_mask(mask)
            combo_rows = []
            for k, (s, e) in enumerate(eps):
                W = slice(s, e)
                truth, hits = classify_episode(s, e, wins)
                bw = bits_all[W]
                step_types = [type_step(*b) for b in bw]
                pk = int(np.argmax(score[W]))
                # P0 C3: three verdict columns. peak = the source score's
                # most-confident step; onset = first episode step (fixed-
                # position selection-bias control); majority with the C4
                # peak-step tie-break.
                v_peak = type_step(*bw[pk])
                v_onset = type_step(*bw[0])
                vals, counts = np.unique(step_types, return_counts=True)
                mx = counts.max()
                tied = [str(v) for v, c in zip(vals, counts) if c == mx]
                v_majority = v_peak if (len(tied) > 1 and v_peak in tied) else tied[0]
                # primary verdict: peak-step for supervised-score sources,
                # majority for the anchored-M0 residual source.
                verdict = v_majority if source == "m0" else v_peak
                conf = float(mx / len(step_types))
                peak_pctl = float(100.0 * (nom_score < float(score[W].max())).mean())
                row = dict(
                    backbone=bb, variant=V, seed=seed, source=source, ep_id=k,
                    start=s, end=e, dur=e - s, truth=truth,
                    attack_ids=";".join(f"A{h:02d}" for h in hits),
                    verdict=verdict, confidence=round(conf, 3),
                    verdict_majority=v_majority, verdict_peak=v_peak,
                    verdict_onset=v_onset,
                    bits_peak="".join("HL"[1 - int(b)] for b in bw[pk]),
                    peak_pctl=round(peak_pctl, 3),
                    zpeak_R=round(float(R[W].max()), 2),
                    zpeak_A=round(float(A[W].max()), 2),
                    zpeak_E=round(float(E[W].max()), 2),
                    zpeak_O=round(float(O[W].max()), 2),
                    margin_R=round(float(R[W].max()) / thr["R"], 3),
                    peak_sensor=nfeat[int(np.argmax(ctx["rzV"][W].max(0)))],
                    in_val_slice=int(VAL_SLICE[0] <= s < VAL_SLICE[1]),
                    in_heldout=int(s >= held0),
                )
                combo_rows.append(row)
            rows.extend(combo_rows)

            tr = [r for r in combo_rows if r["truth"] == "true"]
            fa = [r for r in combo_rows if r["truth"] == "false"]
            ts = [r for r in combo_rows if r["truth"] == "transient"]
            conf_tab = {}
            for r in combo_rows:
                conf_tab.setdefault(r["verdict"], {"true": 0, "transient": 0, "false": 0})
                conf_tab[r["verdict"]][r["truth"]] += 1
            aur = {f: rank_auroc([r[f] for r in tr], [r[f] for r in fa])
                   for f in ("dur", "peak_pctl", "zpeak_R", "zpeak_A", "zpeak_E", "zpeak_O")}
            base_events = sorted({h for r in tr for h in r["attack_ids"].split(";") if h})

            def policy(keep_fn, name):
                kept = [r for r in combo_rows if keep_fn(r)]
                kt = [r for r in kept if r["truth"] == "true"]
                kf = [r for r in kept if r["truth"] == "false"]
                ev = sorted({h for r in kt for h in r["attack_ids"].split(";") if h})
                return dict(policy=name, episodes=len(kept),
                            true_kept=len(kt), true_lost=len(tr) - len(kt),
                            fa_per_day=round(len(kf) / days, 1),
                            fa_suppressed_pct=round(100 * (1 - len(kf) / max(len(fa), 1)), 1),
                            events_flagged=len(ev), events_lost=len(base_events) - len(ev))

            policies = [
                policy(lambda r: True, "P0 all alarms"),
                policy(lambda r: r["verdict"] != "R2_noisy_sensor", "P1 drop R2"),
                policy(lambda r: r["dur"] >= 3 or r["verdict"].startswith("R4"),
                       "P2 drop blips <3 unless R4"),
                policy(lambda r: (r["verdict"] != "R2_noisy_sensor")
                       and (r["dur"] >= 3 or r["verdict"].startswith("R4")),
                       "P3 = P1 + P2"),
                policy(lambda r: r["verdict"].startswith("R4")
                       or r["verdict"] == "R1_high_confidence", "P4 keep R1/R4 only"),
            ]
            fa_sensors = {}
            for r in fa:
                fa_sensors[r["peak_sensor"]] = fa_sensors.get(r["peak_sensor"], 0) + 1
            top_fa = sorted(fa_sensors.items(), key=lambda kv: -kv[1])[:5]

            # P0 C5: held-out FA/day (numerator AND denominator restricted to
            # the held-out region) with an exact Poisson 95% CI. Indicative
            # only (~2.3 days), never a staffing figure.
            k_ho = sum(r["in_heldout"] for r in fa)
            ci_lo = 0.0 if k_ho == 0 else float(chi2.ppf(0.025, 2 * k_ho) / 2 / days_heldout)
            ci_hi = float(chi2.ppf(0.975, 2 * k_ho + 2) / 2 / days_heldout)
            summary[f"{gtag}_{source}"] = dict(
                episodes=len(combo_rows), true=len(tr), transient=len(ts), false=len(fa),
                fa_per_day=round(len(fa) / days, 1),
                fa_heldout=k_ho, days_heldout=round(days_heldout, 2),
                fa_per_day_heldout=round(k_ho / days_heldout, 2),
                fa_per_day_heldout_ci95=[round(ci_lo, 2), round(ci_hi, 2)],
                threshold_primary_cslice=round(thr["R"] if source == "m0" else gthr, 4),
                threshold_fullstream=round(thr_full["R"] if source == "m0" else gthr_full, 4),
                transient_per_day=round(len(ts) / days, 1),
                events_flagged=len(base_events),
                confusion=conf_tab, auroc_true_vs_false=
                {k: (round(v, 3) if np.isfinite(v) else None) for k, v in aur.items()},
                fa_in_val_slice=sum(r["in_val_slice"] for r in fa),
                top_fa_sensors=top_fa, policies=policies,
                dur_median=dict(true=float(np.median([r["dur"] for r in tr])) if tr else None,
                                false=float(np.median([r["dur"] for r in fa])) if fa else None),
                peak_pctl_median=dict(
                    true=round(float(np.median([r["peak_pctl"] for r in tr])), 2) if tr else None,
                    false=round(float(np.median([r["peak_pctl"] for r in fa])), 2) if fa else None),
            )
            s_ = summary[f"{gtag}_{source}"]
            print(f"[{gtag} {source}] episodes={s_['episodes']} true={s_['true']} "
                  f"transient={s_['transient']} false={s_['false']} (FA/day {s_['fa_per_day']}) "
                  f"events={s_['events_flagged']}", flush=True)
            print(f"   confusion: " + "  ".join(
                f"{v}:[T{c['true']}/X{c['transient']}/F{c['false']}]"
                for v, c in sorted(conf_tab.items())), flush=True)
            print(f"   AUROC(true vs false): " + "  ".join(
                f"{k}={v}" for k, v in s_["auroc_true_vs_false"].items()), flush=True)
            print(f"   dur median T/F: {s_['dur_median']['true']}/{s_['dur_median']['false']}"
                  f"   peak_pctl median T/F: {s_['peak_pctl_median']['true']}/"
                  f"{s_['peak_pctl_median']['false']}   top FA sensors: {top_fa}", flush=True)
            for p in policies:
                print(f"   {p['policy']:28s} FA/day {p['fa_per_day']:>6} "
                      f"({p['fa_suppressed_pct']:>5}% cut)  true lost {p['true_lost']:>2} "
                      f"events {p['events_flagged']:>2} (lost {p['events_lost']})", flush=True)

    with open(os.path.join(OUTDIR, "alarm_triage_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]
    with open(os.path.join(OUTDIR, "alarm_triage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUTDIR}/alarm_triage_episodes.csv ({len(rows)} episodes) + summary",
          flush=True)


if __name__ == "__main__":
    main()
