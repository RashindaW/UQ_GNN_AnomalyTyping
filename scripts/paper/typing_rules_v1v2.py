#!/usr/bin/env python3
"""Rule-based anomaly typing on the V1/V2 cross-backbone arrays (pilot).

Implements the 6-rule triage table on per-timestep channel signatures
(Residual, Aleatoric sigma2, Epistemic U_par, Distributional Omega), with the
nominal-quantile High/Low convention. P0 change C1 (2026-06-07, see
docs/PART2_PREREGISTRATION.md): High = channel > Q0.995 of its own C-SLICE
nominal distribution (rows [0, C_END+offset), label==0) -- the leakage-free
PRIMARY operating point; the legacy full-stream-nominal quantile is retained
as the verdict_fullstream sensitivity arm. Residual-High coincides with the
(C-slice-thresholded) m0 alarm.

Rule table (ordered; precedence resolves multi-High signatures cost-sensitively:
within Residual-High, Omega > Epistemic > Aleatoric > none, so a dismissal
verdict requires all other channels quiet):
  R1 [H,L,L,L] real anomaly, high-confidence flag
  R2 [H,H,L,L] expected noise, likely a noisy sensor, not a real attack
  R3 [H,-,H,L] uncertain in a familiar regime, borderline, escalate
  R4 [H,-,-,H] out-of-distribution, model output suspect, cannot confirm
  R5 [L,H,-,-] expected noise, got noise, not an anomaly
  R6 [L,L,H,-] unsure but predicted correctly, collect more data
  R4b [L,L,L,H] OOD-rescue (no residual alarm but Omega fires)
  (patterns match type_step() exactly: '-' = don't-care, evaluation is
  top-down first-match, so R4b requires A-Low AND E-Low -- an A-High or
  E-High step diverts to R5/R6 first. Notation corrected 2026-06-07; the
  old R4b [L,*,*,H] was over-inclusive and R3/R4 were under-inclusive.)

Per-event protocol: detected events (any Residual-High step in the window) are
typed at the alarm steps, verdict = majority; P0 change C4: ties are resolved
at the peak-residual step's verdict (implemented 2026-06-07; previously the
tie fell to np.unique order). Missed events are typed on the whole window with
the hierarchy Omega > Epistemic > Aleatoric > quiet. Confidence = fraction of
typed steps agreeing with the verdict (a step-agreement/stability measure,
NOT a correctness probability). P0 change C2: R2's response text is
"investigate sensor health, low priority" -- it is not a dismissal verdict.

Mechanism annex (sensor-spoof vs physical, prior-phase rule adapted to event
level): tg_z_resid and tg_U_par are the in-window maxima over the attack's
TARGET sensors (top-3 residual sensors as a proxy when no target maps);
spoofness = z(tg_U_par) + z(tg_z_resid) standardised across the pilot events of
the combo; spoofness > 0 -> sensor spoof. Ground truth: attack_list
actual_change == False means spoof.

Outputs: results/typing_v1v2/typing_events.csv, traces/<combo>_A<id>.json,
typing_summary.json. Runs in the rashindaNew-torch-env. CPU, seconds per combo.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
from analyze_multistage_attacks import robust_z, smooth_cols, estimate_offset, norm_name  # noqa: E402

# Dataset switch: export UQ_DATASET=wadi to retarget every module constant at
# import time (consumers use from-imports, so this must happen here, not in main).
DATASET = os.environ.get("UQ_DATASET", "swat")
if DATASET == "wadi":
    DIRMAP = {"gdn": "baseline_wadi_v2/gdn", "topogdn": "uq_wadi_v2/topogdn", "cstgl": "uq_wadi_v2/cstgl", "dualstage": "uq_wadi_v2/dualstage"}
    ATT_CSV = os.path.join(ROOT, "data/wadi/attack_list.csv")
    FEAT_TXT = os.path.join(ROOT, "data/wadi/list.txt")
    OUTDIR = os.path.join(ROOT, "results/typing_wadi_v2")
    C_END = 5970                 # windowed C-slice end (bundle C_row_range[1] - 60)
    VAL_SLICE = (5970, 9445)     # windowed stacker-train slice (val range - 60)
    PILOT_EVENTS = list(range(1, 16))    # all 15 WADI attacks
    COMBOS = [("gdn", "V2", 42), ("topogdn", "V2", 42), ("cstgl", "V2", 42)]
else:
    DIRMAP = {"gdn": "baseline_v1v2/gdn", "topogdn": "uq_v1v2/topogdn", "cstgl": "uq_v1v2/cstgl", "dualstage": "uq_v1v2/dualstage"}
    ATT_CSV = os.path.join(ROOT, "data/swat/attack_list.csv")
    FEAT_TXT = os.path.join(ROOT, "data/swat/list.txt")
    OUTDIR = os.path.join(ROOT, "results/typing_v1v2")
    C_END = 15593
    VAL_SLICE = (15593, 24530)
    PILOT_EVENTS = [3, 8, 22, 23, 26, 27, 28, 30, 33, 36, 38, 39]
    COMBOS = [("gdn", "V2", 42), ("topogdn", "V2", 42), ("cstgl", "V2", 42), ("cstgl", "V2", 3)]

# P1: env override so BOTH engines (this one and alarm_triage_v1v2, which
# imports COMBOS at module load) can target arbitrary combos without CLI
# surgery: UQ_COMBOS="bb:V:seed,bb:V:seed,..."
if os.environ.get("UQ_COMBOS"):
    COMBOS = [(p.split(":")[0], p.split(":")[1], int(p.split(":")[2]))
              for p in os.environ["UQ_COMBOS"].split(",")]

Q_HIGH = 0.995          # primary High threshold (nominal quantile)
Q_BAND_HI = 0.99        # sensitivity: dual-quantile band
Q_BAND_LO = 0.90
EPS = 1e-9

RULE_TEXT = {
    "R1_high_confidence": "real anomaly the model should catch, high-confidence flag",
    "R2_noisy_sensor": "sensor-health anomaly: investigate sensor, low priority - do not dismiss",
    "R3_borderline": "uncertain in a familiar regime, borderline, escalate",
    "R4_ood_suspect": "out-of-distribution, model output suspect, cannot confirm",
    "R5_benign_noise": "expected noise, got noise, not an anomaly",
    "R6_data_gap": "unsure but predicted correctly, collect more data",
    "R4b_ood_rescue": "no residual alarm but Omega fires, out-of-distribution rescue, escalate",
    "normal_quiet": "all channels quiet",
    "missed_quiet": "missed, all channels quiet",
}


def load_attack_table():
    atts = []
    for r in csv.DictReader(open(ATT_CSV)):
        if r["category"] in ("", "NONE") or r["no_physical_impact"] == "True":
            continue
        atts.append(dict(
            aid=int(r["attack_id"]), cat=r["category"], n_stages=int(r["n_stages"]),
            n_points=int(r["n_points"]),
            targets=[norm_name(t) for t in r["targets"].split(";") if t],
            impacts=[norm_name(t) for t in (r["impact_sensors"] or "").split(";") if t],
            s=int(r["start_idx"]), e=int(r["end_idx"]),
            spoof_gt=(r["actual_change"] == "False"),     # False actual change = sensor spoof
        ))
    return atts


def type_step(rb, ab, eb, ob):
    """Per-timestep rule, ordered (cost-sensitive precedence on multi-High)."""
    if rb:
        if ob:
            return "R4_ood_suspect"
        if eb:
            return "R3_borderline"
        if ab:
            return "R2_noisy_sensor"
        return "R1_high_confidence"
    if ab:
        return "R5_benign_noise"
    if eb:
        return "R6_data_gap"
    if ob:
        return "R4b_ood_rescue"
    return "normal_quiet"


def event_verdict(types_in_scope, peak_type, detected, window_bits):
    """Majority over typed steps for detected events; hierarchy for missed.

    P0 change C4: ties in the majority are resolved at the PEAK-RESIDUAL
    step's verdict (the documented intent), falling back to the
    alphabetically-first tied verdict only if the peak verdict is not among
    the tied set. Returns (verdict, agree, n, tie_used)."""
    if detected:
        vals, counts = np.unique(types_in_scope, return_counts=True)
        mx = counts.max()
        tied = [str(v) for v, c in zip(vals, counts) if c == mx]
        tie_used = 0
        if len(tied) > 1 and peak_type in tied:
            verdict = str(peak_type)
            tie_used = 1
        else:
            verdict = tied[0]
            tie_used = int(len(tied) > 1)
        agree = int(mx)
        return verdict, agree, len(types_in_scope), tie_used
    rb, ab, eb, ob = window_bits          # any-High flags over the whole window
    if ob:
        verdict = "R4b_ood_rescue"
    elif eb:
        verdict = "R6_data_gap"
    elif ab:
        verdict = "R5_benign_noise"
    else:
        verdict = "missed_quiet"
    agree = int(np.sum(np.asarray(types_in_scope) == verdict))
    return verdict, agree, len(types_in_scope), 0


def load_combo(bb, V, seed):
    fp = os.path.join(ROOT, "results", DIRMAP[bb], V, f"seed{seed}", "arrays_full.npz")
    z = np.load(fp)
    mu = z["test_mu_bar"].astype(np.float64)
    gt = z["test_ground_truth"].astype(np.float64)
    lab = z["test_attack_label"].astype(int)
    nominal = lab == 0
    sale = z["test_sigma2_ale"].astype(np.float64)
    upar = z["test_U_par"].astype(np.float64)
    om_pn = z["test_U_dist_maha_pernode"].astype(np.float64)

    rzV = smooth_cols(robust_z(np.abs(gt - mu), nominal), 5)     # (T,V) residual z
    R = rzV.max(1)
    A = robust_z(sale, nominal).max(1)
    E = robust_z(upar, nominal).max(1)
    O = robust_z(om_pn, nominal).max(1)
    O_mean = robust_z(z["test_U_dist_maha_mean"].astype(np.float64)[:, None], nominal)[:, 0]

    # prior-phase sigma-standardised residual for the mechanism annex
    z_sig = np.abs(gt - mu) / np.sqrt(sale + upar + EPS)         # (T,V)

    thr = {k: float(np.quantile(v[nominal], Q_HIGH)) for k, v in
           dict(R=R, A=A, E=E, O=O).items()}
    band_hi = {k: float(np.quantile(v[nominal], Q_BAND_HI)) for k, v in
               dict(R=R, A=A, E=E, O=O).items()}
    band_lo = {k: float(np.quantile(v[nominal], Q_BAND_LO)) for k, v in
               dict(R=R, A=A, E=E, O=O).items()}
    med_anom = {k: float(np.median(v[~nominal])) for k, v in
                dict(R=R, A=A, E=E, O=O).items()}

    # C-slice-nominal threshold (leakage-free deployable equivalent): the first
    # ~15593 (+offset) rows are pre-attack calibration territory in the harness.
    return dict(z=z, lab=lab, nominal=nominal, rzV=rzV, R=R, A=A, E=E, O=O,
                O_mean=O_mean, z_sig=z_sig, upar=upar, thr=thr,
                band_hi=band_hi, band_lo=band_lo, med_anom=med_anom, T=len(lab))


def c_slice_thresholds(ctx, offset, q=Q_HIGH):
    """Quantile thresholds fit on C-SLICE nominal rows only (P0 change C1:
    this is the PRIMARY, leakage-free operating point; the full-stream
    quantile is retained as a sensitivity arm)."""
    c_end = min(ctx["T"], C_END + max(0, offset))
    cm = ctx["nominal"].copy()
    cm[c_end:] = False
    out = {}
    for k in ["R", "A", "E", "O"]:
        ref = ctx[k][cm]
        out[k] = float(np.quantile(ref, q)) if ref.size > 100 else float("nan")
    return out


# P0 change C6: sensor-modality map (frozen; prefix of the normalized name).
def sensor_modality(name):
    n = (name or "").upper()
    for pre, mod in [("DPIT", "diff_pressure"), ("PIT", "pressure"),
                     ("FIT", "flow"), ("LIT", "level"), ("AIT", "analyzer"),
                     ("MV", "valve"), ("UV", "uv"), ("P", "pump")]:
        if n.startswith(pre):
            return mod
    return "other"


def main():
    global COMBOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=",".join(str(x) for x in PILOT_EVENTS))
    ap.add_argument("--all-events", action="store_true")
    ap.add_argument("--combos", default=None,
                    help="override pilot combos: 'bb:V:seed,bb:V:seed,...'")
    args = ap.parse_args()
    if args.combos:
        COMBOS = [(p.split(":")[0], p.split(":")[1], int(p.split(":")[2]))
                  for p in args.combos.split(",")]
    pilot = None if args.all_events else {int(x) for x in args.events.split(",")}

    feats = [l.strip() for l in open(FEAT_TXT) if l.strip()]
    nfeat = [norm_name(f) for f in feats]
    atts = load_attack_table()
    os.makedirs(os.path.join(OUTDIR, "traces"), exist_ok=True)

    rows, summary = [], {}
    for bb, V, seed in COMBOS:
        ctx = load_combo(bb, V, seed)
        doff = estimate_offset(ctx["lab"], atts)
        T = ctx["T"]
        thr_full = ctx["thr"]                  # full-stream nominal (sensitivity arm)
        thr = c_slice_thresholds(ctx, doff)    # P0 C1: PRIMARY = C-slice nominal
        assert all(np.isfinite(list(thr.values()))), \
            f"C-slice thresholds degenerate for {bb} {V} s{seed}: {thr}"
        band_hi = c_slice_thresholds(ctx, doff, Q_BAND_HI)
        band_lo = c_slice_thresholds(ctx, doff, Q_BAND_LO)
        R, A, E, O = ctx["R"], ctx["A"], ctx["E"], ctx["O"]
        bits_all = np.stack([R > thr["R"], A > thr["A"], E > thr["E"], O > thr["O"]], 1)

        combo_rows = []
        for a in atts:
            if pilot is not None and a["aid"] not in pilot:
                continue
            s, e = max(0, a["s"] + doff), min(T, a["e"] + doff)
            if e <= s:
                continue
            W = slice(s, e)
            bw = bits_all[W]
            detected = bool(bw[:, 0].any())
            step_types = [type_step(*b) for b in bw]
            scope_idx = np.nonzero(bw[:, 0])[0] if detected else np.arange(e - s)
            scope_types = [step_types[i] for i in scope_idx]
            window_bits = tuple(bool(bw[:, j].any()) for j in range(4))
            pk = int(np.argmax(R[W]))                       # P0 C4: peak first
            peak_bits = tuple(bool(x) for x in bw[pk])
            peak_type = type_step(*peak_bits)
            verdict, agree, n_typed, tie_used = event_verdict(
                scope_types, peak_type, detected, window_bits)
            conf = agree / max(n_typed, 1)
            peak_sensor = nfeat[int(np.argmax(ctx["rzV"][W].max(0)))]

            # sensitivity: dual-quantile band + median-split verdicts
            def verdict_under(thrs):
                b = np.stack([R[W] > thrs["R"], A[W] > thrs["A"],
                              E[W] > thrs["E"], O[W] > thrs["O"]], 1)
                det = bool(b[:, 0].any())
                st = [type_step(*x) for x in b]
                sc = np.nonzero(b[:, 0])[0] if det else np.arange(e - s)
                wb = tuple(bool(b[:, j].any()) for j in range(4))
                ptype = type_step(*tuple(bool(x) for x in b[pk]))
                v, _, _, _ = event_verdict([st[i] for i in sc], ptype, det, wb)
                return v
            v_band = verdict_under(band_hi)
            in_band = any(band_lo[k] < float(x[W].max()) <= band_hi[k]
                          for k, x in dict(R=R, A=A, E=E, O=O).items())
            v_med = verdict_under(ctx["med_anom"])   # label-using stress arm
            v_full = verdict_under(thr_full)         # legacy full-stream arm

            # mechanism annex: targeted channels (targets, else top-3 residual proxy)
            tcols = [nfeat.index(t) for t in a["targets"] if t in nfeat]
            proxy = False
            if not tcols:
                proxy = True
                tcols = list(np.argsort(-ctx["rzV"][W].max(0))[:3])
            tg_zres = float(ctx["z_sig"][W][:, tcols].max())
            tg_upar = float(ctx["upar"][W][:, tcols].max())

            row = dict(
                backbone=bb, variant=V, seed=seed, attack_id=a["aid"],
                category=a["cat"], n_stages=a["n_stages"],
                targets=";".join(a["targets"]), spoof_gt=int(a["spoof_gt"]),
                start=s, end=e, dur_s=e - s, detected=int(detected),
                verdict=verdict, verdict_text=RULE_TEXT[verdict],
                confidence=round(conf, 3), n_typed=n_typed,
                peak_type=peak_type, peak_sensor=peak_sensor,
                bits_peak="".join("HL"[1 - int(b)] for b in peak_bits),
                zpeak_R=round(float(R[W].max()), 2), zpeak_A=round(float(A[W].max()), 2),
                zpeak_E=round(float(E[W].max()), 2), zpeak_O=round(float(O[W].max()), 2),
                zpeak_Omean=round(float(ctx["O_mean"][W].max()), 2),
                thr_R=round(thr["R"], 2), thr_A=round(thr["A"], 2),
                thr_E=round(thr["E"], 2), thr_O=round(thr["O"], 2),
                verdict_band=v_band, band_flag=int(in_band), verdict_medsplit=v_med,
                tg_z_resid=round(tg_zres, 3), tg_U_par=round(tg_upar, 6),
                tg_proxy=int(proxy),
                verdict_fullstream=v_full,
                fullstream_flip=int(verdict != v_full),
                tie_break_used=tie_used,
                in_heldout=int(s >= VAL_SLICE[1]),
                in_val_slice=int(VAL_SLICE[0] <= s < VAL_SLICE[1]),
                target_modality=sensor_modality(
                    a["targets"][0] if a["targets"] else peak_sensor),
            )
            combo_rows.append(row)

            trace = dict(row, thresholds_primary_cslice=thr,
                         thresholds_fullstream=thr_full,
                         series=dict(
                             R=[round(float(x), 3) for x in R[W]],
                             A=[round(float(x), 3) for x in A[W]],
                             E=[round(float(x), 3) for x in E[W]],
                             O=[round(float(x), 3) for x in O[W]],
                             step_types=step_types))
            with open(os.path.join(OUTDIR, "traces", f"{bb}_{V}_s{seed}_A{a['aid']:02d}.json"), "w") as f:
                json.dump(trace, f, indent=1)

        # mechanism rule: standardise the two targeted channels across the combo's events
        tz = np.array([r["tg_z_resid"] for r in combo_rows], float)
        tu = np.log10(np.array([r["tg_U_par"] for r in combo_rows], float) + EPS)
        def _z(v):
            sd = v.std()
            return (v - v.mean()) / (sd if sd > EPS else 1.0)
        spoof_score = _z(tz) + _z(tu)
        correct = 0
        for r, sc in zip(combo_rows, spoof_score):
            r["spoofness"] = round(float(sc), 3)
            r["mech_pred_spoof"] = int(sc > 0)
            r["mech_correct"] = int((sc > 0) == bool(r["spoof_gt"]))
            correct += r["mech_correct"]
        summary[f"{bb}_{V}_s{seed}"] = dict(
            n_events=len(combo_rows),
            detected=sum(r["detected"] for r in combo_rows),
            verdicts={v: sum(1 for r in combo_rows if r["verdict"] == v)
                      for v in sorted({r["verdict"] for r in combo_rows})},
            mech_acc=round(correct / max(len(combo_rows), 1), 3),
            band_flips=sum(1 for r in combo_rows if r["verdict_band"] != r["verdict"]),
            medsplit_flips=sum(1 for r in combo_rows if r["verdict_medsplit"] != r["verdict"]),
            fullstream_flips=sum(r["fullstream_flip"] for r in combo_rows),
            ties=sum(r["tie_break_used"] for r in combo_rows),
            thresholds_primary_cslice=thr, thresholds_fullstream=thr_full,
            offset=doff,
        )
        rows.extend(combo_rows)
        print(f"[{bb} {V} s{seed}] events={len(combo_rows)} detected="
              f"{sum(r['detected'] for r in combo_rows)} mech_acc={summary[f'{bb}_{V}_s{seed}']['mech_acc']}"
              f" thr={ {k: round(v,1) for k,v in thr.items()} }", flush=True)

    with open(os.path.join(OUTDIR, "typing_events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]
    with open(os.path.join(OUTDIR, "typing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUTDIR}/typing_events.csv ({len(rows)} rows) + traces + summary", flush=True)

    print("\n===== per-event verdicts =====")
    for r in rows:
        print(f"A{r['attack_id']:02d} {r['category']:4s} {r['backbone']:7s} s{r['seed']:<3} "
              f"det={r['detected']} [{r['bits_peak']}] {r['verdict']:18s} conf={r['confidence']:.2f} "
              f"zR={r['zpeak_R']:>7} zA={r['zpeak_A']:>7} zE={r['zpeak_E']:>7} zO={r['zpeak_O']:>7} "
              f"mech={'spoof' if r['mech_pred_spoof'] else 'phys '}({'ok' if r['mech_correct'] else 'X'})")


if __name__ == "__main__":
    main()
