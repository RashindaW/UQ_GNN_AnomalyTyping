#!/usr/bin/env python3
"""Per-anomaly TYPING + EXPLANATION using TARGETED (attacked-sensor) channels.

This is the corrected explanation engine: the earlier global-channel version
collapsed ~all anomalies to "OOD" because global Omega is a DETECTION signal
(high on every attack). The validated typing result lives in the TARGETED
channels (read on the attacked sensor). This engine:

  1. For each attack, reads the uncertainty channels ON ITS TARGETED SENSOR(S)
     (per-node sigma2_ale, per-node U_par, per-node Omega) + the surprise.
  2. Standardizes each targeted channel WITHIN the anomalous population (so the
     decision is a within-anomaly contrast, not a detection level).
  3. Applies the reject-option rule to assign a MECHANISM type
     {SENSOR_SPOOF, PHYSICAL_ATTACK, OOD_NOVEL} + a plain-language explanation.
  4. Validates the assigned mechanism against ground truth (spoof vs physical)
     -> confusion matrix + balanced accuracy.

Uses the real-Omega arrays (arrays_omega.npz, all channels real). GDN seed42.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results/paper/typing")
os.makedirs(OUT, exist_ok=True)


def zscore_pop(x, mask):
    """Standardize x using the mean/std of the masked (anomalous) population."""
    mu, sd = x[mask].mean(), x[mask].std() + 1e-9
    return (x - mu) / sd


def main():
    d = np.load(os.path.join(ROOT, "results/gdn/ref_seed42/arrays_omega.npz"))
    mu = d["test_mu_bar"].astype(np.float64)
    gt = d["test_ground_truth"].astype(np.float64)
    lab = d["test_attack_label"].astype(int)
    sig_ale = d["test_sigma2_ale"].astype(np.float64)          # (T,V)
    upar = d["test_U_par"].astype(np.float64)                   # (T,V)
    omega_pn = d["test_U_dist_maha_pernode"].astype(np.float64) # (T,V)  real per-node Omega
    resid = np.abs(gt - mu)                                     # (T,V)
    T, V = mu.shape

    names = [l.strip() for l in open(os.path.join(ROOT, "data/swat/list.txt")) if l.strip()]
    name2idx = {n: i for i, n in enumerate(names)}

    al = list(csv.DictReader(open(os.path.join(ROOT, "data/swat/attack_list.csv"))))

    # ---- gather targeted-channel readings per attack (anomalous timesteps only) ----
    recs = []
    anom_mask = lab == 1
    for r in al:
        s, e = int(r["start_idx"]), int(r["end_idx"])
        if e <= 0 or s >= T:
            continue
        s = max(0, s); e = min(T, e)
        tgt = [t.strip() for t in r["targets"].replace(";", ",").split(",") if t.strip()]
        tidx = [name2idx[t] for t in tgt if t in name2idx]
        if not tidx:
            continue  # unmappable target (NONE / MV504 not-in-model)
        seg = slice(s, e)
        # targeted channel = mean over the attack's targeted sensors, over its window
        rec = dict(
            aid=int(r["attack_id"]), cat=r["category"], target=",".join(tgt),
            truth=("spoof" if str(r["actual_change"]).lower() == "false"
                   else "physical" if str(r["actual_change"]).lower() == "true" else "none"),
            tg_resid=float(resid[seg][:, tidx].mean()),
            tg_ale=float(sig_ale[seg][:, tidx].mean()),
            tg_epi=float(upar[seg][:, tidx].mean()),
            tg_omega=float(omega_pn[seg][:, tidx].mean()),
            s=s, e=e,
        )
        recs.append(rec)

    # keep attacks with a real mechanism label (spoof / physical)
    recs = [r for r in recs if r["truth"] in ("spoof", "physical")]

    # ---- standardize each targeted channel ACROSS attacks (within-anomaly contrast) ----
    for key in ["tg_resid", "tg_ale", "tg_epi", "tg_omega"]:
        vals = np.array([r[key] for r in recs])
        m, sd = vals.mean(), vals.std() + 1e-9
        for r, v in zip(recs, vals):
            r["z_" + key] = float((v - m) / sd)

    # ---- the reject-option MECHANISM rule (on targeted, standardized channels) ----
    # Validated signal: a sensor SPOOF drives the attacked sensor off-manifold ->
    # high targeted epistemic + standardized residual on that sensor; a PHYSICAL
    # attack keeps the targeted sensor in an agreed region (signal shows elsewhere).
    def assign(r):
        spoofness = r["z_tg_epi"] + r["z_tg_omega"]  # off-manifold-ness of the targeted sensor
        if r["z_tg_omega"] > 1.0 and r["z_tg_epi"] > 0.5:
            return "OOD_NOVEL", ("targeted sensor strongly off-manifold (high targeted Omega + epistemic) "
                                 "-> novel/out-of-distribution event on the attacked sensor; escalate")
        if spoofness > 0:
            return "SENSOR_SPOOF", ("attacked sensor pushed off its learned manifold (high targeted "
                                    "epistemic + distributional uncertainty) while the plant residual "
                                    "stays local -> sensor-spoofing")
        return "PHYSICAL_ATTACK", ("attacked sensor stays in an agreed-upon region (low targeted "
                                   "epistemic/distributional uncertainty); deviation is physical and "
                                   "propagates through the plant -> actuator/physical manipulation")

    for r in recs:
        r["assigned"], r["explanation"] = assign(r)

    # ---- confusion: spoof vs physical (2-class on the mechanism axis) ----
    def mech(a):  # collapse OOD_NOVEL into spoof for the 2-class mechanism eval (both off-manifold)
        return "spoof" if a in ("SENSOR_SPOOF", "OOD_NOVEL") else "physical"
    conf = {("spoof", "spoof"): 0, ("spoof", "physical"): 0,
            ("physical", "spoof"): 0, ("physical", "physical"): 0}
    for r in recs:
        conf[(r["truth"], mech(r["assigned"]))] += 1
    tp = conf[("spoof", "spoof")]; fn = conf[("spoof", "physical")]
    fp = conf[("physical", "spoof")]; tn = conf[("physical", "physical")]
    rec_spoof = tp / max(1, tp + fn); rec_phys = tn / max(1, tn + fp)
    bal_acc = 0.5 * (rec_spoof + rec_phys)
    acc = (tp + tn) / max(1, len(recs))

    # ---- write explanation sheet ----
    md = ["# Per-Anomaly Typing + Explanation (targeted channels, GDN seed42)", "",
          "Each attack is typed by the reject-option rule on the TARGETED-sensor",
          "uncertainty channels (read on the attacked sensor, standardized across",
          "attacks). z_* are standardized targeted-channel levels.", "",
          "| # | cat | target | TRUTH | ASSIGNED | z_resid | z_ale | z_epi | z_Omega |",
          "|---|-----|--------|-------|----------|---------|-------|-------|---------|"]
    for r in sorted(recs, key=lambda z: z["aid"]):
        ok = "OK" if mech(r["assigned"]) == r["truth"] else "x"
        md.append(f"| {r['aid']} | {r['cat']} | {r['target']} | {r['truth']} | "
                  f"{r['assigned']} {ok} | {r['z_tg_resid']:.2f} | {r['z_tg_ale']:.2f} | "
                  f"{r['z_tg_epi']:.2f} | {r['z_tg_omega']:.2f} |")
    md += ["", "## Confusion (mechanism: spoof vs physical)", "",
           "```",
           f"               pred_spoof  pred_phys",
           f"truth_spoof    {tp:>9}  {fn:>9}",
           f"truth_phys     {fp:>9}  {tn:>9}",
           "```",
           f"- spoof recall    = {rec_spoof:.3f}",
           f"- physical recall = {rec_phys:.3f}",
           f"- balanced accuracy = {bal_acc:.3f}  (accuracy {acc:.3f}; n={len(recs)} typed attacks)", ""]
    open(os.path.join(OUT, "explanation_sheet_targeted.md"), "w").write("\n".join(md) + "\n")

    summary = dict(n_typed_attacks=len(recs), confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn),
                   spoof_recall=round(rec_spoof, 3), phys_recall=round(rec_phys, 3),
                   balanced_accuracy=round(bal_acc, 3), accuracy=round(acc, 3),
                   assigned_dist={a: sum(1 for r in recs if r["assigned"] == a)
                                  for a in set(r["assigned"] for r in recs)})
    json.dump(summary, open(os.path.join(OUT, "explanation_summary_targeted.json"), "w"), indent=2)

    print("TYPED ATTACKS:", len(recs))
    print("ASSIGNED TYPE DISTRIBUTION:", summary["assigned_dist"])
    print(f"CONFUSION spoof/phys: tp={tp} fn={fn} fp={fp} tn={tn}")
    print(f"balanced_accuracy={bal_acc:.3f}  accuracy={acc:.3f}  spoof_recall={rec_spoof:.3f}  phys_recall={rec_phys:.3f}")
    print("\nWORKED EXPLANATIONS (3 examples):")
    seen = set()
    for r in sorted(recs, key=lambda z: z["aid"]):
        if r["assigned"] not in seen:
            seen.add(r["assigned"])
            print(f"  #{r['aid']} ({r['truth']}) -> {r['assigned']}: {r['explanation']}")
    print(f"\nwrote {OUT}/explanation_sheet_targeted.md")


if __name__ == "__main__":
    main()
