#!/usr/bin/env python3
"""P2: the pre-registered Part-2 hypothesis battery (H1-H5).

Implements docs/PART2_PREREGISTRATION.md (frozen, sha256 88dc96e8...) plus
the 17 P1 review-gate flags. Reads ONLY the P1 artifacts + canonical ground
truth; writes results/thesis_part2/PART2_TYPING_STATS.md and a tidy
event-level CSV. Deterministic: rng seed 20260607.

Conventions honored (gate flags): dualstage seeds {3,4} excluded from every
DualSTGF-specific statement (C7) with with/without splits on aleatoric-
driven analyses; claims use the conservative of peak/onset episode verdicts;
FA/day always with Poisson CI + 'indicative only'; R4 escalation gate;
combo-cluster bootstrap for episode rates (events are the inference unit;
episodes are descriptors); confirmatory view = the 20 non-pilot combos
reported alongside the all-24 view.
"""
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results/thesis_part2")
RNG = np.random.default_rng(20260607)
N_BOOT = 10000

THIRD_BUCKET = {3, 6, 7, 8, 10, 11, 32, 33, 40}      # prereg S3.1, frozen
# Amendment A1 (2026-06-07): DualSTGF removed from thesis scope; the live
# prereg is the post-A1 file (manifest sha 'sha256_after_amendment'). The C7
# screen + dualstage flags are retired as moot; screen numbers live in
# results/thesis_part2/P1_RUN_REPORT.txt.
BACKBONES = ["gdn", "topogdn", "cstgl"]
PILOTS = {("gdn", 42), ("topogdn", 42), ("cstgl", 42), ("cstgl", 3)}
CORROB = {"R1_high_confidence", "R3_borderline", "R4_ood_suspect"}
SUPER = {"R1_high_confidence": "corroborated", "R3_borderline": "corroborated",
         "R4_ood_suspect": "corroborated", "R2_noisy_sensor": "low_priority",
         "R5_benign_noise": "low_priority", "R4b_ood_rescue": "ood",
         "R6_data_gap": "ood", "normal_quiet": "quiet", "missed_quiet": "quiet"}
HELD0 = 24530


def f3(x):
    return "nan" if not np.isfinite(x) else f"{x:.3f}"


def exact_binom(k, n):
    """One-sided P(X >= k | n, 0.5)."""
    if n == 0:
        return float("nan")
    return float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)


def holm(ps):
    order = np.argsort(ps)
    out = np.empty(len(ps))
    mx = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (len(ps) - rank) * ps[i])
        mx = max(mx, adj)
        out[i] = mx
    return out


def cliffs(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return float((gt - lt) / (len(x) * len(y)))


def combo_boot_rate(df, num_fn, den_fn, n_boot=N_BOOT):
    """Cluster bootstrap over combos: resample combos, recompute pooled rate."""
    combos = list(df.groupby(["backbone", "seed"]).groups)
    if not combos:
        return float("nan"), (float("nan"), float("nan"))
    groups = {c: g for c, g in df.groupby(["backbone", "seed"])}
    point_n, point_d = num_fn(df), den_fn(df)
    point = point_n / point_d if point_d else float("nan")
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(len(combos), len(combos), replace=True)
        n = d = 0
        for i in pick:
            g = groups[combos[i]]
            n += num_fn(g)
            d += den_fn(g)
        vals.append(n / d if d else np.nan)
    vals = np.array(vals, float)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, (float(lo), float(hi))


def event_boot_mean(values, n_boot=N_BOOT):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), (float("nan"), float("nan"))
    bs = RNG.choice(v, (n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def kripp_alpha_nominal(unit_ratings):
    """Krippendorff's alpha, nominal, units = list of lists of category labels."""
    pairs_dis = pairs_tot = 0
    allvals = []
    for u in unit_ratings:
        u = [x for x in u if x is not None]
        allvals.extend(u)
        m = len(u)
        if m < 2:
            continue
        for a, b in combinations(u, 2):
            pairs_tot += 1
            pairs_dis += int(a != b)
    if pairs_tot == 0:
        return float("nan"), float("nan")
    Do = pairs_dis / pairs_tot
    vals, counts = np.unique(allvals, return_counts=True)
    n = counts.sum()
    De = 1.0 - float((counts * (counts - 1)).sum()) / (n * (n - 1)) if n > 1 else np.nan
    alpha = 1.0 - Do / De if De and De > 0 else float("nan")
    return float(alpha), float(1.0 - Do)


def modal(series, tie_value=None):
    vals, counts = np.unique(list(series), return_counts=True)
    mx = counts.max()
    tied = [v for v, c in zip(vals, counts) if c == mx]
    if len(tied) == 1:
        return tied[0], False
    return (tie_value if tie_value is not None else sorted(tied)[0]), True


def main():
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_csv(f"{ROOT}/results/typing_v1v2/typing_events.csv")
    e = pd.read_csv(f"{ROOT}/results/typing_v1v2/alarm_triage_episodes.csv")
    tsum = json.load(open(f"{ROOT}/results/typing_v1v2/typing_summary.json"))
    asum = json.load(open(f"{ROOT}/results/typing_v1v2/alarm_triage_summary.json"))
    canon = {a["attack_id"]: a for a in
             json.load(open(f"{ROOT}/data/swat/attack_gt_canonical.json"))["attacks"]}
    alist = pd.read_csv(f"{ROOT}/data/swat/attack_list.csv")
    npts = dict(zip(alist.attack_id.astype(int), alist.n_points.astype(int)))
    assert len(t) == 846 and len(e) == 4593, (len(t), len(e))
    # A1 scope filter: three published backbones only.
    t = t[t.backbone.isin(BACKBONES)].copy()
    e = e[e.backbone.isin(BACKBONES)].copy()
    assert t.groupby(["backbone", "seed"]).ngroups == 18, "expect 18 combos"
    assert e.groupby(["backbone", "seed", "source"]).ngroups == 36
    assert len(t) == 636, len(t)

    t["combo"] = list(zip(t.backbone, t.seed))
    e["combo"] = list(zip(e.backbone, e.seed))
    t["is_pilot"] = t.combo.isin(PILOTS)
    e["is_pilot"] = e.combo.isin(PILOTS)
    # mechanism buckets (prereg S3.1)
    t["deception"] = t.spoof_gt.astype(int)
    t["bucket"] = np.where(t.deception == 0, "physical",
                   np.where(t.attack_id.isin(THIRD_BUCKET),
                            "deception_with_effect", "deception_only"))

    L = []
    A = L.append
    A("# PART2_TYPING_STATS: the pre-registered hypothesis battery")
    A("")
    A("Generated by scripts/paper/p2_battery.py from the P1 artifacts,")
    A("restricted to the post-Amendment-A1 thesis scope: 3 published backbones")
    A("(GDN, TopoGDN, CST-GL) x 6 seeds = 18 combos, 636 event-verdicts.")
    A("Prereg: docs/PART2_PREREGISTRATION.md as amended (manifest")
    A("results/thesis_part2/prereg_manifest.json; live sha = the")
    A("sha256_after_amendment entry). rng 20260607; N_boot 10,000.")
    A("Views: all-18 combos AND the confirmatory 14 non-pilot combos.")
    A("DualSTGF artifacts remain on disk, unreported (A1, timing disclosed);")
    A("its C7 screen numbers are in results/thesis_part2/P1_RUN_REPORT.txt.")
    A("")

    # ================= H1 =================
    A("## H1. Mechanism typing (Layer A, multi-label; events are the unit)")
    A("")
    A("Task: predict the DECEPTION flag (official actual_change == False; 18/18")
    A("split among eligible events). Channel rule = the event-level target-")
    A("agnostic spoofness sign (engine annex). Baselines: majority (0.50),")
    A("proportional chance (0.50), residual-only sign, cardinality")
    A("(n_points==1 -> deception). Aggregation: per-event MODAL prediction")
    A("across the view's combos; exact ties resolve to PHYSICAL (conservative")
    A("against H1). Primary test: one-sided exact binomial on discordant")
    A("events vs each baseline, Holm over the 4 comparisons; plus a 10,000-")
    A("draw label-permutation test. Pre-stated MDE: ~0.30 absolute accuracy")
    A("gap at 80% power; smaller true effects are INCONCLUSIVE, not negative.")
    A("Baseline-construction note (verifier fix 7): tg_z_resid is a sign-less")
    A("magnitude, so the residual-only baseline is a per-combo relative-")
    A("residual (above-mean) threshold on the standardized scale, not a")
    A("literal sign rule -- apples-to-apples with the channel rule's scale.")
    A("")
    for view, vname in [(t, "all-18"), (t[~t.is_pilot], "confirmatory-14")]:
        # residual-only baseline: per-combo z of tg_z_resid > 0
        vr = view.copy()
        zres_pred = []
        for c, g in vr.groupby("combo"):
            v = g.tg_z_resid.astype(float).values
            sd = v.std() if v.std() > 1e-9 else 1.0
            zres_pred.extend(zip(g.index, ((v - v.mean()) / sd > 0).astype(int)))
        vr.loc[[i for i, _ in zres_pred], "resid_pred"] = [p for _, p in zres_pred]
        per_event, gt_map = {}, {}
        for aid, g in vr.groupby("attack_id"):
            gt_map[aid] = int(g.deception.iloc[0])
            ch, _ = modal(g.mech_pred_spoof, tie_value=0)
            ro, _ = modal(g.resid_pred.astype(int), tie_value=0)
            card = int(npts[aid] == 1)
            per_event[aid] = (int(ch), int(ro), card)
        aids = sorted(per_event)
        y = np.array([gt_map[a] for a in aids])
        ch = np.array([per_event[a][0] for a in aids])
        ro = np.array([per_event[a][1] for a in aids])
        cd = np.array([per_event[a][2] for a in aids])
        mj = np.zeros_like(y)  # majority: physical (either class; 0.5 anyway)
        accs = {"channel_rule": (ch == y).mean(), "residual_only": (ro == y).mean(),
                "cardinality": (cd == y).mean(), "majority": (mj == y).mean()}
        ps = []
        for base in (mj, ro, cd):
            disc = ch != base
            k = int(((ch == y) & disc).sum())
            n = int(disc.sum())
            ps.append(exact_binom(k, n))
        ps.append(ps[0])  # proportional chance == majority at 18/18; same test
        hp = holm(np.array(ps[:4]))
        # permutation
        perm = RNG.permuted(np.tile(y, (10000, 1)), axis=1)
        perm_acc = (perm == ch).mean(axis=1)
        p_perm = float((perm_acc >= (ch == y).mean()).mean())
        A(f"### H1 view: {vname} (n={len(aids)} events)")
        A("")
        A("| rule | accuracy | exact-binomial p vs channel (Holm) |")
        A("|---|---|---|")
        A(f"| channel rule | **{f3(accs['channel_rule'])}** | permutation p={f3(p_perm)} |")
        A(f"| majority / prop-chance | {f3(accs['majority'])} | {f3(hp[0])} |")
        A(f"| residual-only | {f3(accs['residual_only'])} | {f3(hp[1])} |")
        A(f"| cardinality (n_points==1) | {f3(accs['cardinality'])} | {f3(hp[2])} |")
        A("")
        bk = vr.groupby("bucket").apply(
            lambda g: f"{(g.mech_pred_spoof == g.deception).mean():.3f} (n={g.attack_id.nunique()} ev)",
            include_groups=False)
        A("3-bucket channel-rule agreement (combo-level rows): " +
          "; ".join(f"{k}: {v}" for k, v in bk.items()))
        A("")
    A("CLAIM GATE (pre-committed): the phrase 'channels classify attack")
    A("mechanism' requires beating ALL baselines at Holm p<0.05 in the")
    A("confirmatory view. See the H1 verdict in the Conclusions section.")
    A("")

    # ---- Oracle ablation (per-timestep, GT targets) ----
    A("### H1 oracle ablation (upper bound only, NEVER the headline)")
    A("")
    A("Per-timestep rule of typing_separation.py: spoofness(t) = z(tg_U_par)+")
    A("z(tg_z_resid) on GROUND-TRUTH target sensors over labeled attack steps,")
    A("threshold 0. Oracle by construction (reads the attack's true targets).")
    A("")
    sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))
    os.environ.pop("UQ_COMBOS", None)
    from typing_rules_v1v2 import load_combo, load_attack_table  # noqa: E402
    from analyze_multistage_attacks import estimate_offset, norm_name  # noqa: E402
    atts = load_attack_table()
    feats = [l.strip() for l in open(f"{ROOT}/data/swat/list.txt") if l.strip()]
    nf = [norm_name(x) for x in feats]
    combos24 = sorted(set(zip(t.backbone, t.seed)), key=str)
    step_accs, ev_accs = [], []
    for bb, seed in combos24:
        ctx = load_combo(bb, "V2", int(seed))
        doff = estimate_offset(ctx["lab"], atts)
        rows = []
        for a in atts:
            tc = [nf.index(x) for x in a["targets"] if x in nf]
            if not tc:
                continue
            s_, e_ = max(0, a["s"] + doff), min(ctx["T"], a["e"] + doff)
            if e_ <= s_:
                continue
            zr = ctx["z_sig"][s_:e_][:, tc].max(1)
            up = ctx["upar"][s_:e_][:, tc].max(1)
            rows.append((a["aid"], int(a["spoof_gt"]), zr, up))
        if not rows:
            continue
        allz = np.concatenate([r[2] for r in rows])
        allu = np.log10(np.concatenate([r[3] for r in rows]) + 1e-9)
        mz, sz = allz.mean(), max(allz.std(), 1e-9)
        mu_, su = allu.mean(), max(allu.std(), 1e-9)
        ok = n_ = 0
        ev_ok = ev_n = 0
        for aid, sg, zr, up in rows:
            sc = (zr - mz) / sz + (np.log10(up + 1e-9) - mu_) / su
            pred = sc > 0
            ok += int((pred == bool(sg)).sum()); n_ += len(pred)
            ev_ok += int((pred.mean() > 0.5) == bool(sg)); ev_n += 1
        step_accs.append(ok / n_); ev_accs.append(ev_ok / ev_n)
    A(f"Pooled over {len(combos24)} combos: per-timestep accuracy {f3(np.mean(step_accs))} "
      f"(combo range {f3(min(step_accs))}-{f3(max(step_accs))}); event-majority "
      f"accuracy {f3(np.mean(ev_accs))} (range {f3(min(ev_accs))}-{f3(max(ev_accs))}).")
    A("")

    # ================= H2 =================
    A("## H2. Channel separation across mechanism buckets and categories")
    A("")
    A("Unit = events; per-event channel value = median zpeak across the view's")
    A("combos. Kruskal-Wallis per channel; pairwise exact Mann-Whitney with")
    A("Holm within each channel's pairwise set; Cliff's delta; cells n<5 are")
    A("descriptive. (The pilot-era dualstage-s3,s4 aleatoric split is")
    A("retired by Amendment A1; DualSTGF is out of thesis scope.)")
    A("")
    SHORT = {"physical": "phys", "deception_only": "dec_only",
             "deception_with_effect": "dec_eff", "SSSP": "SSSP",
             "SSMP": "SSMP", "MSMP": "MSMP"}
    for view, vname in [(t, "all-18"), (t[~t.is_pilot], "confirmatory-14")]:
        A(f"### H2 view: {vname}")
        A("")
        ev = view.groupby("attack_id").agg(
            bucket=("bucket", "first"), category=("category", "first"),
            modality=("target_modality", "first"),
            R=("zpeak_R", "median"), Aa=("zpeak_A", "median"),
            E=("zpeak_E", "median"), O=("zpeak_O", "median")).reset_index()
        for grp, gname in [("bucket", "mechanism buckets"), ("category", "SSSP/SSMP/MSMP")]:
            A(f"**Grouping: {gname}** "
              f"({dict(ev[grp].value_counts())})")
            A("")
            A("| channel | KW p | pairwise (Holm p, Cliff's d) |")
            A("|---|---|---|")
            for chn, cname in [("R", "residual"), ("Aa", "aleatoric"),
                               ("E", "epistemic"), ("O", "Omega")]:
                grps = {k: g[chn].values for k, g in ev.groupby(grp)}
                ks = list(grps)
                kw = stats.kruskal(*grps.values()).pvalue if len(grps) > 1 else np.nan
                prs, labs, ds = [], [], []
                for a_, b_ in combinations(ks, 2):
                    x_, y_ = grps[a_], grps[b_]
                    sa_, sb_ = SHORT.get(a_, a_), SHORT.get(b_, b_)
                    if min(len(x_), len(y_)) < 5:
                        labs.append(f"{sa_}~{sb_}: n<5 descr d={f3(cliffs(x_, y_))}")
                        continue
                    p = stats.mannwhitneyu(x_, y_, method="exact").pvalue
                    prs.append(p); ds.append(cliffs(x_, y_)); labs.append((sa_, sb_))
                hp = holm(np.array(prs)) if prs else []
                cells, j = [], 0
                for lb in labs:
                    if isinstance(lb, str):
                        cells.append(lb)
                    else:
                        cells.append(f"{lb[0]}~{lb[1]}: p={f3(hp[j])} d={f3(ds[j])}")
                        j += 1
                A(f"| {cname} | {f3(kw)} | " + "; ".join(cells) + " |")
            A("")
        # modality robustness (descriptive)
        A("**Modality robustness (KW p per channel on mechanism buckets, within "
          "modalities with >=8 events):**")
        for mod, g in ev.groupby("modality"):
            if len(g) < 8 or g.bucket.nunique() < 2:
                continue
            ps_ = {c: f3(stats.kruskal(*[h[c].values for _, h in g.groupby('bucket')]).pvalue)
                   for c in ["R", "Aa", "E", "O"]}
            A(f"- {mod} (n={len(g)}): " + ", ".join(f"{k}={v}" for k, v in ps_.items()))
        A("")
    # Manipulation-dynamics axis (prereg S9, frozen keyword rule; exploratory
    # stratifier -- verifier fix 5).
    def dyn_class(desc):
        d = (desc or "").lower()
        if "every second" in d or "per second" in d or "mm/s" in d:
            return "ramp"
        if d.startswith("set") or " set " in f" {d} ":
            return "set_point"
        for w in ("open", "close", "turn on", "turn off", "start", "stop",
                  "keep", "do not let"):
            if w in d:
                return "actuation"
        return "other"
    dyn = {aid: dyn_class(canon.get(aid, {}).get("attack_desc"))
           for aid in t.attack_id.unique()}
    evd = t.groupby("attack_id").agg(R=("zpeak_R", "median"), Aa=("zpeak_A", "median"),
                                     E=("zpeak_E", "median"), O=("zpeak_O", "median")).reset_index()
    evd["dyn"] = evd.attack_id.map(dyn)
    cnts = dict(evd.dyn.value_counts())
    A(f"### H2 exploratory stratifier: manipulation dynamics (frozen keyword rule; {cnts})")
    A("")
    big = [k for k, v in cnts.items() if v >= 5]
    sub = evd[evd.dyn.isin(big)]
    if sub.dyn.nunique() >= 2:
        ps_ = {c: f3(stats.kruskal(*[g[c].values for _, g in sub.groupby('dyn')]).pvalue)
               for c in ["R", "Aa", "E", "O"]}
        A("KW p across dynamics classes (>=5 events each, descriptive): " +
          ", ".join(f"{k}={v}" for k, v in ps_.items()))
    A("")
    # LOEO exploratory
    A("### H2 exploratory: LOEO logistic (pre-registered guards)")
    A("")
    from sklearn.linear_model import LogisticRegression
    evx = t.groupby("attack_id").agg(y=("deception", "first"),
        R=("zpeak_R", "median"), Aa=("zpeak_A", "median"),
        E=("zpeak_E", "median"), O=("zpeak_O", "median")).reset_index()
    X = evx[["R", "Aa", "E", "O"]].values
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    y_ = evx.y.values
    def loeo_auc(Xm, ym):
        sc = np.zeros(len(ym), float)
        for i in range(len(ym)):
            m = np.ones(len(ym), bool); m[i] = False
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xm[m], ym[m])
            sc[i] = lr.predict_proba(Xm[i:i + 1])[0, 1]
        pos, neg = sc[ym == 1], sc[ym == 0]
        return float((np.add.outer(pos, -neg) > 0).mean() +
                     0.5 * (np.add.outer(pos, -neg) == 0).mean())
    auc_obs = loeo_auc(X, y_)
    null = [loeo_auc(X, RNG.permutation(y_)) for _ in range(1000)]
    p_lo = float((np.array(null) >= auc_obs).mean())
    A(f"4-feature LOEO AUC = {f3(auc_obs)}; label-permutation p = {f3(p_lo)} "
      f"(1000 shuffles; null 95th pct = {f3(np.percentile(null, 95))}). "
      "Exploratory only; never enters the claims ladder. Definitional note "
      "(verifier fix 9): features are per-event MEDIANS of the max-over-"
      "sensors zpeaks across combos (the prereg named the channels by their "
      "max aggregation; the median-over-combos event summary is the choice "
      "made here and is stated for the record).")
    A("")

    # ================= H3 =================
    A("## H3. Triage utility (Layer B; held-out episodes; conservative verdicts)")
    A("")
    eh = e[e.in_heldout == 1]
    A("Verdict-conditional outcome tables, HELD-OUT episodes, per source.")
    A("Claims use the CONSERVATIVE (higher) false rate of the peak/onset")
    A("columns (prereg C3). CIs = combo-cluster bootstrap (10k; combos are")
    A("the replication cluster; episode counts are descriptors). FA/day is")
    A("indicative only, never a staffing figure.")
    A("")
    for src in ["gbm", "m0"]:
        sub = eh[eh.source == src]
        A(f"### Source: {src} ({'S2 supervised operating point' if src=='gbm' else 'anchored-M0 residual'})")
        A("")
        A("| verdict | n(peak) | false-rate peak | false-rate onset | CONSERVATIVE | combo-boot CI (cons.) |")
        A("|---|---|---|---|---|---|")
        # Footnote (verifier fix 8): the CONSERVATIVE column is the max of two
        # MARGINAL cells (different episode subsets) -- a descriptor. The
        # claim-bearing ordering below uses the stricter peak-AND-onset
        # intersection; never quote a table cell as the intersection figure.
        for v in sorted(sub.verdict_peak.unique()):
            rates = {}
            for col in ["verdict_peak", "verdict_onset"]:
                g = sub[sub[col] == v]
                rates[col] = (len(g), (g.truth == "false").mean() if len(g) else np.nan)
            cons_col = max(["verdict_peak", "verdict_onset"],
                           key=lambda c: (rates[c][1] if np.isfinite(rates[c][1]) else -1))
            gސ = sub[sub[cons_col] == v]
            pt, ci = combo_boot_rate(gސ, lambda d: int((d.truth == "false").sum()),
                                     lambda d: len(d))
            A(f"| {v} | {rates['verdict_peak'][0]} | {f3(rates['verdict_peak'][1])} | "
              f"{f3(rates['verdict_onset'][1])} | **{f3(pt)}** | "
              f"[{f3(ci[0])},{f3(ci[1])}] |")
        A("")
    A("Footnote (verifier fix 8): the CONSERVATIVE column above is the max of")
    A("two MARGINAL cells computed on different episode subsets -- a")
    A("descriptor. The claim-bearing ordering below uses the stricter")
    A("peak-AND-onset INTERSECTION; do not quote a table cell as that figure.")
    A("")
    # confirmatory ordering
    gh = eh[eh.source == "gbm"].copy()
    gh["cons_corrob"] = gh.verdict_peak.isin(CORROB) & gh.verdict_onset.isin(CORROB)
    corr_g = gh[gh.cons_corrob]
    quiet_g = gh[(gh.verdict_peak == "normal_quiet") & (gh.verdict_onset == "normal_quiet")]
    diffs = []
    groups = {c: g for c, g in gh.groupby("combo")}
    keys = list(groups)
    for _ in range(N_BOOT):
        pick = RNG.choice(len(keys), len(keys), replace=True)
        cn = cd_ = qn = qd = 0
        for i in pick:
            g = groups[keys[i]]
            cg = g[g.verdict_peak.isin(CORROB) & g.verdict_onset.isin(CORROB)]
            qg = g[(g.verdict_peak == "normal_quiet") & (g.verdict_onset == "normal_quiet")]
            cn += int((cg.truth == "false").sum()); cd_ += len(cg)
            qn += int((qg.truth == "false").sum()); qd += len(qg)
        if cd_ and qd:
            diffs.append(qn / qd - cn / cd_)
    p_ord = float((np.array(diffs) <= 0).mean()) if diffs else float("nan")
    A("### H3 confirmatory ordering (gbm, held-out, conservative verdicts)")
    A("")
    A(f"corroborated-verdict false rate = {f3((corr_g.truth=='false').mean())} "
      f"(n={len(corr_g)}) vs quiet false rate = {f3((quiet_g.truth=='false').mean())} "
      f"(n={len(quiet_g)}); combo-cluster bootstrap P(ordering violated) = {f3(p_ord)}.")
    A("")
    # R4 gate
    for src in ["gbm", "m0"]:
        g = eh[(eh.source == src) & (eh.verdict_peak == "R4_ood_suspect") &
               (eh.verdict_onset == "R4_ood_suspect")]
        prec = (g.truth == "true").mean() if len(g) else np.nan
        A(f"R4 escalation gate ({src}): held-out conservative precision = {f3(prec)} "
          f"(n={len(g)}) -> {'LIVE-ESCALATION ELIGIBLE' if (np.isfinite(prec) and prec>0) else 'EXPERIMENTAL'}")
    A("")
    # FA/day table
    A("### FA/day (held-out, exact Poisson 95% CI; INDICATIVE ONLY)")
    A("")
    A("| combo_source | FA/day held-out | CI95 | flag |")
    A("|---|---|---|---|")
    for k in sorted(asum):
        if k.startswith("dualstage"):
            continue                      # A1: out of thesis scope
        s_ = asum[k]
        flag = ""
        if k == "gdn_V2_s0_m0":
            flag = "gdn m0 seed-instability (collapsed C-slice thr_R)"
        if k == "gdn_V2_s42_m0":
            flag = "gdn m0 borderline seed-instability"
        A(f"| {k} | {s_['fa_per_day_heldout']} | {s_['fa_per_day_heldout_ci95']} | {flag} |")
    A("")
    m0_ok = [asum[k]["fa_per_day_heldout"] for k in asum
             if k.endswith("_m0") and not k.startswith("dualstage")
             and k not in ("gdn_V2_s0_m0",)]
    A(f"m0 band excluding the flagged gdn_s0 instability: "
      f"{f3(min(m0_ok))}-{f3(max(m0_ok))} FA/day held-out (per-combo rows are "
      f"descriptive; pooled statements only).")
    A("")

    # ================= H4 =================
    A("## H4. Stability across seeds (modal share primary; Krippendorff on")
    A("collapsed super-categories; Fleiss NOT used per prereg)")
    A("")
    A("| backbone | seeds | mean modal-verdict share (event-boot CI) | Kripp. alpha (collapsed) | Pbar |")
    A("|---|---|---|---|---|")
    for bb in BACKBONES:
        for tag, g in [("all 6", t[t.backbone == bb])]:
            shares, units = [], []
            for aid, ge in g.groupby("attack_id"):
                vs = list(ge.verdict)
                _, _c = modal(vs)
                vals, counts = np.unique(vs, return_counts=True)
                shares.append(counts.max() / len(vs))
                units.append([SUPER[v] for v in vs])
            mn, ci = event_boot_mean(shares)
            al, pbar = kripp_alpha_nominal(units)
            A(f"| {bb} ({tag}) | {g.seed.nunique()} | {f3(mn)} [{f3(ci[0])},{f3(ci[1])}] | "
              f"{f3(al)} | {f3(pbar)} |")
    A("")
    bf = {k: (v.get("band_flips"), v.get("medsplit_flips"), v.get("fullstream_flips"),
              v.get("ties")) for k, v in tsum.items()
          if k.split("_")[0] in BACKBONES}
    A("Threshold-sensitivity totals (band / med-split[label-using stress] / "
      "fullstream / ties): " +
      "; ".join(f"{k}:{v}" for k, v in sorted(bf.items())))
    A("")
    A(f"Cross-backbone modal-verdict agreement (collapsed categories, "
      f"all-{t.groupby(['backbone','seed']).ngroups}):")
    mod_bb = {}
    for bb, g in t.groupby("backbone"):
        mod_bb[bb] = {aid: SUPER[modal(list(ge.verdict))[0]]
                      for aid, ge in g.groupby("attack_id")}
    bbs = sorted(mod_bb)
    A("")
    A("| | " + " | ".join(bbs) + " |")
    A("|---" * (len(bbs) + 1) + "|")
    for a_ in bbs:
        cells = []
        for b_ in bbs:
            common = set(mod_bb[a_]) & set(mod_bb[b_])
            agr = np.mean([mod_bb[a_][k] == mod_bb[b_][k] for k in common])
            cells.append(f3(agr))
        A(f"| {a_} | " + " | ".join(cells) + " |")
    A("")

    # ================= H5 =================
    A("## H5. Capability on the 13 held-out attacks (confirmatory region)")
    A("")
    th = t[t.in_heldout == 1]
    A("### The deployment centerpiece (detection/triage exhibit; NEVER")
    A("mechanism-scored -- prereg S7)")
    A("")
    A("| attack | official point | intent | det m0 (k/6 per backbone g/t/c) | modal verdict (cstgl) | modal peak sensor (cstgl) | note |")
    A("|---|---|---|---|---|---|---|")
    for aid in sorted(th.attack_id.unique()):
        g = th[th.attack_id == aid]
        det = "/".join(str(int(g[g.backbone == b].detected.sum()))
                       for b in BACKBONES)
        gc = g[g.backbone == "cstgl"]
        if len(gc):
            mv, mtie = modal(list(gc.verdict))
            mv = f"{mv} (tie)" if mtie else mv
        else:
            mv = "-"
        msens = modal(list(gc.peak_sensor))[0] if len(gc) else "-"
        c = canon.get(aid, {})
        note = ""
        if aid == 29:
            note = "PRE-DECLARED CORRECT-QUIET ELIGIBLE (official: pumps never started, mechanical interlock); never counts toward recall"
        A(f"| A{aid} | {c.get('attack_point_raw','-')} | "
          f"{(c.get('expected_impact') or '-')[:40]} | {det} | {mv} | {msens} | {note} |")
    A("")
    # UQ-only catches
    A("### UQ-only visibility among held-out misses (typed R4b/R5/R6 when residual-missed)")
    A("")
    for bb, g in th[th.detected == 0].groupby("backbone"):
        uq = g[g.verdict.isin(["R4b_ood_rescue", "R5_benign_noise", "R6_data_gap"])]
        A(f"- {bb}: {len(uq)}/{len(g)} missed event-rows carry a UQ-channel verdict "
          f"({dict(uq.verdict.value_counts())})")
    A("")
    # latency -- CORRECTED metric (verification fix 1): within-window latency
    # = max(0, first-overlapping-true-episode.start - window.start). Episodes
    # already active at window start score 0 and are ALSO reported separately
    # as the 'alarm already active at onset' rate (a symptom of the documented
    # supervised-threshold FA flood, NOT faster response). The original
    # unclamped metric produced spurious 'gbm earlier' significance from
    # pre-window mega-episodes and is withdrawn.
    A("### Detection latency, m0 vs gbm -- WITHIN-WINDOW metric (clamped at 0;")
    A("one value per event = median over seeds; exact sign test; 10 s steps)")
    A("")
    et = e[(e.truth == "true") & (e.in_heldout == 1)].copy()
    et["aids"] = et.attack_ids.fillna("").apply(
        lambda s_: [int(x[1:]) for x in s_.split(";") if x])
    win = {(r.backbone, r.seed, r.attack_id): r.start for r in
           th[["backbone", "seed", "attack_id", "start"]].itertuples()}
    A("| backbone | n events | median lat m0 | median lat gbm | gbm earlier (k/n) | sign p | already-active@onset m0/gbm |")
    A("|---|---|---|---|---|---|---|")
    for bb in BACKBONES:
        lat, act = {}, {"m0": [0, 0], "gbm": [0, 0]}
        sub = et[et.backbone == bb]
        for r in sub.itertuples():
            for aid in r.aids:
                key = (r.backbone, r.seed, aid)
                if key not in win:
                    continue
                raw = r.start - win[key]
                d = lat.setdefault(aid, {}).setdefault(r.seed, {})
                prev = d.get(r.source)
                d[r.source] = raw if prev is None else min(prev, raw)
        per_ev = []
        for aid, seeds in lat.items():
            ds_ = [(v["m0"], v["gbm"]) for v in seeds.values()
                   if "m0" in v and "gbm" in v]
            if not ds_:
                continue
            for m0r, gbr in ds_:
                act["m0"][0] += int(m0r < 0); act["m0"][1] += 1
                act["gbm"][0] += int(gbr < 0); act["gbm"][1] += 1
            per_ev.append((aid,
                           float(np.median([max(0, x[0]) for x in ds_])),
                           float(np.median([max(0, x[1]) for x in ds_]))))
        if not per_ev:
            A(f"| {bb} | 0 | - | - | - | - | - |")
            continue
        m0l = [x[1] for x in per_ev]; gbl = [x[2] for x in per_ev]
        k = sum(1 for x in per_ev if x[2] < x[1])
        n = sum(1 for x in per_ev if x[2] != x[1])
        a_m0 = f"{act['m0'][0]}/{act['m0'][1]}"
        a_gb = f"{act['gbm'][0]}/{act['gbm'][1]}"
        A(f"| {bb} | {len(per_ev)} | {f3(np.median(m0l))} | {f3(np.median(gbl))} | "
          f"{k}/{n} | {f3(exact_binom(k, n))} | {a_m0} / {a_gb} |")
    A("")
    A("Reading: the already-active@onset column counts (event,seed) cases whose")
    A("first overlapping episode STARTED BEFORE the attack window -- an artifact")
    A("of the supervised source's threshold flood, reported as a rate and never")
    A("as a latency advantage. No 'earlier detection' claim is made unless the")
    A("clamped sign test is significant.")
    A("")
    # Exploratory localization (verifier fix 4; prereg S7/S9 descriptive
    # companions): modal peak-sensor hit@1 vs official targets and vs the
    # intent (expected-impact) equipment tokens.
    import re as _re
    def equip_tokens(txt):
        return {x.replace("-", "").upper()
                for x in _re.findall(r"[A-Z]{1,4}-?\d{3}", txt or "")}
    hits_t = hits_i = n_ev = 0
    for aid in sorted(th.attack_id.unique()):
        gc = th[(th.attack_id == aid) & (th.backbone == "cstgl")]
        if not len(gc):
            continue
        sens = modal(list(gc.peak_sensor))[0].replace("-", "").upper()
        tg = {x.replace("-", "").upper()
              for x in (canon.get(aid, {}).get("points") or [])}
        it = equip_tokens(canon.get(aid, {}).get("expected_impact"))
        n_ev += 1
        hits_t += int(sens in tg)
        hits_i += int(sens in it)
    A("### Exploratory localization (descriptive; hit@1 of the modal CST-GL")
    A("peak sensor on the 13 held-out attacks)")
    A("")
    A(f"target-hit@1 = {hits_t}/{n_ev}; intent-equipment-hit@1 = {hits_i}/{n_ev}.")
    A("(hit@3 was pre-registered as the exploratory metric; the episode")
    A("artifacts carry only the top-1 peak sensor, so hit@1 is reported and")
    A("hit@3 is explicitly deferred to the figures pass.)")
    A("")

    # ================= Conclusions =================
    A("## Conclusions against the pre-registered claims ladder")
    A("")
    A("Filled by the verification pass: the H1 claim gate outcome, the H2/H3")
    A("fallback wording, and the flags carried into the chapter are asserted")
    A("in the companion review (this file reports the numbers).")
    A("")
    open(f"{OUT}/PART2_TYPING_STATS.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}/PART2_TYPING_STATS.md ({len(L)} lines)")


if __name__ == "__main__":
    main()
