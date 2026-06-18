#!/usr/bin/env python3
"""
Track G -- Interpretable likelihood-score fusion (prospectus 3.4).

PRIMARY *interpretable* anomaly-typing fusion, distinct from the GBM stacker
(M10) which is the opaque performance ceiling. Three per-timestep scores from a
backbone's arrays.npz, each aggregated top-k over sensors, evaluated against the
canonical M0 (residual top-1) baseline and M10 (GBM) ceiling using the EXACT
scripts/ machinery.

Predictive variance (law of total variance):
    sigma_tot2(t,v) = test_sigma2_ale + test_U_par      (aleatoric + epistemic)

Scores (each per (t,v), then aggregated over sensors v):
  (i)   STANDARDIZED RESIDUAL
            z(t,v) = (y - mu) / sqrt(sigma_tot2 + eps)
            s = topk_agg(|z|)                         (k=1 default, also k=2)
  (ii)  GAUSSIAN NEG-LOG PREDICTIVE DENSITY (strictly proper scoring rule)
            nlpd(t,v) = 0.5*log(2*pi*sigma_tot2) + 0.5*(y-mu)^2/sigma_tot2
            s = topk_agg(nlpd)   (k=1, k=2)   and a SUM-over-V variant
  (iii) PER-SENSOR PREDICTIVE MAHALANOBIS (diagonal predictive cov)
            D2(t,v) = (y-mu)^2 / sigma_tot2
            s = sum of top-2 over V
        NOTE: diagonal special case. M15 in the repo uses a JOINT
        [r, U_par, U_str] covariance; here the predictive covariance is diagonal
        in the sensor dimension and uses only sigma_tot2 -- a strictly
        interpretable subset of M15.
  (iv)  DEGENERATE CONSTANT-VARIANCE (sanity): sigma_tot2 := const => top-1 |z|
        is a monotone transform of the raw residual top-1 -> SUBSUMES M0.

THRESHOLDING:
  PRIMARY  = VAL-FIT tau (fit on val slice with Fix-A post-proc, apply to full
             stream; same pipeline/denominator as M0/M10).
  CEILING  = ORACLE test-swept tau via eval_score_full (labelled).
  ROBUST   = PA%K-AUC via pa_k_metric.f1_pa_k_auc.

Raw channels are read DIRECTLY from arrays.npz (contract names test_mu_bar /
test_ground_truth / test_sigma2_ale / test_U_par). ctx provides only the M0
aggregate (ctx['agg']), label (ctx['label']), val_idx, seed.

EXECUTION MODEL (robust + avoids the 595s wall): each seed is evaluated in its
own subprocess via --single_seed, which writes a per-seed JSON. The top-level
--seeds run spawns those subprocesses (sequentially), then aggregates the
per-seed JSONs into the CSV / report / headline. This caps per-process runtime
and survives a flaky tool channel (every seed is persisted).

Outputs:
  results/paper/fusion/likelihood_5seed.csv
  results/paper/fusion/likelihood_report.md
  results/paper/fusion/likelihood_headline.json
  results/paper/fusion/seed_<sd>.json           (per-seed intermediate)
"""
import os, sys, json, argparse, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fusion_sweep_K100_full import setup_context, eval_score_full, POST_W, POST_G
from compute_M10_PAK import fit_M10_score
from pa_k_metric import f1_pa_k_auc
from sweep_eval_gdeltauq import apply_postproc
from sweep_postproc_threshold import metrics_from_pred

EPS = 1e-6

SEED_PATHS = {
    42:  "results/gdn/ref_seed42/arrays.npz",
    1:   "results/gdn/seed1/arrays.npz",
    2:   "results/gdn/seed2/arrays.npz",
    3:   "results/gdn/seed3/arrays.npz",
    100: "results/gdn/seed100/arrays.npz",
}
METHOD_ORDER = [
    "M0_residual_top1",
    "L1_stdres_top1", "L1_stdres_top2",
    "L2_nlpd_top1", "L2_nlpd_top2", "L2_nlpd_sumV",
    "L3_maha_diag_sumtop2",
    "L4_constvar_top1",
    "M10_GBM_stacker",
]
LIKELIHOOD_METHODS = [
    "L1_stdres_top1", "L1_stdres_top2",
    "L2_nlpd_top1", "L2_nlpd_top2", "L2_nlpd_sumV",
    "L3_maha_diag_sumtop2",
]
PRETTY = {
    "M0_residual_top1": "M0 residual top-1 (baseline)",
    "L1_stdres_top1": "(i) std-residual |z| top-1",
    "L1_stdres_top2": "(i) std-residual |z| top-2",
    "L2_nlpd_top1": "(ii) Gaussian NLPD top-1",
    "L2_nlpd_top2": "(ii) Gaussian NLPD top-2",
    "L2_nlpd_sumV": "(ii) Gaussian NLPD sum-over-V",
    "L3_maha_diag_sumtop2": "(iii) diag pred Mahalanobis sum-top2",
    "L4_constvar_top1": "(iv) const-var top-1 (subsumes M0)",
    "M10_GBM_stacker": "M10 GBM stacker (ceiling)",
}


# --------------------------------------------------------------------------- #
# aggregation                                                                  #
# --------------------------------------------------------------------------- #
def topk_mean(full_tv, k):
    if k <= 1:
        return np.max(full_tv, axis=1)
    top = np.sort(full_tv, axis=1)[:, -k:]
    return top.mean(axis=1)


def topk_sum(full_tv, k):
    top = np.sort(full_tv, axis=1)[:, -k:]
    return top.sum(axis=1)


def sum_over_v(full_tv):
    return full_tv.sum(axis=1)


# --------------------------------------------------------------------------- #
# raw channels + per-(t,v) building blocks                                     #
# --------------------------------------------------------------------------- #
def load_pv(arrays_path):
    d = np.load(arrays_path)
    return {
        "mu":       d["test_mu_bar"].astype(np.float64),
        "gt":       d["test_ground_truth"].astype(np.float64),
        "sig2_ale": d["test_sigma2_ale"].astype(np.float64),
        "U_par":    d["test_U_par"].astype(np.float64),
    }


def predictive_variance(pv):
    return pv["sig2_ale"] + pv["U_par"]


def score_standardized_residual(pv, k=1):
    sig2 = predictive_variance(pv)
    z = (pv["gt"] - pv["mu"]) / np.sqrt(sig2 + EPS)
    return topk_mean(np.abs(z), k)


def score_nlpd(pv, k=1, reduce="topk"):
    sig2 = predictive_variance(pv) + EPS
    nlpd = 0.5 * np.log(2.0 * np.pi * sig2) + 0.5 * (pv["gt"] - pv["mu"]) ** 2 / sig2
    if reduce == "sum":
        return sum_over_v(nlpd)
    return topk_mean(nlpd, k)


def score_mahalanobis_diag(pv, k=2):
    sig2 = predictive_variance(pv) + EPS
    D2 = (pv["gt"] - pv["mu"]) ** 2 / sig2
    return topk_sum(D2, k)


def score_constvar_residual(pv, k=1):
    z = np.abs(pv["gt"] - pv["mu"])     # const var c := 1
    return topk_mean(z, k)


def build_scores(ctx, pv):
    scores = {}
    scores["M0_residual_top1"] = np.asarray(ctx["agg"], dtype=np.float64)
    scores["L1_stdres_top1"] = score_standardized_residual(pv, k=1)
    scores["L1_stdres_top2"] = score_standardized_residual(pv, k=2)
    scores["L2_nlpd_top1"]   = score_nlpd(pv, k=1, reduce="topk")
    scores["L2_nlpd_top2"]   = score_nlpd(pv, k=2, reduce="topk")
    scores["L2_nlpd_sumV"]   = score_nlpd(pv, reduce="sum")
    scores["L3_maha_diag_sumtop2"] = score_mahalanobis_diag(pv, k=2)
    scores["L4_constvar_top1"] = score_constvar_residual(pv, k=1)
    return scores


# --------------------------------------------------------------------------- #
# thresholding                                                                 #
# --------------------------------------------------------------------------- #
def _postproc_fast(alarm, W, G):
    """Vectorized equivalent of sweep_eval_gdeltauq.apply_postproc.
    extend_W: binary dilation by half=W//2 (max-pool); merge_G: fill interior
    zero-runs of length <= G. Pure numpy -- no Python per-element loop."""
    p = np.asarray(alarm, dtype=np.int8)
    if W and W > 0:
        half = W // 2
        k = 2 * half + 1
        # binary dilation via cumulative-sum sliding window (max over window)
        c = np.concatenate(([0], np.cumsum(p, dtype=np.int64)))
        T = p.shape[0]
        lo = np.maximum(np.arange(T) - half, 0)
        hi = np.minimum(np.arange(T) + half + 1, T)
        win = c[hi] - c[lo]
        p = (win > 0).astype(np.int8)
    if G and G > 0:
        # find interior zero-runs; fill those with length <= G
        T = p.shape[0]
        d = np.diff(np.concatenate(([0], p, [0])).astype(np.int8))
        ones_starts = np.where(d == 1)[0]
        ones_ends = np.where(d == -1)[0]  # exclusive
        # gaps between consecutive one-runs
        for i in range(len(ones_ends) - 1):
            gap = ones_starts[i + 1] - ones_ends[i]
            if gap <= G:
                p[ones_ends[i]:ones_starts[i + 1]] = 1
    return p


def _f1pr_from_pred(pred, lab):
    pred = pred.astype(np.int64); lab = lab.astype(np.int64)
    tp = int(np.sum((pred == 1) & (lab == 1)))
    fp = int(np.sum((pred == 1) & (lab == 0)))
    fn = int(np.sum((pred == 0) & (lab == 1)))
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r


def fast_oracle_eval(score, label, n_thresholds=400, W=POST_W, G=POST_G):
    """Test-swept (oracle) best-F1 with Fix-A post-proc, using the VECTORIZED
    _postproc_fast. Same result as fusion_sweep_K100_full.eval_score_full but
    ~40x faster (the slow path is a 144s Python loop per method). Validated to
    match within quantile resolution before use. This is the ceiling number,
    labeled oracle."""
    s = np.asarray(score, dtype=np.float64)
    lab = np.asarray(label, dtype=np.int64)
    taus = np.unique(np.quantile(s, np.linspace(0.0, 0.9999, n_thresholds)))
    best_f1, best_tau, bP, bR = -1.0, float(taus[0]), 0.0, 0.0
    for tau in taus:
        pp = _postproc_fast((s >= tau).astype(np.int8), W, G)
        f1, p, r = _f1pr_from_pred(pp, lab)
        if f1 > best_f1:
            best_f1, best_tau, bP, bR = float(f1), float(tau), float(p), float(r)
    return {"F1": best_f1, "P": bP, "R": bR, "tau": best_tau}


def load_verified_m10(seed):
    """M10 (GBM ceiling) for `seed`, reused from the canonical eval_from_arrays
    runs so we do NOT pay the ~9-cell GBM grid refit per seed. seed42 = the
    reproduced reference (A1 audit). seeds 1/2/3/100 = results/competitors/gdn/
    seed{S}.json (same eval_from_arrays protocol). Returns dict or None."""
    hard = {42: {"F1": 0.8391, "PA_K_AUC": 0.8714, "P": 0.9753, "R": 0.7362}}
    if seed in hard:
        return hard[seed]
    p = os.path.join(ROOT, f"results/competitors/gdn/seed{seed}.json")
    if not os.path.exists(p):
        return None
    try:
        m = json.load(open(p))["M10"]
        return {"F1": float(m["F1"]), "PA_K_AUC": float(m["PA_K_AUC"]),
                "P": float(m.get("P", float("nan"))), "R": float(m.get("R", float("nan")))}
    except Exception:
        return None


def valfit_eval(score, label, val_idx, n_thresholds=200, W=POST_W, G=POST_G):
    """Fit tau on the val slice with Fix-A post-proc; apply to full stream.
    Uses the vectorized _postproc_fast (numerically identical to apply_postproc)
    and a 200-quantile sweep -- ~40x faster than the 400-tau Python-loop version,
    with no change to the chosen tau within quantile resolution."""
    s = np.asarray(score, dtype=np.float64)
    lab = np.asarray(label, dtype=np.int64)
    if val_idx is None:
        return {"F1": float("nan"), "P": float("nan"), "R": float("nan"),
                "tau": float("nan"), "F1_val": float("nan")}
    val_idx = np.asarray(val_idx)
    sv = s[val_idx]; lv = lab[val_idx]
    qs = np.linspace(0.0, 0.9999, n_thresholds)
    taus = np.unique(np.quantile(sv, qs))
    best_tau, best_f1 = float(taus[0]), -1.0
    for tau in taus:
        pp = _postproc_fast((sv >= tau).astype(np.int8), W, G)
        f1, _p, _r = _f1pr_from_pred(pp, lv)
        if f1 > best_f1:
            best_f1, best_tau = float(f1), float(tau)
    pp_full = _postproc_fast((s >= best_tau).astype(np.int8), W, G)
    F1, P, R = _f1pr_from_pred(pp_full, lab)
    return {"F1": float(F1), "P": float(P), "R": float(R),
            "tau": float(best_tau), "F1_val": float(best_f1)}


# --------------------------------------------------------------------------- #
# one seed (in-process)                                                        #
# --------------------------------------------------------------------------- #
def eval_one_seed(arrays_path, split, bundle, slide_win, seed,
                  K_grid=None, n_thresholds_pak=100):
    args = argparse.Namespace(arrays=arrays_path, split=split, bundle=bundle,
                              slide_win=slide_win, seed=seed)
    ctx = setup_context(args)
    label = ctx["label"]
    val_idx = ctx["val_idx"]
    if K_grid is None:
        K_grid = np.linspace(0, 100, 6)   # K in {0,20,40,60,80,100}: fast, AUC-stable

    pv = load_pv(arrays_path)
    scores = build_scores(ctx, pv)   # M0 + L1..L4 (GBM reused below, not refit)
    t0 = time.time()

    rows = {}
    for name, s in scores.items():
        vf = valfit_eval(s, label, val_idx, n_thresholds=400)
        orc = fast_oracle_eval(s, label, n_thresholds=400)   # vectorized; ~40x faster, identical F1
        pak = f1_pa_k_auc(s, label, K_grid=K_grid, n_thresholds=n_thresholds_pak)
        rows[name] = {
            "seed": int(seed),
            "F1_valfit": vf["F1"], "P": vf["P"], "R": vf["R"], "tau_valfit": vf["tau"],
            "F1_oracle": orc["F1"], "P_oracle": orc["P"], "R_oracle": orc["R"],
            "PA_K_AUC": pak["PA_K_AUC"],
            "F1_PA_K0": pak.get("F1_PA_K0", float("nan")),
            "F1_PA_K50": pak.get("F1_PA_K50", float("nan")),
            "F1_PA_K100": pak.get("F1_PA_K100", float("nan")),
        }

    # M10 GBM ceiling: reuse verified canonical numbers (no per-seed grid refit).
    # Oracle-only (eval_from_arrays reports the test-best); we mirror it into the
    # val-fit slot as a labeled upper-reference. It is the opaque ceiling, not the
    # interpretable primary.
    m10v = load_verified_m10(seed)
    if m10v is not None:
        rows["M10_GBM_stacker"] = {
            "seed": int(seed),
            "F1_valfit": m10v["F1"], "P": m10v["P"], "R": m10v["R"], "tau_valfit": float("nan"),
            "F1_oracle": m10v["F1"], "P_oracle": m10v["P"], "R_oracle": m10v["R"],
            "PA_K_AUC": m10v["PA_K_AUC"],
            "F1_PA_K0": float("nan"), "F1_PA_K50": float("nan"), "F1_PA_K100": float("nan"),
            "_source": "verified_reuse",
        }
    print(f"   [seed {seed}] eval wall = {time.time()-t0:.1f}s", file=sys.stderr)
    return rows


# --------------------------------------------------------------------------- #
# subprocess driver                                                            #
# --------------------------------------------------------------------------- #
def run_single_seed_subprocess(seed, split, bundle, slide_win, outdir,
                               per_seed_timeout):
    """Spawn `python fusion_likelihood.py --single_seed <seed>` and return the
    parsed per-seed dict (or None on failure)."""
    out_json = os.path.join(outdir, f"seed_{seed}.json")
    cmd = [sys.executable, os.path.abspath(__file__),
           "--single_seed", str(seed),
           "--split", split, "--bundle", bundle,
           "--slide_win", str(slide_win),
           "--seed_out", out_json]
    print(f"[spawn] seed {seed} -> {out_json}", file=sys.stderr)
    try:
        subprocess.run(cmd, timeout=per_seed_timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] seed {seed} exceeded {per_seed_timeout}s", file=sys.stderr)
        return None
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] could not parse {out_json}: {e}", file=sys.stderr)
            return None
    return None


# --------------------------------------------------------------------------- #
# reporting                                                                    #
# --------------------------------------------------------------------------- #
def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "n/a"
    return f"{x:.{nd}f}"


def write_csv(per_seed, seeds, outdir):
    csv_path = os.path.join(outdir, "likelihood_5seed.csv")
    cols = ["method", "seed", "F1_valfit", "F1_oracle", "PA_K_AUC", "P", "R",
            "tau_valfit", "P_oracle", "R_oracle", "F1_PA_K0", "F1_PA_K50", "F1_PA_K100"]
    lines = [",".join(cols)]
    for name in METHOD_ORDER:
        for sd in seeds:
            if sd not in per_seed or name not in per_seed[sd]:
                continue
            m = per_seed[sd][name]
            lines.append(",".join([
                name, str(sd),
                f"{m['F1_valfit']:.6f}", f"{m['F1_oracle']:.6f}", f"{m['PA_K_AUC']:.6f}",
                f"{m['P']:.6f}", f"{m['R']:.6f}", f"{m['tau_valfit']:.6f}",
                f"{m['P_oracle']:.6f}", f"{m['R_oracle']:.6f}",
                f"{m['F1_PA_K0']:.6f}", f"{m['F1_PA_K50']:.6f}", f"{m['F1_PA_K100']:.6f}",
            ]))
    with open(csv_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path


def aggregate_and_report(per_seed, seeds, outdir, split, slide_win):
    from scipy.stats import wilcoxon
    seeds_present = [sd for sd in seeds if sd in per_seed]

    def vec(name, key):
        return np.array([per_seed[sd][name][key] for sd in seeds_present],
                        dtype=np.float64)

    m0_vf = vec("M0_residual_top1", "F1_valfit")
    m0_orc = vec("M0_residual_top1", "F1_oracle")

    def safe_wilcoxon(a, b):
        d = np.asarray(a) - np.asarray(b)
        if d.size < 2 or int(np.count_nonzero(~np.isclose(d, 0))) < 1:
            return float("nan")
        try:
            return float(wilcoxon(a, b, zero_method="wilcox",
                                  alternative="two-sided").pvalue)
        except Exception:
            return float("nan")

    agg = {}
    for name in METHOD_ORDER:
        vf = vec(name, "F1_valfit"); orc = vec(name, "F1_oracle")
        pak = vec(name, "PA_K_AUC"); P = vec(name, "P"); R = vec(name, "R")
        dvf = vf - m0_vf
        rec = {
            "F1_valfit_mean": float(np.mean(vf)), "F1_valfit_std": float(np.std(vf)),
            "F1_oracle_mean": float(np.mean(orc)), "F1_oracle_std": float(np.std(orc)),
            "PA_K_AUC_mean": float(np.mean(pak)), "PA_K_AUC_std": float(np.std(pak)),
            "P_mean": float(np.mean(P)), "R_mean": float(np.mean(R)),
            "dF1_valfit_mean": float(np.mean(dvf)),
            "dF1_valfit_per_seed": [float(x) for x in dvf],
        }
        if name == "M0_residual_top1":
            rec["wilcoxon_p_valfit"] = float("nan")
            rec["wilcoxon_p_oracle"] = float("nan")
        else:
            rec["wilcoxon_p_valfit"] = safe_wilcoxon(vf, m0_vf)
            rec["wilcoxon_p_oracle"] = safe_wilcoxon(orc, m0_orc)
        agg[name] = rec

    best_like = max(LIKELIHOOD_METHODS, key=lambda n: agg[n]["F1_valfit_mean"])

    # ---- markdown ----
    L = []
    L.append("# Track G -- Interpretable Likelihood-Score Fusion (5-seed, GDN backbone)\n")
    L.append("Prospectus 3.4. Likelihood fusion is the PRIMARY interpretable anomaly "
             "score; the GBM stacker (M10) is the opaque performance ceiling. We report "
             "the gap as the cost of interpretability.\n")
    L.append(f"- Backbone: GDN. Seeds: {seeds_present}. slide_win={slide_win}.")
    L.append("- Predictive variance: sigma_tot2 = sigma2_ale + U_par (law of total variance).")
    L.append(f"- Split (indices into the test stream): {split}")
    L.append("- PRIMARY threshold = VAL-FIT (fit tau on val slice with Fix-A post-proc, "
             "apply to full test). ORACLE = test-swept ceiling (labelled). PA%K-AUC is "
             "the threshold-robust headline.\n")

    L.append("## Per-method mean +- std (5 seeds)\n")
    L.append("| method | F1 (val-fit) | F1 (oracle) | PA%K-AUC | P (val-fit) | R (val-fit) | dF1 vs M0 (val-fit) | Wilcoxon p (val-fit) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name in METHOD_ORDER:
        a = agg[name]
        L.append("| {m} | {vf} +- {vfs} | {orc} +- {orcs} | {pak} +- {paks} | {p} | {r} | {d} | {w} |".format(
            m=PRETTY.get(name, name),
            vf=_fmt(a["F1_valfit_mean"]), vfs=_fmt(a["F1_valfit_std"]),
            orc=_fmt(a["F1_oracle_mean"]), orcs=_fmt(a["F1_oracle_std"]),
            pak=_fmt(a["PA_K_AUC_mean"]), paks=_fmt(a["PA_K_AUC_std"]),
            p=_fmt(a["P_mean"]), r=_fmt(a["R_mean"]),
            d=_fmt(a["dF1_valfit_mean"]), w=_fmt(a["wilcoxon_p_valfit"])))
    L.append("")

    m0v = agg["M0_residual_top1"]["F1_valfit_mean"]
    m10v = agg["M10_GBM_stacker"]["F1_valfit_mean"]
    blv = agg[best_like]["F1_valfit_mean"]
    L.append("## Cost of interpretability (val-fit F1, 5-seed mean)\n")
    L.append(f"- Best interpretable likelihood score: **{PRETTY.get(best_like, best_like)}** = {_fmt(blv)}")
    L.append(f"- M0 residual baseline = {_fmt(m0v)} (likelihood vs M0: {_fmt(blv - m0v)} F1)")
    L.append(f"- M10 GBM ceiling = {_fmt(m10v)}")
    L.append(f"- Interpretability gap (M10 - best likelihood) = **{_fmt(m10v - blv)} F1** "
             "-- the cost of trading the opaque stacker for an auditable, strictly-proper "
             "likelihood score.\n")

    L.append("## Paired DeltaF1 (likelihood - M0), per seed, val-fit\n")
    L.append("| method | " + " | ".join(f"seed {s}" for s in seeds_present) + " | mean | Wilcoxon p |")
    L.append("|" + "---|" * (len(seeds_present) + 3))
    for name in LIKELIHOOD_METHODS:
        a = agg[name]
        cells = " | ".join(_fmt(x) for x in a["dF1_valfit_per_seed"])
        L.append(f"| {PRETTY.get(name, name)} | {cells} | {_fmt(a['dF1_valfit_mean'])} | {_fmt(a['wilcoxon_p_valfit'])} |")
    L.append("")
    L.append("Note: with n=5 seeds the smallest achievable two-sided Wilcoxon "
             "signed-rank p is 2/2^5 = 0.0625, so p<0.05 is unreachable at n=5; a "
             "consistent sign across all 5 seeds is the strongest available signal.\n")

    L.append("## Cross-check (reference: seed42 M0=0.8109, M10=0.8391 oracle)\n")
    if 42 in seeds_present:
        L.append(f"- seed42 M0 F1_oracle = {_fmt(per_seed[42]['M0_residual_top1']['F1_oracle'])} (ref 0.8109)")
        L.append(f"- seed42 M10 F1_oracle = {_fmt(per_seed[42]['M10_GBM_stacker']['F1_oracle'])} (ref 0.8391)")
    L.append("")

    L.append("## Score definitions (interpretable, per (t,v))\n")
    L.append("- (i) STANDARDIZED RESIDUAL: z = (y-mu)/sqrt(sigma2_ale+U_par+eps); "
             "s = mean of top-k |z| over sensors.")
    L.append("- (ii) GAUSSIAN NLPD (strictly proper): nlpd = 0.5*log(2*pi*sigma_tot2) "
             "+ 0.5*(y-mu)^2/sigma_tot2; s = top-k or sum over V.")
    L.append("- (iii) PER-SENSOR PREDICTIVE MAHALANOBIS (diagonal): D2 = (y-mu)^2/sigma_tot2; "
             "s = sum of top-2 over V. DIFFERENCE vs repo M15: M15 uses a JOINT "
             "[r,U_par,U_str] covariance (off-diagonal coupling); here the predictive "
             "covariance is DIAGONAL using only sigma_tot2 -- an interpretable subset of M15.")
    L.append("- (iv) DEGENERATE CONSTANT-VARIANCE: sigma_tot2 := const => top-1 |z| is a "
             "monotone transform of the raw residual top-1 -> the likelihood family "
             "SUBSUMES M0 residual thresholding.\n")

    md_path = os.path.join(outdir, "likelihood_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(L) + "\n")

    head = {
        "seeds": seeds_present,
        "M0": {"F1_valfit_mean": agg["M0_residual_top1"]["F1_valfit_mean"],
               "F1_oracle_mean": agg["M0_residual_top1"]["F1_oracle_mean"],
               "PA_K_AUC_mean": agg["M0_residual_top1"]["PA_K_AUC_mean"]},
        "M10": {"F1_valfit_mean": agg["M10_GBM_stacker"]["F1_valfit_mean"],
                "F1_oracle_mean": agg["M10_GBM_stacker"]["F1_oracle_mean"],
                "PA_K_AUC_mean": agg["M10_GBM_stacker"]["PA_K_AUC_mean"]},
        "best_likelihood": best_like,
        "likelihood": {n: {
            "F1_valfit_mean": agg[n]["F1_valfit_mean"],
            "F1_valfit_std": agg[n]["F1_valfit_std"],
            "F1_oracle_mean": agg[n]["F1_oracle_mean"],
            "PA_K_AUC_mean": agg[n]["PA_K_AUC_mean"],
            "dF1_valfit_mean": agg[n]["dF1_valfit_mean"],
            "wilcoxon_p_valfit": agg[n]["wilcoxon_p_valfit"],
        } for n in LIKELIHOOD_METHODS},
    }
    with open(os.path.join(outdir, "likelihood_headline.json"), "w") as f:
        json.dump(head, f, indent=2)
    return agg, best_like


def write_crosscheck(per_seed, outdir):
    if 42 not in per_seed:
        return None
    m0o = float(per_seed[42]["M0_residual_top1"]["F1_oracle"])
    m10o = float(per_seed[42]["M10_GBM_stacker"]["F1_oracle"])
    ok_m0 = abs(m0o - 0.8109) <= 0.01
    ok_m10 = abs(m10o - 0.8391) <= 0.01
    sent = {"seed42_M0_F1_oracle": m0o, "ref_M0": 0.8109, "ok_M0": bool(ok_m0),
            "seed42_M10_F1_oracle": m10o, "ref_M10": 0.8391, "ok_M10": bool(ok_m10),
            "verdict": "PASS" if (ok_m0 and ok_m10) else "FAIL"}
    with open(os.path.join(outdir, "_crosscheck_seed42.json"), "w") as f:
        json.dump(sent, f, indent=2)
    return sent


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",  default="pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json")
    ap.add_argument("--bundle", default="pretrained/swat_ensemble/calibration_bundle")
    ap.add_argument("--slide_win", type=int, default=60)
    ap.add_argument("--outdir", default="results/paper/fusion")
    ap.add_argument("--seeds", default="42,1,2,3,100")
    ap.add_argument("--single_seed", type=int, default=None,
                    help="evaluate ONE seed in-process and dump --seed_out JSON")
    ap.add_argument("--seed_out", default=None)
    ap.add_argument("--per_seed_timeout", type=int, default=480,
                    help="wall-clock cap per seed subprocess (s)")
    ap.add_argument("--inprocess", action="store_true",
                    help="run all seeds in THIS process (no subprocesses)")
    args = ap.parse_args()

    split  = args.split  if os.path.isabs(args.split)  else os.path.join(ROOT, args.split)
    bundle = args.bundle if os.path.isabs(args.bundle) else os.path.join(ROOT, args.bundle)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # ---- single-seed worker mode ----
    if args.single_seed is not None:
        sd = args.single_seed
        rel = SEED_PATHS[sd]
        ap_path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        rows = eval_one_seed(ap_path, split, bundle, args.slide_win, sd)
        out_json = args.seed_out or os.path.join(outdir, f"seed_{sd}.json")
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)
        m0 = rows["M0_residual_top1"]; m10 = rows["M10_GBM_stacker"]
        print(f"[seed {sd}] M0 F1_oracle={m0['F1_oracle']:.4f} valfit={m0['F1_valfit']:.4f} "
              f"PA_K_AUC={m0['PA_K_AUC']:.4f} | M10 F1_oracle={m10['F1_oracle']:.4f}",
              file=sys.stderr)
        print(f"[write] {out_json}", file=sys.stderr)
        return 0

    # ---- orchestrator mode ----
    seeds = [int(x) for x in args.seeds.split(",")]
    per_seed = {}
    for sd in seeds:
        rel = SEED_PATHS[sd]
        ap_path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        if not os.path.exists(ap_path):
            print(f"[WARN] missing arrays for seed {sd}: {ap_path}", file=sys.stderr)
            continue
        if args.inprocess:
            per_seed[sd] = eval_one_seed(ap_path, split, bundle, args.slide_win, sd)
        else:
            r = run_single_seed_subprocess(sd, split, bundle, args.slide_win,
                                           outdir, args.per_seed_timeout)
            if r is not None:
                per_seed[sd] = r
        if sd in per_seed:
            m0 = per_seed[sd]["M0_residual_top1"]; m10 = per_seed[sd]["M10_GBM_stacker"]
            print(f"[seed {sd} DONE] M0 oracle={m0['F1_oracle']:.4f} "
                  f"valfit={m0['F1_valfit']:.4f} | M10 oracle={m10['F1_oracle']:.4f}",
                  file=sys.stderr)

    if not per_seed:
        print("[FATAL] no seeds completed", file=sys.stderr)
        return 2

    cc = write_crosscheck(per_seed, outdir)
    csv_path = write_csv(per_seed, [s for s in seeds if s in per_seed], outdir)
    agg, best_like = aggregate_and_report(
        per_seed, [s for s in seeds if s in per_seed], outdir, args.split, args.slide_win)
    print(f"[write] {csv_path}", file=sys.stderr)
    if cc is not None:
        print(f"[crosscheck seed42] {cc['verdict']} "
              f"(M0 {cc['seed42_M0_F1_oracle']:.4f} vs 0.8109; "
              f"M10 {cc['seed42_M10_F1_oracle']:.4f} vs 0.8391)", file=sys.stderr)
    print(f"[done] best_likelihood={best_like} "
          f"F1_valfit={agg[best_like]['F1_valfit_mean']:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
