#!/usr/bin/env python
"""
Track H - Step 2: Typing separation test.

Loads results/<backbone>/arrays.npz + results/paper/typing/type_labels.npz and
asks the KILL-TEST question: do the UQ channels separate SWaT attack categories?

Channels computed per timestep (T,):
  z_resid_top1  : top-1 over V of standardized residual |y-mu|/sqrt(sig2_ale+U_par+eps)
  resid_top1    : top-1 over V of raw |y-mu| (smoothed err score, reused topk_aggregate)
  sigma_ale_max : max over V of sqrt(sigma2_ale)
  U_par_max     : max over V of epistemic var
  U_str_max     : max over E of structural unc (if present, else NaN)
  U_dist        : distributional channel (T,)
Plus TARGETED versions: same channels but restricted to the targeted sensor
indices of the attack active at that timestep (only defined on anomalous steps).

HEADLINE SEPARATION TEST (anomalous timesteps only):
  group A = sensor_spoof  (actual_change == false  -> type_spoof==1)
  group B = actuator/phys (actual_change == true   -> type_spoof==2)
  Report per-channel AUROC(A vs B), Cohen-d, rank-biserial.

Also: NORMAL-vs-each-category AUROC (does any channel flag a category above
the normal baseline), and a transparent-rule CONFUSION MATRIX.

Dependency-light: numpy + sklearn (roc_auc_score) only. ASCII only.
"""
import os, sys, json
import numpy as np

ROOT = "/mnt/datassd3/rashinda/UQ_GNN_AnomalyTyping"
LISTTXT = os.path.join(ROOT, "data/swat/list.txt")
TARGETS = os.path.join(ROOT, "data/swat/attack_targets.json")
TYPEFILE = os.path.join(ROOT, "results/paper/typing/type_labels.npz")
OUTDIR = os.path.join(ROOT, "results/paper/typing")

EPS = 1e-6

try:
    from sklearn.metrics import roc_auc_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def auroc(y, s):
    """AUROC of score s separating binary y (1=positive). NaN-safe."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    if HAVE_SK:
        try:
            return float(roc_auc_score(y, s))
        except Exception:
            return float("nan")
    # Mann-Whitney fallback
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt + 1
    avg = (start + csum) / 2.0
    ranks = avg[inv]
    n1 = y.sum()
    n0 = len(y) - n1
    r1 = ranks[y == 1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))


def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp)


def rank_biserial_from_auc(au):
    if not np.isfinite(au):
        return float("nan")
    return float(2.0 * au - 1.0)


def load_point_names(path):
    names = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                names.append(s)
    return names


def load_attacks(path):
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "attacks" in obj:
        return obj["attacks"], obj.get("metadata", {})
    if isinstance(obj, list):
        return obj, {}
    raise ValueError("bad attack json")


def smooth_err(test_mu, test_gt, val_mu, val_gt, smooth=3):
    """Per-sensor standardized error score, median-normalized using val stats.
    Returns (V,T) error matrix. Lightweight reimplementation matching the
    spirit of sweep_eval_gdeltauq.build_full_err_scores (median/IQR scale)."""
    V = test_mu.shape[1]
    err = np.abs(test_gt - test_mu)            # (T,V)
    verr = np.abs(val_gt - val_mu)             # (Tv,V)
    med = np.median(verr, axis=0)              # (V,)
    iqr = (np.percentile(verr, 75, axis=0) - np.percentile(verr, 25, axis=0))
    scale = iqr + EPS
    sc = (err - med) / scale                   # (T,V)
    sc = sc.T                                   # (V,T)
    if smooth and smooth > 1:
        k = smooth
        ker = np.ones(k) / k
        sc = np.vstack([np.convolve(sc[v], ker, mode="same") for v in range(V)])
    return sc                                   # (V,T)


def build_channels(arrays_path):
    d = np.load(arrays_path)
    mu = d["test_mu_bar"].astype(np.float64)
    gt = d["test_ground_truth"].astype(np.float64)
    vmu = d["val_mu_bar"].astype(np.float64)
    vgt = d["val_ground_truth"].astype(np.float64)
    upar = d["test_U_par"].astype(np.float64)          # (T,V) epistemic var
    sale = d["test_sigma2_ale"].astype(np.float64)     # (T,V) aleatoric var
    label = d["test_attack_label"].astype(np.int64)
    T, V = mu.shape

    udist = d["test_U_dist"].astype(np.float64) if "test_U_dist" in d.files else upar.mean(1)
    has_ustr = "test_U_str" in d.files
    ustr = d["test_U_str"].astype(np.float64) if has_ustr else None  # (T,E)

    sigtot = sale + upar                                # total predictive var
    z = np.abs(gt - mu) / np.sqrt(sigtot + EPS)         # (T,V) standardized resid

    # raw smoothed err score (V,T) -> top-1 over V
    errVT = smooth_err(mu, gt, vmu, vgt, smooth=3)      # (V,T)

    ch = {}
    ch["z_resid_top1"] = z.max(1)                       # (T,)
    ch["resid_top1"] = errVT.max(0)                     # (T,)
    ch["sigma_ale_max"] = np.sqrt(sale).max(1)
    ch["U_par_max"] = upar.max(1)
    ch["U_dist"] = udist
    if has_ustr:
        ch["U_str_max"] = ustr.max(1)
    # also means (more stable for category-level contrasts)
    ch["sigma_ale_mean"] = np.sqrt(sale).mean(1)
    ch["U_par_mean"] = upar.mean(1)
    if has_ustr:
        ch["U_str_mean"] = ustr.mean(1)

    extras = {"mu": mu, "gt": gt, "upar": upar, "sale": sale, "z": z,
              "errVT": errVT, "ustr": ustr, "has_ustr": has_ustr,
              "label": label, "T": T, "V": V}
    return ch, extras


def build_targeted_channels(attacks, names, type_aid, extras):
    """For each timestep where an attack is active, compute channel values
    restricted to that attack's targeted sensor indices (max over targets).
    Returns dict of (T,) arrays with NaN where no targeted sensor maps."""
    T = extras["T"]
    name2idx = {n: i for i, n in enumerate(names)}
    aid2idx = {}
    n_unmapped = 0
    for a in attacks:
        aid = int(a.get("attack_id", -1))
        tg = a.get("targets", []) or []
        idxs = []
        for t in tg:
            if t in name2idx:
                idxs.append(name2idx[t])
        aid2idx[aid] = idxs
        if tg and not idxs:
            n_unmapped += 1

    z = extras["z"]; sale = extras["sale"]; upar = extras["upar"]; errVT = extras["errVT"]
    tg_z = np.full(T, np.nan); tg_sa = np.full(T, np.nan)
    tg_up = np.full(T, np.nan); tg_re = np.full(T, np.nan)
    for t in range(T):
        aid = int(type_aid[t])
        if aid < 0:
            continue
        idxs = aid2idx.get(aid, [])
        if not idxs:
            continue
        tg_z[t] = z[t, idxs].max()
        tg_sa[t] = np.sqrt(sale[t, idxs]).max()
        tg_up[t] = upar[t, idxs].max()
        tg_re[t] = errVT[idxs, t].max()
    return {
        "tg_z_resid": tg_z,
        "tg_sigma_ale": tg_sa,
        "tg_U_par": tg_up,
        "tg_resid": tg_re,
    }, n_unmapped


def run_for_arrays(arrays_path, tag):
    typ = np.load(TYPEFILE)
    type_cat = typ["type_cat"]
    type_spoof = typ["type_spoof"]
    type_aid = typ["attack_id"]

    names = load_point_names(LISTTXT)
    attacks, meta = load_attacks(TARGETS)

    ch, extras = build_channels(arrays_path)
    label = extras["label"]
    T = extras["T"]

    # sanity: type labels length must match
    if type_cat.shape[0] != T:
        print("[warn] type label length %d != arrays T %d (tag=%s)"
              % (type_cat.shape[0], T, tag))

    tg, n_unmapped = build_targeted_channels(attacks, names, type_aid, extras)
    ch.update(tg)

    chan_order = ["z_resid_top1", "resid_top1", "sigma_ale_max", "U_par_max",
                  "U_dist"]
    if extras["has_ustr"]:
        chan_order += ["U_str_max", "U_str_mean"]
    chan_order += ["sigma_ale_mean", "U_par_mean",
                   "tg_z_resid", "tg_resid", "tg_sigma_ale", "tg_U_par"]
    chan_order = [c for c in chan_order if c in ch]

    anom = type_cat != 0
    spoof_mask = type_spoof == 1
    phys_mask = type_spoof == 2
    A = np.logical_and(anom, spoof_mask)   # sensor spoof
    B = np.logical_and(anom, phys_mask)    # physical/actuator

    res = {"tag": tag, "arrays": arrays_path, "has_ustr": bool(extras["has_ustr"]),
           "n_unmapped_targets": int(n_unmapped),
           "n_spoof": int(A.sum()), "n_phys": int(B.sum()),
           "n_anom": int(anom.sum()), "n_normal": int((~anom).sum())}

    # ---- headline: spoof (A,pos=1) vs physical (B) ----
    sep = {}
    for c in chan_order:
        s = ch[c]
        # label spoof as positive class (1), physical as 0
        yy = np.concatenate([np.ones(A.sum()), np.zeros(B.sum())])
        ss = np.concatenate([s[A], s[B]])
        au = auroc(yy, ss)
        d = cohen_d(s[A], s[B])
        sep[c] = {"AUROC_spoof_vs_phys": au,
                  "rank_biserial": rank_biserial_from_auc(au),
                  "cohen_d_spoof_minus_phys": d,
                  "mean_spoof": float(np.nanmean(s[A])) if A.sum() else float("nan"),
                  "mean_phys": float(np.nanmean(s[B])) if B.sum() else float("nan")}
    res["spoof_vs_phys"] = sep

    # ---- normal vs each category ----
    inv_cat = {1: "SSSP", 2: "SSMP", 3: "MSMP"}
    normal_mask = ~anom
    cat_sep = {}
    for code, nm in inv_cat.items():
        catm = type_cat == code
        if catm.sum() == 0:
            continue
        per = {}
        for c in chan_order:
            s = ch[c]
            yy = np.concatenate([np.ones(catm.sum()), np.zeros(normal_mask.sum())])
            ss = np.concatenate([s[catm], s[normal_mask]])
            per[c] = auroc(yy, ss)
        cat_sep[nm] = per
    res["normal_vs_cat_AUROC"] = cat_sep

    # ---- category vs category (multi-class separability, anomalous only) ----
    # one-vs-one AUROC for the two best channels reported in confusion below
    # ---- transparent-rule confusion matrix (spoof vs physical) ----
    # Rule: decide per anomalous timestep using channel z-scores standardized
    # on the anomalous population. Use the most separating uncertainty channel
    # vs the residual channel.
    cm, rule_acc, rule_desc, rule_detail = transparent_rule_confusion(
        ch, A, B, anom, extras)
    res["rule_confusion"] = cm
    res["rule_accuracy"] = rule_acc
    res["rule_desc"] = rule_desc
    res["rule_detail"] = rule_detail

    return res, ch, extras, (anom, A, B, type_cat, type_spoof)


def _z(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    mu = np.nanmean(x[m]) if m.any() else 0.0
    sd = np.nanstd(x[m]) if m.any() else 1.0
    sd = sd if sd > 0 else 1.0
    out = (x - mu) / sd
    return out


def transparent_rule_confusion(ch, A, B, anom, extras):
    """A transparent, fixed rule mapping anomalous timesteps -> {spoof, phys}.

    KEY EMPIRICAL FINDING that motivates the rule:
      GLOBAL (top-1/max over all V) channels are INVERTED for typing: physical
      attacks raise the *global* residual and *global* uncertainty MORE than
      sensor-spoofs (they perturb the whole plant), so global channels give
      AUROC<0.5 for spoof. The discriminative signal lives in the TARGETED
      channels -- the UQ measured ON THE ATTACKED SENSOR ITSELF:
        - a sensor spoof drives the targeted sensor far OUT of distribution, so
          the model's epistemic disagreement on that sensor (tg_U_par) and the
          targeted standardized residual (tg_z_resid) spike;
        - a physical/actuator attack keeps the targeted sensor in a region the
          ensemble agrees on (low tg_U_par) while the *consequences* show up
          elsewhere in the plant (high global residual).

    Score_spoofness(t) = z(tg_U_par) + z(tg_z_resid)   (both targeted)
    Decide SENSOR-SPOOF if Score_spoofness > 0 else PHYSICAL.
    Standardization is over the anomalous population only; threshold 0 is the
    population mean of the contrast -- transparent, no per-class fitting.
    """
    pop = anom
    if "tg_U_par" in ch and "tg_z_resid" in ch:
        a1 = _z(np.where(pop, ch["tg_U_par"], np.nan))
        a2 = _z(np.where(pop, ch["tg_z_resid"], np.nan))
        score = a1 + a2
        desc = "spoofness = z(tg_U_par) + z(tg_z_resid)  [targeted-sensor channels]"
    else:
        # fallback for backbones without a sensible targeted map: use global
        sa = _z(np.where(pop, ch["sigma_ale_max"], np.nan))
        rs = _z(np.where(pop, ch["resid_top1"], np.nan))
        score = sa - rs
        desc = "spoofness = z(sigma_ale_max) - z(resid_top1)  [global fallback]"

    pred_spoof = score > 0.0   # boolean over all T; only meaningful where A|B

    # confusion on labeled anomalous (spoof=A, phys=B)
    AB = np.logical_or(A, B)
    y_spoof = A[AB]                 # true spoof
    p_spoof = pred_spoof[AB]        # predicted spoof
    tp = int(np.sum(y_spoof & p_spoof))            # spoof->spoof
    fn = int(np.sum(y_spoof & ~p_spoof))           # spoof->phys
    fp = int(np.sum(~y_spoof & p_spoof))           # phys->spoof
    tn = int(np.sum(~y_spoof & ~p_spoof))          # phys->phys
    tot = tp + fn + fp + tn
    acc = float((tp + tn) / tot) if tot else float("nan")
    # balanced accuracy
    rec_spoof = tp / (tp + fn) if (tp + fn) else float("nan")
    rec_phys = tn / (tn + fp) if (tn + fp) else float("nan")
    bal = float(np.nanmean([rec_spoof, rec_phys]))

    cm = {"spoof_as_spoof": tp, "spoof_as_phys": fn,
          "phys_as_spoof": fp, "phys_as_phys": tn}
    detail = {"recall_spoof": rec_spoof, "recall_phys": rec_phys,
              "balanced_accuracy": bal,
              "majority_class_acc": float(max(tp + fn, fp + tn) / tot) if tot else float("nan")}
    return cm, acc, desc, detail


def fmt_sep_table(res):
    lines = []
    sep = res["spoof_vs_phys"]
    lines.append("  channel                AUROC(spoof>phys)  rankbiser  cohen_d   mean_spoof   mean_phys")
    for c, v in sep.items():
        lines.append("  %-22s  %8.3f          %8.3f  %8.3f  %10.4g  %10.4g" %
                     (c, v["AUROC_spoof_vs_phys"], v["rank_biserial"],
                      v["cohen_d_spoof_minus_phys"], v["mean_spoof"], v["mean_phys"]))
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default=os.path.join(ROOT, "results/gdn/ref_seed42/arrays.npz"))
    ap.add_argument("--tag", default="gdn_seed42")
    ap.add_argument("--json_out", default=os.path.join(OUTDIR, "separation_gdn_seed42.json"))
    args = ap.parse_args()

    res, ch, extras, masks = run_for_arrays(args.arrays, args.tag)

    print("==== TYPING SEPARATION : %s ====" % args.tag)
    print("[counts] n_anom=%d n_spoof=%d n_phys=%d n_normal=%d has_ustr=%s unmapped_targets=%d"
          % (res["n_anom"], res["n_spoof"], res["n_phys"], res["n_normal"],
             res["has_ustr"], res["n_unmapped_targets"]))
    print("[HEADLINE spoof vs physical separation]")
    print(fmt_sep_table(res))
    print("[rule] %s" % res["rule_desc"])
    cm = res["rule_confusion"]
    print("  confusion: spoof->spoof=%d spoof->phys=%d phys->spoof=%d phys->phys=%d"
          % (cm["spoof_as_spoof"], cm["spoof_as_phys"], cm["phys_as_spoof"], cm["phys_as_phys"]))
    print("  rule_accuracy=%.4f balanced_acc=%.4f majority_baseline=%.4f"
          % (res["rule_accuracy"], res["rule_detail"]["balanced_accuracy"],
             res["rule_detail"]["majority_class_acc"]))
    print("[normal vs category AUROC]")
    for nm, per in res["normal_vs_cat_AUROC"].items():
        best = max(((c, a) for c, a in per.items() if np.isfinite(a)),
                   key=lambda x: x[1], default=("none", float("nan")))
        print("  %-5s : best channel %s AUROC=%.3f" % (nm, best[0], best[1]))
        # also print z_resid and the uncertainty channels explicitly
        for c in ["z_resid_top1", "resid_top1", "sigma_ale_max", "U_par_max", "U_dist"]:
            if c in per:
                print("        %-16s %.3f" % (c, per[c]))
        if "U_str_max" in per:
            print("        %-16s %.3f" % ("U_str_max", per["U_str_max"]))

    with open(args.json_out, "w") as f:
        json.dump(res, f, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else o)
    print("[saved] %s" % args.json_out)
    print("__DONE__")


if __name__ == "__main__":
    main()
