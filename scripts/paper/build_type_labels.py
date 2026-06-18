#!/usr/bin/env python
"""
Track H - Step 1: Build per-timestep anomaly TYPE labels aligned to the
arrays.npz T index space (T=44716 for GDN seed42).

For each attack in data/swat/attack_targets.json with end_idx>0:
  type_cat[start:end]   <- category in {SSSP, SSMP, MSMP, NONE}
  type_spoof[start:end] <- 'spoof' (actual_change false) or 'physical' (true)
  attack_id[start:end]  <- attack_id

KILL-CHECK: |type!=normal| vs |label==1|, and the overlap fraction
  |type!=normal AND label==1| / |label==1| using results/gdn/ref_seed42 label.
If overlap < 0.9, search a small integer offset in [-300,300] that maximizes
overlap and report it (labels are saved using the BEST offset found).

Outputs (pure ASCII, NEW files only):
  results/paper/typing/type_labels.npz  (type_cat, type_spoof, attack_id, meta)
Prints per-category timestep counts and the kill-check.

Dependency-light: numpy only.
"""
import os, sys, json
import numpy as np

ROOT = "/mnt/datassd3/rashinda/UQ_GNN_AnomalyTyping"
ARRAYS = os.path.join(ROOT, "results/gdn/ref_seed42/arrays.npz")
TARGETS = os.path.join(ROOT, "data/swat/attack_targets.json")
LISTTXT = os.path.join(ROOT, "data/swat/list.txt")
OUTDIR = os.path.join(ROOT, "results/paper/typing")

CAT_CODES = {"normal": 0, "SSSP": 1, "SSMP": 2, "MSMP": 3, "NONE": 4}
SPOOF_CODES = {"normal": 0, "spoof": 1, "physical": 2}


def load_point_names(path):
    names = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                names.append(s)
    return names


def load_attacks(path):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for key in ("attacks", "attack_targets", "data", "items"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        vals = list(obj.values())
        if vals and isinstance(vals[0], dict):
            return vals
    if isinstance(obj, list):
        return obj
    raise ValueError("Unrecognized attack_targets.json structure")


def build_labels(attacks, T, offset=0):
    cat = np.zeros(T, dtype=np.int16)
    spoof = np.zeros(T, dtype=np.int16)
    aid = np.full(T, -1, dtype=np.int32)
    for a in attacks:
        end = int(a.get("end_idx", 0) or 0)
        start = int(a.get("start_idx", 0) or 0)
        if end <= 0:
            continue
        s = max(0, start + offset)
        e = min(T, end + offset)
        if e <= s:
            continue
        category = str(a.get("category", "NONE")) or "NONE"
        ccode = CAT_CODES.get(category, CAT_CODES["NONE"])
        ac = a.get("actual_change", None)
        if ac is True:
            scode = SPOOF_CODES["physical"]
        elif ac is False:
            scode = SPOOF_CODES["spoof"]
        else:
            scode = SPOOF_CODES["spoof"]
        aval = int(a.get("attack_id", -1))
        cat[s:e] = ccode
        spoof[s:e] = scode
        aid[s:e] = aval
    return cat, spoof, aid


def overlap_stats(cat_code, label):
    nz = cat_code != 0
    lab = label.astype(bool)
    inter = int(np.logical_and(nz, lab).sum())
    lab_n = int(lab.sum())
    nz_n = int(nz.sum())
    frac = float(inter) / float(lab_n) if lab_n > 0 else 0.0
    return frac, inter, nz_n, lab_n


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = np.load(ARRAYS)
    label = d["test_attack_label"].astype(np.int64)
    T = label.shape[0]
    names = load_point_names(LISTTXT)
    attacks = load_attacks(TARGETS)

    n_with_end = sum(1 for a in attacks if int(a.get("end_idx", 0) or 0) > 0)
    print("[info] T=%d  label.sum=%d  attack_rate=%.4f" %
          (T, int(label.sum()), float(label.mean())))
    print("[info] n_attacks_total=%d  n_with_end_idx_gt0=%d  n_point_names=%d" %
          (len(attacks), n_with_end, len(names)))

    cat0, spoof0, aid0 = build_labels(attacks, T, offset=0)
    frac0, inter0, nz0, labn0 = overlap_stats(cat0, label)
    print("[killcheck off=0] overlap_frac=%.4f inter=%d |type!=normal|=%d |label==1|=%d"
          % (frac0, inter0, nz0, labn0))

    best_off = 0
    best_frac = frac0
    if frac0 < 0.90:
        print("[killcheck] overlap < 0.90 -> searching offset in [-300,300] ...")
        for off in range(-300, 301):
            c, _, _ = build_labels(attacks, T, offset=off)
            fr, _, _, _ = overlap_stats(c, label)
            if fr > best_frac:
                best_frac = fr
                best_off = off
        print("[killcheck] best_offset=%d best_overlap_frac=%.4f" %
              (best_off, best_frac))
    else:
        print("[killcheck] overlap >= 0.90 at offset 0 -> alignment OK, offset=0")

    cat, spoof, aid = build_labels(attacks, T, offset=best_off)
    frac, inter, nz, labn = overlap_stats(cat, label)
    prec = float(inter) / float(nz) if nz > 0 else 0.0
    jac = float(inter) / float(nz + labn - inter) if (nz + labn - inter) > 0 else 0.0
    print("[final off=%d] overlap_recall=%.4f precision=%.4f |type!=normal|=%d |label==1|=%d jaccard=%.4f"
          % (best_off, frac, prec, nz, labn, jac))

    inv_cat = {v: k for k, v in CAT_CODES.items()}
    inv_spoof = {v: k for k, v in SPOOF_CODES.items()}
    print("[per-category timestep counts]")
    for code in sorted(inv_cat):
        print("   cat %-7s (code %d): %d" % (inv_cat[code], code, int((cat == code).sum())))
    print("[per-spoof timestep counts]")
    for code in sorted(inv_spoof):
        print("   spoof %-9s (code %d): %d" % (inv_spoof[code], code, int((spoof == code).sum())))

    n_distinct = len(set(int(x) for x in np.unique(aid) if x >= 0))
    print("[info] distinct attack_ids present in labels: %d" % n_distinct)

    outpath = os.path.join(OUTDIR, "type_labels.npz")
    np.savez_compressed(
        outpath,
        type_cat=cat.astype(np.int16),
        type_spoof=spoof.astype(np.int16),
        attack_id=aid.astype(np.int32),
        offset=np.array([best_off], dtype=np.int32),
        overlap_frac=np.array([frac], dtype=np.float64),
        cat_codes_keys=np.array(list(CAT_CODES.keys())),
        cat_codes_vals=np.array(list(CAT_CODES.values()), dtype=np.int16),
        spoof_codes_keys=np.array(list(SPOOF_CODES.keys())),
        spoof_codes_vals=np.array(list(SPOOF_CODES.values()), dtype=np.int16),
    )
    print("[saved] %s" % outpath)

    summ = {
        "T": int(T),
        "label_sum": int(label.sum()),
        "n_attacks_total": int(len(attacks)),
        "n_with_end_idx": int(n_with_end),
        "offset_used": int(best_off),
        "overlap_recall": float(frac),
        "overlap_precision": float(prec),
        "n_type_nonnormal": int(nz),
        "n_distinct_attacks": int(n_distinct),
        "cat_counts": {inv_cat[c]: int((cat == c).sum()) for c in sorted(inv_cat)},
        "spoof_counts": {inv_spoof[c]: int((spoof == c).sum()) for c in sorted(inv_spoof)},
    }
    with open(os.path.join(OUTDIR, "type_labels_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("[saved] %s" % os.path.join(OUTDIR, "type_labels_summary.json"))
    print("__DONE__")


if __name__ == "__main__":
    main()
