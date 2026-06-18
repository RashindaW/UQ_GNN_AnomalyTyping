"""Build a REAL distributional/OOD channel (Omega) for the G-DeltaUQ pipeline.

The shipped inference (inference_gdeltauq.run_inference) sets
    U_dist = U_par.mean(dim=-1)
which is a PLACEHOLDER and is ~chance (AUROC ~0.50) against attacks. This script
computes a genuine OOD signal on the anchor-averaged penultimate representation
h_bar by fitting a per-node Gaussian on TRAIN (all-normal) reps and scoring TEST
reps with a per-node Mahalanobis distance (Omega). A kNN-distance OOD variant is
also computed as a cross-check. The results are spliced into a NEW arrays file
(arrays_omega.npz) alongside the existing keys; the original placeholder
test_U_dist and all other keys are preserved.

This module is split into two layers:
  (1) Pure-array math (fit / score / knn) -- numpy/torch only, no repo imports.
      These are unit-testable on CPU with synthetic data (see test_omega_math.py).
  (2) GPU pipeline glue (build datasets, load the model, run inference, splice).
      Heavy repo imports happen lazily inside these functions so that importing
      this module for the CPU math test never pulls in the GPU model code. The
      glue mirrors scripts/detect_gdeltauq.py: same _build_test_dataset, same
      GDN_GDeltaUQ construction, same AleatoricHead + anchor_pool/q_v/u_bar_norm
      bundle loading, and the same LoadedGDeltaUQ wrapper.

CLI:
  python scripts/paper/build_omega.py \
      --checkpoint <best_*.pt> --hyperparameters <hp.json> \
      --bundle_dir <calibration_bundle_K100> --split <gdeltauq_split.json> \
      --in_arrays <results/.../arrays.npz> --out_arrays <results/.../arrays_omega.npz> \
      --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Make the repo root importable when run as a script from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Layer 1: pure-array OOD math (CPU-testable; no repo / GPU imports)
# ---------------------------------------------------------------------------

def fit_per_node_gaussian(train_hbar, eps_reg=1e-3):
    """Fit a per-node Gaussian on TRAIN penultimate reps.

    Mirrors util/ood.fit_mahalanobis math but operates directly on an array.
    util/ood divides by T; we use the same convention to stay consistent.

    Parameters
    ----------
    train_hbar : array-like (T_tr, V, d)
    eps_reg    : float, diagonal regulariser added to each per-node covariance.

    Returns
    -------
    mean_v   : np.ndarray (V, d)
    inv_cov_v: np.ndarray (V, d, d)
    """
    x = np.asarray(train_hbar, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, None, :]  # (T,1,d)
    T, V, d = x.shape
    mean_v = x.mean(axis=0)  # (V,d)
    centred = x - mean_v[None]  # (T,V,d)
    # cov[v,i,j] = (1/T) sum_t centred[t,v,i] centred[t,v,j]   (matches util/ood)
    cov = np.einsum('tvi,tvj->vij', centred, centred) / max(T, 1)
    inv_cov_v = np.empty((V, d, d), dtype=np.float64)
    eye = np.eye(d, dtype=np.float64)
    for v in range(V):
        inv_cov_v[v] = np.linalg.inv(cov[v] + eps_reg * eye)
    return mean_v.astype(np.float64), inv_cov_v


def score_mahalanobis_arrays(test_hbar, mean_v, inv_cov_v):
    """Per-node Mahalanobis distance Omega_v(t) = sqrt((phi-mean)^T inv_cov (phi-mean)).

    Mirrors util/ood.score_mahalanobis (which returns the same sqrt'd distance)
    but works on arrays.

    Parameters
    ----------
    test_hbar : array-like (T, V, d)
    mean_v    : (V, d)
    inv_cov_v : (V, d, d)

    Returns
    -------
    omega_pernode : np.ndarray (T, V)  -- per-node Mahalanobis distance (>=0)
    """
    x = np.asarray(test_hbar, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, None, :]
    mean_v = np.asarray(mean_v, dtype=np.float64)
    inv_cov_v = np.asarray(inv_cov_v, dtype=np.float64)
    centred = x - mean_v[None]  # (T,V,d)
    tmp = np.einsum('tvi,vij->tvj', centred, inv_cov_v)
    quad = np.einsum('tvj,tvj->tv', tmp, centred)  # (T,V) squared Mahalanobis
    quad = np.clip(quad, 0.0, None)  # guard tiny negatives from numerics
    return np.sqrt(quad)


def knn_omega(train_hbar, test_hbar, k=10, max_train=5000, seed=0):
    """kNN-distance OOD cross-check on the flattened (over V) penultimate rep.

    For each test timestep, flatten h_bar over nodes -> (V*d,) and compute the
    Euclidean distance to its k-th nearest TRAIN point. Train is subsampled to
    at most ``max_train`` points for tractability.

    Returns
    -------
    omega_knn : np.ndarray (T,)  -- distance to k-th nearest train point.
    """
    import torch

    tr = np.asarray(train_hbar, dtype=np.float32)
    te = np.asarray(test_hbar, dtype=np.float32)
    if tr.ndim == 3:
        tr = tr.reshape(tr.shape[0], -1)  # (T_tr, V*d)
    if te.ndim == 3:
        te = te.reshape(te.shape[0], -1)  # (T, V*d)

    Ntr = tr.shape[0]
    if Ntr > max_train:
        rng = np.random.default_rng(seed)
        sel = rng.choice(Ntr, size=max_train, replace=False)
        tr = tr[sel]
    kk = min(k, tr.shape[0])

    tr_t = torch.as_tensor(tr, dtype=torch.float32)
    te_t = torch.as_tensor(te, dtype=torch.float32)
    out = np.zeros((te_t.shape[0],), dtype=np.float64)
    chunk = 1024  # bound the (chunk, Ntr) distance-matrix memory
    with torch.no_grad():
        for i in range(0, te_t.shape[0], chunk):
            d = torch.cdist(te_t[i:i + chunk], tr_t)  # (c, Ntr)
            kth = d.topk(kk, dim=1, largest=False).values[:, -1]  # (c,)
            out[i:i + chunk] = kth.cpu().numpy()
    return out


def compute_omega_from_hbar(train_hbar, test_hbar, eps_reg=1e-3, k=10, max_train=5000, seed=0):
    """Fit on train h_bar, score test h_bar; return all Omega variants.

    Returns dict with keys:
      omega_pernode (T,V), omega_max (T,), omega_mean (T,), omega_knn (T,)
    """
    mean_v, inv_cov_v = fit_per_node_gaussian(train_hbar, eps_reg=eps_reg)
    omega_pernode = score_mahalanobis_arrays(test_hbar, mean_v, inv_cov_v)  # (T,V)
    omega_max = omega_pernode.max(axis=-1)
    omega_mean = omega_pernode.mean(axis=-1)
    omega_knn = knn_omega(train_hbar, test_hbar, k=k, max_train=max_train, seed=seed)
    return {
        "omega_pernode": omega_pernode,
        "omega_max": omega_max,
        "omega_mean": omega_mean,
        "omega_knn": omega_knn,
    }


def safe_auroc(scores, labels):
    """AUROC of higher-score == anomaly. Returns nan if a class is missing or
    sklearn is unavailable (so validation printing never crashes the run)."""
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.min() == y.max():
        return float("nan")
    if not np.all(np.isfinite(s)):
        finite = s[np.isfinite(s)]
        fill = float(np.nanmax(finite)) if finite.size else 0.0
        s = np.nan_to_num(s, nan=fill, posinf=fill, neginf=0.0)
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------------------
# Layer 2: GPU pipeline glue (lazy repo imports happen INSIDE these functions)
# Mirrors scripts/detect_gdeltauq.py exactly for the model/bundle/dataset path.
# ---------------------------------------------------------------------------

def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _build_dataset(dataset_name, csv_name, slide_win, row_range=None):
    """Build a TimeDataset from data/<dataset_name>/<csv_name>, optionally
    restricted to a ROW range using detect_gdeltauq's window->row Subset logic.

    Reuses the exact preprocessing helpers from the project so the feature map,
    edge index and windowing match training/detection. Returns
    (dataset, feature_map, fc_edge_index).
    """
    import pandas as pd
    import torch
    from torch.utils.data import Subset

    from datasets.TimeDataset import TimeDataset
    from util.net_struct import get_feature_map, get_fc_graph_struc
    from util.preprocess import build_loc_net, construct_data

    df = pd.read_csv(f'./data/{dataset_name}/{csv_name}', sep=',', index_col=0)
    feature_map = get_feature_map(dataset_name)
    fc_struc = get_fc_graph_struc(dataset_name)
    cols_no_attack = [c for c in df.columns if c != 'attack']
    fc_edge_index = build_loc_net(fc_struc, cols_no_attack, feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

    attack_col = df['attack'].tolist() if 'attack' in df.columns else 0
    indata = construct_data(df, feature_map, labels=attack_col)
    cfg = {'slide_win': slide_win, 'slide_stride': 1}
    ds = TimeDataset(indata, fc_edge_index, mode='test', config=cfg)
    if row_range is not None:
        r0, r1 = row_range
        w0 = max(0, r0 - slide_win)
        w1 = min(len(ds), r1 - slide_win)
        ds = Subset(ds, list(range(w0, w1)))
    return ds, feature_map, fc_edge_index


def _build_loaded(checkpoint, hp, bundle_dir, fc_edge_index, V, device):
    """Construct LoadedGDeltaUQ exactly as scripts/detect_gdeltauq.py does."""
    import numpy as _np
    import torch
    from pathlib import Path

    from inference_gdeltauq import LoadedGDeltaUQ
    from models.GDN_GDeltaUQ import GDN_GDeltaUQ
    from models.aleatoric_head import AleatoricHead

    bundle_dir = Path(bundle_dir)

    model = GDN_GDeltaUQ(
        [fc_edge_index], V,
        dim=int(hp['dim']),
        input_dim=int(hp['slide_win']),
        out_layer_num=int(hp['out_layer_num']),
        out_layer_inter_dim=int(hp['out_layer_inter_dim']),
        topk=int(hp['topk']),
        n_gnn_layers=int(hp['n_gnn_layers']),
        use_learnable_adj=bool(hp.get('use_learnable_adj', 0)),
        lsa_tau=float(hp.get('lsa_tau', 1.0)),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    anchor_pool = torch.load(bundle_dir / 'anchor_pool.pt', map_location='cpu')
    aleatoric_head = AleatoricHead(
        hidden_dim=int(hp['dim']), num_sensors=V,
        sensor_embed_dim=16, mlp_hidden=64,
    )
    aleatoric_head.load_state_dict(torch.load(bundle_dir / 'aleatoric_head.pt',
                                              map_location='cpu'))
    aleatoric_head.to(device).eval()

    q_v = _np.load(bundle_dir / 'q_v.npz')['q_v']
    u_norm = _np.load(bundle_dir / 'u_bar_norm.npz')
    u_bar = {
        'U_par': float(u_norm['U_par']),
        'U_str': float(u_norm['U_str']),
        'U_dist': float(u_norm['U_dist']),
    }

    # cfg must expose 'topk' and 'dim' for run_inference; hp has both.
    loaded = LoadedGDeltaUQ(
        model=model, aleatoric_head=aleatoric_head, anchor_pool=anchor_pool,
        q_v=q_v, u_bar_norm=u_bar, feature_map=list(range(V)), cfg=hp, device=device,
    )
    return loaded


def run_pipeline(checkpoint, hyperparameters, bundle_dir, split_path, in_arrays,
                 out_arrays, device, batch_size=None,
                 eps_reg=1e-3, k=10, max_train=5000, seed=0):
    """Full GPU path: load model, run inference on TRAIN+TEST h_bar, compute
    Omega variants, splice into a NEW arrays file, print validation AUROCs.

    Requires the trained checkpoint/bundle and the SWaT csvs and runs the
    G-DeltaUQ model on the GPU. Intended to be launched by a human; it is NOT
    exercised by the CPU unit tests (those call the Layer-1 math directly).
    """
    import torch  # noqa: F401
    from inference_gdeltauq import run_inference
    from util.env import set_device, get_device

    hp = _read_json(hyperparameters)
    dataset_name = hp['dataset']
    slide_win = int(hp['slide_win'])
    if batch_size is None:
        batch_size = int(hp.get('batch', 64))
    split = _read_json(split_path) if split_path else None

    set_device(device)
    dev = get_device()

    # TRAIN reps (all-normal) for the Gaussian / kNN reference set, restricted to
    # the split's train ROW range (default [0, train_end]).
    train_range = None
    if split is not None and 'train_rows' in split:
        train_range = tuple(split['train_rows'])
    ds_tr, feature_map, fc_edge_index = _build_dataset(
        dataset_name, 'train.csv', slide_win, row_range=train_range)
    V = len(feature_map)

    loaded = _build_loaded(checkpoint, hp, bundle_dir, fc_edge_index, V, dev)

    out_tr = run_inference(loaded, ds_tr, batch_size=batch_size)
    train_hbar = np.asarray(out_tr.h_bar)  # (T_tr, V, d)

    # TEST reps for scoring (full test set, no row restriction).
    ds_te, _, _ = _build_dataset(dataset_name, 'test.csv', slide_win, row_range=None)
    out_te = run_inference(loaded, ds_te, batch_size=batch_size)
    test_hbar = np.asarray(out_te.h_bar)   # (T, V, d)
    test_attack_label = np.asarray(out_te.attack_label)

    omega = compute_omega_from_hbar(
        train_hbar, test_hbar, eps_reg=eps_reg, k=k, max_train=max_train, seed=seed
    )

    splice_and_save(in_arrays, out_arrays, omega,
                    fallback_attack_label=test_attack_label)
    print_validation(out_arrays)
    return out_arrays


def splice_and_save(in_arrays, out_arrays, omega, fallback_attack_label=None):
    """Load the existing arrays.npz, add the Omega keys, KEEP every original key
    (including the placeholder test_U_dist), and write to out_arrays.

    Added keys:
      test_U_dist_maha_max     (T,)
      test_U_dist_maha_mean    (T,)
      test_U_dist_maha_pernode (T,V)
      test_U_dist_knn          (T,)
    """
    base = {}
    if in_arrays is not None and os.path.exists(in_arrays):
        z = np.load(in_arrays, allow_pickle=True)
        for kk in z.files:
            base[kk] = z[kk]
    elif fallback_attack_label is not None:
        base["test_attack_label"] = np.asarray(fallback_attack_label)

    base["test_U_dist_maha_max"] = omega["omega_max"].astype(np.float32)
    base["test_U_dist_maha_mean"] = omega["omega_mean"].astype(np.float32)
    base["test_U_dist_maha_pernode"] = omega["omega_pernode"].astype(np.float32)
    base["test_U_dist_knn"] = omega["omega_knn"].astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out_arrays)), exist_ok=True)
    np.savez_compressed(out_arrays, **base)
    print("wrote " + out_arrays + " (" + str(len(base)) + " keys)")


def print_validation(out_arrays):
    """Print AUROC of each Omega variant vs the binary attack label, and the
    placeholder U_dist AUROC for comparison. A real Omega should beat ~0.50."""
    z = np.load(out_arrays, allow_pickle=True)
    if "test_attack_label" not in z.files:
        print("[validation] no test_attack_label in arrays; skipping AUROC")
        return
    y = z["test_attack_label"]
    print("[validation] AUROC vs attack_label (higher score == anomaly):")
    if "test_U_dist" in z.files:
        ud = z["test_U_dist"]
        ud = ud if ud.ndim == 1 else ud.reshape(ud.shape[0], -1).mean(-1)
        print("  placeholder test_U_dist          : %.4f" % safe_auroc(ud, y))
    else:
        print("  placeholder test_U_dist          : (absent in source arrays)")
    print("  omega maha_max                    : %.4f" % safe_auroc(z["test_U_dist_maha_max"], y))
    print("  omega maha_mean                   : %.4f" % safe_auroc(z["test_U_dist_maha_mean"], y))
    print("  omega knn                         : %.4f" % safe_auroc(z["test_U_dist_knn"], y))


def main():
    ap = argparse.ArgumentParser(description="Build a real Omega OOD channel and splice into arrays.npz")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hyperparameters", required=True)
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--in_arrays", required=True)
    ap.add_argument("--out_arrays", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--eps_reg", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max_train", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_pipeline(
        checkpoint=args.checkpoint,
        hyperparameters=args.hyperparameters,
        bundle_dir=args.bundle_dir,
        split_path=args.split,
        in_arrays=args.in_arrays,
        out_arrays=args.out_arrays,
        device=args.device,
        batch_size=args.batch_size,
        eps_reg=args.eps_reg,
        k=args.k,
        max_train=args.max_train,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
