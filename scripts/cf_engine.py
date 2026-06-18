"""Static-GNN counterfactual engine: single-window masked CF query.

Given a frozen G-DeltaUQ + M10 stack and a candidate (edge_mask, node_mask)
intervention on the learned graph, re-runs forward_split + K=100
forward_anchored with the masked graph, rebuilds the 8 M10 features,
and returns M10's log-odds.

Design:
- Setup is one-shot (build_cf_context): loads checkpoint, anchor pool,
  aleatoric head; rebuilds normalisation statistics from the cached
  arrays.npz; trains M10 on val_slice (matches FROZEN_HPS); caches the
  unmasked batch_gated and the mapping from edge_index_sample [0, E) to
  columns in batch_gated [0, topk*V).
- Each cf_engine(...) call monkey-patches model._build_learned_graph to
  return a masked batch_gated for the duration of one query, runs the
  K=100 anchored loop, and returns the log-odds.

The empty-mask identity test (verification 1 of the plan) must pass to
within 1e-3 before any masked queries are trusted.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from datasets.TimeDataset import TimeDataset
from models.GDN_GDeltaUQ import GDN_GDeltaUQ
from models.aleatoric_head import AleatoricHead
from models.causal_mask import load_causal_mask
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.preprocess import build_loc_net, construct_data

EPS_FEAT = 1e-6
EPS_RESID = 1e-2
SIGMA_FLOOR = 1e-12
BEFORE_NUM_SMOOTH = 5  # paper protocol: smoothing=5 in setup_context


# --------------------------------------------------------------------------- #
# z-score / normalisation helpers (mirror fusion_sweep_K100_full.py)
# --------------------------------------------------------------------------- #

def _fit_zscore_params(x: np.ndarray, mask: np.ndarray | None = None):
    s = x[mask] if mask is not None else x
    med = np.median(s, axis=0)
    q25 = np.quantile(s, 0.25, axis=0)
    q75 = np.quantile(s, 0.75, axis=0)
    iqr = q75 - q25
    return med, iqr


def _apply_zscore(x: np.ndarray, med, iqr) -> np.ndarray:
    return (x - med) / (iqr + EPS_FEAT)


def _fit_1d_renorm_params(s: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    nominal = s[mask].astype(np.float64)
    med = float(np.median(nominal))
    q25 = float(np.quantile(nominal, 0.25))
    q75 = float(np.quantile(nominal, 0.75))
    return med, q75 - q25


def _apply_1d_renorm(s: np.ndarray, med: float, iqr: float) -> np.ndarray:
    return (s - med) / (iqr + EPS_FEAT)


# --------------------------------------------------------------------------- #
# Residual machinery (mirror sweep_eval_gdeltauq.smoothed_err_scores)
# --------------------------------------------------------------------------- #

def _get_val_err_median_iqr(val_mu: np.ndarray, val_gt: np.ndarray):
    """Per-feature median + IQR of |val_mu - val_gt|. Matches
    util.data.get_err_median_and_iqr semantics."""
    delta = np.abs(val_mu - val_gt)
    med = np.median(delta, axis=0)
    q25 = np.quantile(delta, 0.25, axis=0)
    q75 = np.quantile(delta, 0.75, axis=0)
    iqr = q75 - q25
    return med, iqr


def _per_feature_err_score(mu: np.ndarray, y: np.ndarray,
                           n_err_mid: np.ndarray, n_err_iqr: np.ndarray):
    """err_scores = (|mu - y| - n_err_mid) / (|n_err_iqr| + eps).
    Shape is preserved (single timestep -> (V,))."""
    delta = np.abs(mu - y)
    return (delta - n_err_mid) / (np.abs(n_err_iqr) + EPS_RESID)


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #

def _build_test_dataset(dataset_name: str, slide_win: int):
    test_csv = pd.read_csv(f'./data/{dataset_name}/test.csv', sep=',', index_col=0)
    feature_map = get_feature_map(dataset_name)
    fc_struc = get_fc_graph_struc(dataset_name)
    cols_no_attack = [c for c in test_csv.columns if c != 'attack']
    fc_edge_index = build_loc_net(fc_struc, cols_no_attack, feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    attack_col = test_csv['attack'].tolist() if 'attack' in test_csv.columns else 0
    indata = construct_data(test_csv, feature_map, labels=attack_col)
    cfg = {'slide_win': slide_win, 'slide_stride': 1}
    ds = TimeDataset(indata, fc_edge_index, mode='test', config=cfg)
    return ds, feature_map, fc_edge_index


def _build_train_dataset(dataset_name: str, slide_win: int):
    train_csv = pd.read_csv(f'./data/{dataset_name}/train.csv', sep=',', index_col=0)
    if 'attack' in train_csv.columns:
        train_csv = train_csv.drop(columns=['attack'])
    feature_map = get_feature_map(dataset_name)
    fc_struc = get_fc_graph_struc(dataset_name)
    fc_edge_index = build_loc_net(fc_struc, list(train_csv.columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    indata = construct_data(train_csv, feature_map, labels=0)
    cfg = {'slide_win': slide_win, 'slide_stride': 1}
    ds = TimeDataset(indata, fc_edge_index, mode='test', config=cfg)
    return ds, feature_map, fc_edge_index


# --------------------------------------------------------------------------- #
# Context builder
# --------------------------------------------------------------------------- #

@dataclass
class CFContext:
    """All frozen artefacts needed by cf_engine()."""
    # Model / inference
    model: torch.nn.Module
    aleatoric_head: torch.nn.Module
    anchor_pool: torch.Tensor          # (K, V, d)
    device: torch.device
    K: int
    V: int
    E: int                             # non-self edge count from edge_index_sample
    slide_win: int

    # Test dataset (windows on demand)
    test_ds: TimeDataset

    # Cached arrays (full test arrays)
    test_mu_bar: np.ndarray            # (T, V)
    test_gt: np.ndarray                # (T, V)
    test_U_par: np.ndarray             # (T, V)
    test_U_str: np.ndarray             # (T, E)
    test_U_dist: np.ndarray            # (T,)
    test_sigma2_ale: np.ndarray        # (T, V)
    test_label: np.ndarray             # (T,)
    test_err_scores: np.ndarray        # (T, V) unsmoothed per-feature err score

    # Validation residual stats (for re-normalising on CF queries)
    n_err_mid: np.ndarray              # (V,)
    n_err_iqr: np.ndarray              # (V,)

    # C-slice z-score stats
    med_par: np.ndarray
    iqr_par: np.ndarray
    med_str: np.ndarray                # (E,)
    iqr_str: np.ndarray                # (E,)
    med_sig: np.ndarray
    iqr_sig: np.ndarray
    med_dist: float
    iqr_dist: float
    med_agg: float
    iqr_agg: float
    log_sigma_tot_max_med: float
    log_sigma_tot_max_iqr: float
    # 1d renorm of the 5 z-signals + agg_z
    renorm_params: dict                # {name: (med, iqr)}

    # M10
    m10: HistGradientBoostingClassifier
    m10_tau_star: float                # detector threshold for reporting
    s_M10_full: np.ndarray             # (T,) cached log-odds from the unmasked forward
    feat_full: np.ndarray              # (T, 8) cached feature matrix

    # Learned graph cache
    unmasked_batch_gated: torch.Tensor  # (2, topk*V)
    broadcast_embeddings: torch.Tensor  # (V, d)
    non_self_indices: np.ndarray       # (E,) column indices in batch_gated
    edge_index_sample: np.ndarray      # (2, E)

    # Indices
    c_mask_nominal: np.ndarray         # (T,) bool
    val_idx: np.ndarray                # arrays indices for M10 training
    T: int

    # Topology
    topk: int


def build_cf_context(
    arrays_path: str,
    checkpoint_path: str,
    hp_path: str,
    bundle_dir: str,
    cal_split_path: str,
    dataset_name: str = 'swat',
    device: str = 'cuda:0',
    m10_tau_star: float | None = None,
) -> CFContext:
    """One-shot setup. Loads everything and freezes."""
    with open(hp_path) as f:
        hp = json.load(f)
    slide_win = int(hp['slide_win'])
    topk = int(hp['topk'])
    dim = int(hp['dim'])

    dev = torch.device(device)

    # ---- load cached arrays ----
    print(f'[cf_engine] loading arrays from {arrays_path}', flush=True)
    d = np.load(arrays_path)
    test_mu_bar = d['test_mu_bar'].astype(np.float64)
    test_gt = d['test_ground_truth'].astype(np.float64)
    test_U_par = d['test_U_par'].astype(np.float64)
    test_U_str = d['test_U_str'].astype(np.float64)
    test_U_dist = d['test_U_dist'].astype(np.float64)
    test_sigma2_ale = d['test_sigma2_ale'].astype(np.float64)
    val_mu = d['val_mu_bar'].astype(np.float64)
    val_gt = d['val_ground_truth'].astype(np.float64)
    test_label = d['test_attack_label'].astype(np.int8)
    T, V = test_mu_bar.shape
    E = test_U_str.shape[1]
    print(f'[cf_engine] T={T} V={V} E={E}', flush=True)

    # ---- residual machinery ----
    n_err_mid, n_err_iqr = _get_val_err_median_iqr(val_mu, val_gt)
    test_err = (np.abs(test_mu_bar - test_gt) - n_err_mid) / (np.abs(n_err_iqr) + EPS_RESID)

    # ---- C/val indices ----
    with open(cal_split_path) as f:
        cal_split = json.load(f)
    C_lo, C_hi = cal_split['C_row_range']
    val_lo, val_hi = cal_split['labeled_val_range']
    C_idx_start = max(0, C_lo - slide_win)
    C_idx_end = min(T, max(0, C_hi - slide_win))
    val_idx_start = max(0, val_lo - slide_win)
    val_idx_end = min(T, max(0, val_hi - slide_win))
    c_idx = np.arange(C_idx_start, C_idx_end)
    val_idx = np.arange(val_idx_start, val_idx_end)
    c_mask = np.zeros(T, dtype=bool)
    c_mask[c_idx] = True
    c_mask_nominal = c_mask & (test_label == 0)
    print(f'[cf_engine] C   slice arrays[{C_idx_start},{C_idx_end}) n={C_idx_end-C_idx_start}',
          flush=True)
    print(f'[cf_engine] val slice arrays[{val_idx_start},{val_idx_end}) n={val_idx_end-val_idx_start}',
          flush=True)

    # ---- C-slice z-score stats ----
    med_par, iqr_par = _fit_zscore_params(test_U_par, c_mask_nominal)
    med_str, iqr_str = _fit_zscore_params(test_U_str, c_mask_nominal)
    med_sig, iqr_sig = _fit_zscore_params(test_sigma2_ale, c_mask_nominal)
    med_dist, iqr_dist = _fit_zscore_params(test_U_dist[:, None], c_mask_nominal)
    med_dist, iqr_dist = float(med_dist[0]), float(iqr_dist[0])

    # ---- build cached aggregates (full arrays) to fit 1d renorms ----
    smoothed = np.zeros_like(test_err)
    for i in range(BEFORE_NUM_SMOOTH, T):
        smoothed[i] = test_err[i - BEFORE_NUM_SMOOTH:i + 1].mean(axis=0)
    agg_full = smoothed.max(axis=1)  # top-1 aggregate per timestep
    med_agg, iqr_agg = _fit_1d_renorm_params(agg_full, c_mask_nominal)

    z_U_par_TxV = _apply_zscore(test_U_par, med_par, iqr_par)
    z_U_str_TxE = _apply_zscore(test_U_str, med_str, iqr_str)
    z_sigma2_TxV = _apply_zscore(test_sigma2_ale, med_sig, iqr_sig)
    z_U_dist = _apply_zscore(test_U_dist[:, None], med_dist, iqr_dist)[:, 0]
    raw_full = {
        'U_par_max_v':    z_U_par_TxV.max(axis=1),
        'U_par_mean_v':   z_U_par_TxV.mean(axis=1),
        'U_str_mean_e':   z_U_str_TxE.mean(axis=1),
        'U_dist':         z_U_dist,
        'sigma_ale_max_v': z_sigma2_TxV.max(axis=1),
    }
    renorm_params = {k: _fit_1d_renorm_params(v, c_mask_nominal)
                     for k, v in raw_full.items()}
    signals_full = {k: _apply_1d_renorm(raw_full[k], *renorm_params[k])
                    for k in raw_full}
    agg_z_full = _apply_1d_renorm(agg_full, med_agg, iqr_agg)

    # 8th feature: z(log(sqrt(sigma2_ale + U_par)).max_v)
    sigma_tot_full = np.sqrt(np.maximum(test_sigma2_ale, SIGMA_FLOOR)
                              + np.maximum(test_U_par, 0.0))
    log_sigma_tot_max_full = np.log(sigma_tot_full.max(axis=1) + EPS_FEAT)
    log_sigma_tot_max_med, log_sigma_tot_max_iqr = _fit_zscore_params(
        log_sigma_tot_max_full[:, None], c_mask_nominal)
    log_sigma_tot_max_med = float(log_sigma_tot_max_med[0])
    log_sigma_tot_max_iqr = float(log_sigma_tot_max_iqr[0])

    # ---- train M10 (deployed HPs from FROZEN_HPS / RESEARCH_NOTE) ----
    feat_full = np.column_stack([
        agg_z_full,
        signals_full['U_par_max_v'],
        signals_full['U_par_mean_v'],
        signals_full['sigma_ale_max_v'],
        signals_full['U_str_mean_e'],
        signals_full['U_dist'],
        signals_full['U_par_max_v'] * agg_z_full,
        (log_sigma_tot_max_full - log_sigma_tot_max_med) / (log_sigma_tot_max_iqr + EPS_FEAT),
    ])
    print('[cf_engine] training M10 (depth=5, max_iter=50, lr=0.05, l2=1.0, balanced, '
          'seed=42, train=val_slice)', flush=True)
    m10 = HistGradientBoostingClassifier(
        max_depth=5, max_iter=50, learning_rate=0.05,
        l2_regularization=1.0, random_state=42, class_weight='balanced',
    )
    m10.fit(feat_full[val_idx], test_label[val_idx])
    proba_full = m10.predict_proba(feat_full)[:, 1]
    s_M10_full = np.log(np.clip(proba_full, 1e-8, 1 - 1e-8)
                         / np.clip(1 - proba_full, 1e-8, 1 - 1e-8))
    # tau* is model-specific (M10 is retrained per-model). Auto-compute the
    # Fix-A post-proc-aware best threshold on THIS model's s_M10 when not
    # supplied, rather than reusing the baseline 2.0914.
    if m10_tau_star is None:
        from sweep_postproc_threshold import best_threshold_postproc_aware
        best_thr, _ = best_threshold_postproc_aware(
            s_M10_full, test_label, 5, 5, n_taus=400)
        m10_tau_star = float(best_thr['tau'])
        print(f'[cf_engine] auto tau* = {m10_tau_star:.4f} '
              f'(Fix-A best F1={best_thr["F1"]:.4f})', flush=True)
    print(f'[cf_engine] M10 trained. proba mean={proba_full.mean():.4f} '
          f's_M10 at tau* {m10_tau_star:.4f}: '
          f'#fired={int((s_M10_full > m10_tau_star).sum())}', flush=True)

    # ---- dataset (for window queries) ----
    test_ds, feature_map, fc_edge_index = _build_test_dataset(dataset_name, slide_win)
    assert len(test_ds) == T, (
        f'TimeDataset size {len(test_ds)} != cached array T={T}'
    )

    # ---- model + aleatoric head ----
    # Re-attach the causal graph (mask or restrict) from the hp json so the
    # reconstructed model uses the SAME learned graph the bundle was built on.
    # Without this the CF would run on the unrestricted top-K graph and the
    # edge_index_sample assertion below would fail.
    causal_mask_tensor = None
    if hp.get('causal_mask', ''):
        causal_mask_tensor = load_causal_mask(hp['causal_mask'], feature_map)
    causal_restrict_tensor = None
    if hp.get('causal_restrict', ''):
        causal_restrict_tensor = load_causal_mask(hp['causal_restrict'], feature_map)
        print(f"[cf_engine] causal_restrict re-attached: {hp['causal_restrict']} "
              f"mode={hp.get('causal_restrict_mode','pure')}", flush=True)

    model = GDN_GDeltaUQ(
        [fc_edge_index], V,
        dim=dim,
        input_dim=slide_win,
        out_layer_num=int(hp['out_layer_num']),
        out_layer_inter_dim=int(hp['out_layer_inter_dim']),
        topk=topk,
        n_gnn_layers=int(hp['n_gnn_layers']),
        causal_mask=causal_mask_tensor,
        causal_mask_keep_self=bool(hp.get('causal_mask_keep_self', 1)),
        use_learnable_adj=bool(hp.get('use_learnable_adj', 0)),
        lsa_tau=float(hp.get('lsa_tau', 1.0)),
        causal_restrict=causal_restrict_tensor,
        causal_restrict_mode=hp.get('causal_restrict_mode', 'pure'),
        causal_restrict_keep_self=bool(hp.get('causal_restrict_keep_self', 1)),
    ).to(dev)
    state = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    aleatoric_head = AleatoricHead(
        hidden_dim=dim, num_sensors=V, sensor_embed_dim=16, mlp_hidden=64)
    aleatoric_head.load_state_dict(
        torch.load(Path(bundle_dir) / 'aleatoric_head.pt', map_location='cpu'))
    aleatoric_head.to(dev).eval()
    for p in aleatoric_head.parameters():
        p.requires_grad = False

    anchor_pool = torch.load(Path(bundle_dir) / 'anchor_pool.pt', map_location='cpu')
    anchor_pool = anchor_pool.to(dev)
    K = int(anchor_pool.shape[0])
    print(f'[cf_engine] anchor pool K={K} shape={tuple(anchor_pool.shape)}', flush=True)

    # ---- cache unmasked batch_gated + non_self mapping ----
    with torch.no_grad():
        # _build_learned_graph returns (batch_gated, broadcast_embeddings,
        # edge_weight); edge_weight is only used by the LSA path.
        unmasked_bg, broadcast_emb, _ = model._build_learned_graph(1, dev)
    # batch_gated is (2, E_total). For the plain top-K path E_total = topk*V;
    # for causal_mask/causal_restrict the in-degree is variable, so we only
    # require a 2-row edge index and let the edge_index_sample check below be
    # the real consistency gate.
    assert unmasked_bg.shape[0] == 2, (
        f'unexpected batch_gated shape {tuple(unmasked_bg.shape)}'
    )
    src = unmasked_bg[0].detach().cpu().numpy()
    tgt = unmasked_bg[1].detach().cpu().numpy()
    is_self = (src == tgt)
    non_self_indices = np.where(~is_self)[0]
    assert non_self_indices.shape[0] == E, (
        f'non-self edges {non_self_indices.shape[0]} != E={E} (from arrays)'
    )

    # Load and verify against edge_index_sample.npz
    edge_index_sample = np.load(Path(bundle_dir) / 'edge_index_sample.npz')['edge_index_sample']
    src_cached = edge_index_sample[0]
    tgt_cached = edge_index_sample[1]
    src_ours = src[non_self_indices]
    tgt_ours = tgt[non_self_indices]
    assert np.array_equal(src_cached, src_ours) and np.array_equal(tgt_cached, tgt_ours), (
        'cached edge_index_sample does not match our non-self ordering of batch_gated'
    )

    print(f'[cf_engine] cached unmasked batch_gated; non-self block matches edge_index_sample',
          flush=True)

    ctx = CFContext(
        model=model, aleatoric_head=aleatoric_head, anchor_pool=anchor_pool,
        device=dev, K=K, V=V, E=E, slide_win=slide_win, topk=topk,
        test_ds=test_ds,
        test_mu_bar=test_mu_bar, test_gt=test_gt, test_U_par=test_U_par,
        test_U_str=test_U_str, test_U_dist=test_U_dist,
        test_sigma2_ale=test_sigma2_ale, test_label=test_label,
        test_err_scores=test_err,
        n_err_mid=n_err_mid, n_err_iqr=n_err_iqr,
        med_par=med_par, iqr_par=iqr_par,
        med_str=med_str, iqr_str=iqr_str,
        med_sig=med_sig, iqr_sig=iqr_sig,
        med_dist=med_dist, iqr_dist=iqr_dist,
        med_agg=med_agg, iqr_agg=iqr_agg,
        log_sigma_tot_max_med=log_sigma_tot_max_med,
        log_sigma_tot_max_iqr=log_sigma_tot_max_iqr,
        renorm_params=renorm_params,
        m10=m10, m10_tau_star=m10_tau_star,
        s_M10_full=s_M10_full, feat_full=feat_full,
        unmasked_batch_gated=unmasked_bg, broadcast_embeddings=broadcast_emb,
        non_self_indices=non_self_indices,
        edge_index_sample=edge_index_sample,
        c_mask_nominal=c_mask_nominal, val_idx=val_idx, T=T,
    )
    return ctx


# --------------------------------------------------------------------------- #
# cf_engine: one masked CF query
# --------------------------------------------------------------------------- #

def _build_masked_edge_index(ctx: CFContext, edge_mask: Iterable[int],
                             node_mask: Iterable[int]) -> torch.Tensor:
    """Return a (2, num_remaining) edge_index with the masked edges/nodes
    dropped from the cached unmasked batch_gated."""
    src = ctx.unmasked_batch_gated[0]
    tgt = ctx.unmasked_batch_gated[1]
    total = ctx.unmasked_batch_gated.shape[1]

    drop = torch.zeros(total, dtype=torch.bool, device=src.device)
    for e in edge_mask:
        drop[ctx.non_self_indices[e]] = True
    if node_mask:
        node_mask_t = torch.tensor(list(node_mask), device=src.device, dtype=src.dtype)
        drop |= torch.isin(src, node_mask_t)
        drop |= torch.isin(tgt, node_mask_t)
    keep = ~drop
    return ctx.unmasked_batch_gated[:, keep]


def _aggregate_uncertainty(
    ctx: CFContext, mu_stack: torch.Tensor, h_stack: torch.Tensor,
    att_stack: torch.Tensor, num_nonself: int,
):
    """Compute mu_bar, U_par, U_str (over remaining non-self edges), U_dist,
    sigma2_ale at one timestep (B=1)."""
    mu_bar = mu_stack.mean(dim=0)               # (1, V)
    U_par = mu_stack.var(dim=0, unbiased=True)  # (1, V)
    U_str = att_stack.var(dim=0, unbiased=True) # (1, num_nonself)
    U_dist = U_par.mean(dim=-1)                  # (1,)
    h_bar = h_stack.mean(dim=0)                  # (1, V, d)
    log_sigma2 = ctx.aleatoric_head(h_bar)        # (1, V)
    sigma2_ale = log_sigma2.exp()
    return (mu_bar.cpu().numpy()[0],
            U_par.cpu().numpy()[0],
            U_str.cpu().numpy()[0],
            float(U_dist.item()),
            sigma2_ale.cpu().numpy()[0])


def _masked_mu_bar(ctx: CFContext, alarm_t: int,
                   edge_mask: Iterable[int] | None,
                   node_mask: Iterable[int] | None) -> np.ndarray:
    """Masked forward at one timestep; return the K-averaged mean
    prediction only, shape (V,). Used by faithful smoothing to re-score
    the prior windows under the same structural intervention without the
    att/aleatoric/feature work."""
    edge_mask = list(edge_mask) if edge_mask is not None else []
    node_mask = list(node_mask) if node_mask is not None else []
    model = ctx.model
    dev = ctx.device
    x_t, _, _, _ = ctx.test_ds[alarm_t]
    x_t = x_t.float().unsqueeze(0).to(dev)
    masked_bg = _build_masked_edge_index(ctx, edge_mask, node_mask)
    original = model._build_learned_graph
    bcast = ctx.broadcast_embeddings.to(dev)

    def patched(batch_num, device):
        # 3-tuple to match the model's _build_learned_graph signature
        # (edge_weight=None: CF masking uses hard edges, no LSA gating).
        return masked_bg, bcast, None
    model._build_learned_graph = patched
    try:
        with torch.no_grad():
            h_pre = model.forward_split(x_t)
            mu_stack = torch.empty((ctx.K, 1, ctx.V), device=dev)
            for k in range(ctx.K):
                mu_k, _, _ = model.forward_anchored(h_pre, ctx.anchor_pool[k])
                mu_stack[k] = mu_k
    finally:
        model._build_learned_graph = original
    return mu_stack.mean(dim=0).cpu().numpy()[0]


def cf_engine(
    ctx: CFContext, alarm_t: int,
    edge_mask: Iterable[int] | None = None,
    node_mask: Iterable[int] | None = None,
    return_components: bool = False,
    faithful_smoothing: bool = False,
):
    """Run one masked CF query at timestep `alarm_t` (index into test arrays).

    Args:
        ctx: built by build_cf_context.
        alarm_t: index into cached arrays in [0, T).
        edge_mask: iterable of indices in [0, E) referring to
            edge_index_sample columns (non-self edges).
        node_mask: iterable of sensor indices in [0, V) whose incident
            edges are all dropped.
        return_components: if True, also return the underlying tensors.

    Returns:
        s_M10 (log-odds) under the masked graph. If return_components, a
        dict with keys mu_bar, U_par, U_str, U_dist, sigma2_ale, agg_z, feat.
    """
    edge_mask = list(edge_mask) if edge_mask is not None else []
    node_mask = list(node_mask) if node_mask is not None else []
    model = ctx.model
    dev = ctx.device

    # Window tensor at alarm_t — TimeDataset returns (x, y, label, edge_index)
    x_t, y_t, label_t, _ = ctx.test_ds[alarm_t]
    x_t = x_t.float().unsqueeze(0).to(dev)  # (1, V, W)

    masked_bg = _build_masked_edge_index(ctx, edge_mask, node_mask)
    num_nonself_after = (masked_bg[0] != masked_bg[1]).sum().item()

    original = model._build_learned_graph
    bcast = ctx.broadcast_embeddings.to(dev)

    def patched(batch_num, device):
        # 3-tuple to match the model's _build_learned_graph signature
        # (edge_weight=None: CF masking uses hard edges, no LSA gating).
        return masked_bg, bcast, None
    model._build_learned_graph = patched

    try:
        with torch.no_grad():
            h_pre = model.forward_split(x_t)             # (1, V, d)
            mu_stack = torch.empty((ctx.K, 1, ctx.V), device=dev)
            h_stack = torch.empty((ctx.K, 1, ctx.V, h_pre.shape[-1]), device=dev)
            att_stack = torch.empty((ctx.K, 1, num_nonself_after), device=dev)
            for k in range(ctx.K):
                anchor = ctx.anchor_pool[k]
                mu_k, h_k, att_k = model.forward_anchored(h_pre, anchor)
                mu_stack[k] = mu_k
                h_stack[k] = h_k
                # att_k shape: (num_total_edges_with_selfloops, heads=1, 1).
                # Non-self block is the first num_nonself_after entries.
                att_flat = att_k.view(-1)
                att_stack[k] = att_flat[:num_nonself_after].view(1, num_nonself_after)
    finally:
        model._build_learned_graph = original

    mu_bar, U_par, U_str_e, U_dist, sigma2_ale = _aggregate_uncertainty(
        ctx, mu_stack, h_stack, att_stack, num_nonself_after)

    # --- recompute residual + agg_z at this timestep under mask ---
    err_t = _per_feature_err_score(mu_bar, ctx.test_gt[alarm_t],
                                    ctx.n_err_mid, ctx.n_err_iqr)
    if alarm_t >= BEFORE_NUM_SMOOTH:
        if faithful_smoothing:
            # Re-run the masked forward at the 5 prior windows so the
            # smoothing window reflects the same structural intervention
            # (≈6x cost). The default keeps the cached unmasked residuals.
            prior = np.vstack([
                _per_feature_err_score(
                    _masked_mu_bar(ctx, tprev, edge_mask, node_mask),
                    ctx.test_gt[tprev], ctx.n_err_mid, ctx.n_err_iqr)
                for tprev in range(alarm_t - BEFORE_NUM_SMOOTH, alarm_t)
            ])
        else:
            prior = ctx.test_err_scores[alarm_t - BEFORE_NUM_SMOOTH:alarm_t]
        smoothed_t = np.vstack([prior, err_t[None, :]]).mean(axis=0)
    else:
        smoothed_t = err_t  # boundary: no smoothing window available
    agg_new = float(smoothed_t.max())
    agg_z_new = (agg_new - ctx.med_agg) / (ctx.iqr_agg + EPS_FEAT)

    # --- recompute per-sensor z-scores at this timestep ---
    # For per-sensor channels (U_par, sigma2_ale) we still have V entries; we
    # exclude masked nodes from max/mean aggregates.
    node_mask_set = set(node_mask)
    sensor_keep = np.ones(ctx.V, dtype=bool)
    for v in node_mask_set:
        sensor_keep[v] = False

    z_par_t = (U_par - ctx.med_par) / (ctx.iqr_par + EPS_FEAT)
    z_sig_t = (sigma2_ale - ctx.med_sig) / (ctx.iqr_sig + EPS_FEAT)
    z_par_max = float(z_par_t[sensor_keep].max()) if sensor_keep.any() else 0.0
    z_par_mean = float(z_par_t[sensor_keep].mean()) if sensor_keep.any() else 0.0
    z_sig_max = float(z_sig_t[sensor_keep].max()) if sensor_keep.any() else 0.0

    # For edges: U_str_e has shape (num_nonself_after,). Map back to
    # edge_index_sample positions via the cumulative mapping.
    # Build the mapping by reusing _build_masked_edge_index logic: pick the
    # subset of non-self positions that survived the drop, in order.
    src_full = ctx.unmasked_batch_gated[0].detach().cpu().numpy()
    tgt_full = ctx.unmasked_batch_gated[1].detach().cpu().numpy()
    total = src_full.shape[0]
    drop = np.zeros(total, dtype=bool)
    for e in edge_mask:
        drop[ctx.non_self_indices[e]] = True
    if node_mask:
        nm = np.asarray(list(node_mask), dtype=src_full.dtype)
        drop |= np.isin(src_full, nm) | np.isin(tgt_full, nm)
    keep_full = ~drop  # (total,)
    # Among the original 765 columns, after dropping, the remaining columns
    # are in the same order. The non-self block is the first num_nonself_after
    # of those.
    surviving = np.where(keep_full)[0]
    surviving_non_self = [c for c in surviving if src_full[c] != tgt_full[c]]
    assert len(surviving_non_self) == num_nonself_after, (
        f'non-self count mismatch: {len(surviving_non_self)} vs {num_nonself_after}'
    )
    # surviving_non_self[i] is the column index in batch_gated of the i-th
    # surviving non-self edge. The corresponding position in edge_index_sample
    # is the position of that column in ctx.non_self_indices.
    pos_in_eis = np.searchsorted(ctx.non_self_indices, np.asarray(surviving_non_self))
    z_str_e_t = (U_str_e - ctx.med_str[pos_in_eis]) / (ctx.iqr_str[pos_in_eis] + EPS_FEAT)
    z_str_mean = float(z_str_e_t.mean()) if z_str_e_t.size else 0.0

    z_dist = float((U_dist - ctx.med_dist) / (ctx.iqr_dist + EPS_FEAT))

    # 1d renorm of the 5 signals
    rn = ctx.renorm_params
    sig_U_par_max = (z_par_max - rn['U_par_max_v'][0]) / (rn['U_par_max_v'][1] + EPS_FEAT)
    sig_U_par_mean = (z_par_mean - rn['U_par_mean_v'][0]) / (rn['U_par_mean_v'][1] + EPS_FEAT)
    sig_str_mean = (z_str_mean - rn['U_str_mean_e'][0]) / (rn['U_str_mean_e'][1] + EPS_FEAT)
    sig_U_dist = (z_dist - rn['U_dist'][0]) / (rn['U_dist'][1] + EPS_FEAT)
    sig_sigma_max = (z_sig_max - rn['sigma_ale_max_v'][0]) / (rn['sigma_ale_max_v'][1] + EPS_FEAT)

    # 8th feature
    sigma_tot = np.sqrt(np.maximum(sigma2_ale, SIGMA_FLOOR)
                        + np.maximum(U_par, 0.0))[sensor_keep]
    log_sigma_tot_max = np.log(float(sigma_tot.max()) + EPS_FEAT) if sigma_tot.size else 0.0
    z_log_sigma_tot_max = ((log_sigma_tot_max - ctx.log_sigma_tot_max_med)
                            / (ctx.log_sigma_tot_max_iqr + EPS_FEAT))

    feat = np.array([[
        agg_z_new,
        sig_U_par_max,
        sig_U_par_mean,
        sig_sigma_max,
        sig_str_mean,
        sig_U_dist,
        sig_U_par_max * agg_z_new,
        z_log_sigma_tot_max,
    ]])
    proba = float(ctx.m10.predict_proba(feat)[0, 1])
    proba = max(min(proba, 1 - 1e-8), 1e-8)
    s = float(np.log(proba / (1 - proba)))

    if not return_components:
        return s
    return {
        's_M10': s,
        'mu_bar': mu_bar,
        'U_par': U_par,
        'U_str_e_remaining': U_str_e,
        'U_dist': U_dist,
        'sigma2_ale': sigma2_ale,
        'agg_z': agg_z_new,
        'feat': feat,
        'num_nonself_after': num_nonself_after,
        'pos_in_eis': pos_in_eis,
    }


# --------------------------------------------------------------------------- #
# Smoke test: empty-mask identity
# --------------------------------------------------------------------------- #

def smoke_test_empty_mask(ctx: CFContext, sample_ts: list[int],
                           tol: float = 1e-3,
                           faithful_smoothing: bool = False) -> bool:
    """Run cf_engine with empty mask at each ts and compare to cached
    M10 score s_M10_full. Tolerance: tol absolute on log-odds.
    With faithful_smoothing=True this also exercises the 6-window
    re-scoring path; under an empty mask it must still reproduce the
    cached residuals exactly."""
    label = 'faithful' if faithful_smoothing else 'approx'
    print(f'\n[smoke] empty-mask identity test ({label}, tol=%.3g)' % tol,
          flush=True)
    all_pass = True
    for ts in sample_ts:
        s_cached = float(ctx.s_M10_full[ts])
        s_cf = cf_engine(ctx, ts, edge_mask=None, node_mask=None,
                         faithful_smoothing=faithful_smoothing)
        delta = abs(s_cached - s_cf)
        ok = delta < tol
        all_pass &= ok
        print(f'  ts={ts:>6d}  cached={s_cached:+.6f}  cf={s_cf:+.6f}  '
              f'delta={delta:.2e}  {"PASS" if ok else "FAIL"}', flush=True)
    print(f'[smoke] {"ALL PASS" if all_pass else "FAIL — fix before continuing"}',
          flush=True)
    return all_pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--arrays',
                    default='results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655/arrays.npz')
    ap.add_argument('--checkpoint',
                    default='pretrained/swat_gdeltauq_sw60/best_0513-211014.pt')
    ap.add_argument('--hp',
                    default='pretrained/swat_gdeltauq_sw60/hyperparameters_0513-211014.json')
    ap.add_argument('--bundle',
                    default='pretrained/swat_gdeltauq_sw60/calibration_bundle_K100')
    ap.add_argument('--cal-split',
                    default='pretrained/swat_ensemble/calibration_bundle/calibration_set_indices.json')
    ap.add_argument('--dataset', default='swat')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tau-star', type=float, default=2.0914)
    ap.add_argument('--smoke-ts', type=int, nargs='+',
                    default=[100, 5000, 15000, 25000, 35000])
    args = ap.parse_args()

    ctx = build_cf_context(
        arrays_path=args.arrays, checkpoint_path=args.checkpoint,
        hp_path=args.hp, bundle_dir=args.bundle,
        cal_split_path=args.cal_split, dataset_name=args.dataset,
        device=args.device, m10_tau_star=args.tau_star,
    )
    ok_approx = smoke_test_empty_mask(ctx, args.smoke_ts,
                                       faithful_smoothing=False)
    ok_faithful = smoke_test_empty_mask(ctx, args.smoke_ts,
                                         faithful_smoothing=True)
    print(f'\n[smoke] approx={"PASS" if ok_approx else "FAIL"}  '
          f'faithful={"PASS" if ok_faithful else "FAIL"}', flush=True)


if __name__ == '__main__':
    main()
