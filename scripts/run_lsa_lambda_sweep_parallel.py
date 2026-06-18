"""Parallel multi-GPU lambda sweep for LSA + G-DeltaUQ joint-NLL.

Each lambda trains independently (cold-start) on its own GPU, so the sweep
runs in batches of `--max-parallel` lambdas. Polls nvidia-smi for free
GPUs (utilisation below `--util-threshold` and memory free above
`--mem-threshold-mib`) and dispatches jobs as soon as a GPU clears.

For each lambda we shell out to `train_gdeltauq_jointnll_main.py` with the
LSA flags set, so the per-lambda training is exactly the production
pipeline -- no warm-start, but reproducible end-to-end. After all jobs
finish we read each lambda's checkpoint, run one val pass, and pick
lambda* via the same rule as the sequential driver:
  - lambda* = smallest lambda within --tie-pct of best val_NLL
  - prefer sparser among ties (smaller lambda => sparser at this scale).

Writes:
  pretrained/<save_pattern>/lambda_sweep_parallel_<datestr>.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _query_gpu_state() -> list[dict]:
    """Per-GPU dict {index, util_pct, mem_free_mib}."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.free',
             '--format=csv,noheader,nounits'],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        sys.exit(f'[lsweep] nvidia-smi failed: {e}')
    gpus = []
    for line in out.strip().splitlines():
        idx, util, mem_free = [s.strip() for s in line.split(',')]
        gpus.append({'index': int(idx),
                     'util_pct': int(util),
                     'mem_free_mib': int(mem_free)})
    return gpus


def _free_gpus(util_threshold: int, mem_threshold_mib: int,
               busy_gpus: set[int]) -> list[int]:
    """Return indices currently below thresholds and not already in use
    by our own running jobs."""
    free = []
    for g in _query_gpu_state():
        if g['index'] in busy_gpus:
            continue
        if g['util_pct'] <= util_threshold and g['mem_free_mib'] >= mem_threshold_mib:
            free.append(g['index'])
    return free


@dataclass
class Job:
    lam: float
    gpu: int
    proc: subprocess.Popen
    log_path: Path
    save_path: Path
    head_path: Path
    hp_path: Path
    start_ts: float = field(default_factory=time.time)


def _per_lambda_paths(save_dir: Path, datestr: str, lam: float
                       ) -> tuple[Path, Path, Path, Path]:
    tag = f'lambda{lam:.0e}'
    return (save_dir / f'{tag}_{datestr}.pt',
            save_dir / f'aleatoric_head_{tag}_{datestr}.pt',
            save_dir / f'hyperparameters_{tag}_{datestr}.json',
            save_dir / f'log_{tag}_{datestr}.txt')


def _spawn_one(lam: float, gpu: int, args, datestr: str,
               save_dir: Path) -> Job:
    save_path, head_path, hp_path, log_path = _per_lambda_paths(save_dir, datestr, lam)
    cmd = [
        args.python, '-u', 'train_gdeltauq_jointnll_main.py',
        '-dataset', args.dataset,
        '-epoch', str(args.epoch),
        '-batch', str(args.batch),
        '-slide_win', str(args.slide_win),
        '-slide_stride', str(args.slide_stride),
        '-dim', str(args.dim),
        '-out_layer_num', str(args.out_layer_num),
        '-out_layer_inter_dim', str(args.out_layer_inter_dim),
        '-topk', str(args.topk),
        '-n_gnn_layers', str(args.n_gnn_layers),
        '-K_anchors', str(args.K_anchors),
        '-decay', str(args.decay),
        '-random_seed', str(args.random_seed),
        '-split_path', args.split_path,
        '-save_path_pattern', args.save_path_pattern,
        '-device', f'cuda:{gpu}',
        '-use_learnable_adj', '1',
        '-lambda_adj', str(lam),
        '-lsa_tau', str(args.lsa_tau),
        '-logvar_l2', str(args.logvar_l2),
        '-comment', f'lsa_lambdasweep_lambda{lam:.0e}',
    ]
    log_f = open(log_path, 'w')
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                             cwd=str(REPO_ROOT))
    log_f.close()
    # NOTE: We do NOT pre-write hp_path / save_path -- the spawned main
    # creates them with its own datestr. We rescan the directory after
    # each job completes to find the actual files.
    print(f'[lsweep] LAUNCH  lambda={lam:.4g}  gpu={gpu}  log={log_path}',
          flush=True)
    return Job(lam=lam, gpu=gpu, proc=proc, log_path=log_path,
               save_path=save_path, head_path=head_path, hp_path=hp_path)


def _scan_outputs(save_dir: Path, lam: float, since_ts: float
                   ) -> tuple[Path | None, Path | None, Path | None]:
    """The spawned main names its files with its own datestr. After a job
    completes we look for the most recent (model, head, hp) trio whose
    hyperparameters.json declares this lambda.
    """
    candidates = sorted(save_dir.glob('hyperparameters_*.json'),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for hp in candidates:
        if hp.stat().st_mtime < since_ts:
            continue
        try:
            d = json.loads(hp.read_text())
        except json.JSONDecodeError:
            continue
        if abs(d.get('lambda_adj', -1) - lam) < 1e-12:
            stem = hp.name.replace('hyperparameters_', '').replace('.json', '')
            mdl = save_dir / f'best_{stem}.pt'
            head = save_dir / f'aleatoric_head_{stem}.pt'
            if mdl.exists() and head.exists():
                return mdl, head, hp
    return None, None, None


def _val_one(model_ckpt: Path, head_ckpt: Path, hp: dict, args
              ) -> tuple[float, float, float]:
    """Reload checkpoint on CPU, run one val pass, compute mean degree."""
    import torch
    import argparse as _argparse
    sys.path.insert(0, str(REPO_ROOT))
    from train_gdeltauq_main import build_main_for_training
    from train_gdeltauq_jointnll_lambdasearch import _val_metrics
    from models.aleatoric_head import AleatoricHead

    ns = _argparse.Namespace(
        dataset=args.dataset, device='cpu',
        random_seed=hp['seed'], batch=args.batch,
        slide_win=hp['slide_win'], slide_stride=hp['slide_stride'],
        dim=hp['dim'], out_layer_num=hp['out_layer_num'],
        out_layer_inter_dim=hp['out_layer_inter_dim'],
        topk=hp['topk'], n_gnn_layers=hp['n_gnn_layers'],
        K_anchors=hp['K_anchors'], decay=hp['decay'],
        split_path=args.split_path,
        save_path_pattern=args.save_path_pattern,
        sensor_embed_dim=hp['sensor_embed_dim'],
        aleatoric_mlp_hidden=hp['aleatoric_mlp_hidden'],
        logvar_clamp_low=hp['logvar_clamp'][0],
        logvar_clamp_high=hp['logvar_clamp'][1],
        logvar_l2=hp['logvar_l2'],
        use_learnable_adj=1, lsa_tau=hp['lsa_tau'],
    )
    model, _train, val_loader, device, feature_map, _ = build_main_for_training(ns)
    head = AleatoricHead(
        hidden_dim=hp['dim'], num_sensors=len(feature_map),
        sensor_embed_dim=hp['sensor_embed_dim'],
        mlp_hidden=hp['aleatoric_mlp_hidden'],
        logvar_clamp=tuple(hp['logvar_clamp']),
    ).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    head.load_state_dict(torch.load(head_ckpt, map_location=device))
    val_nll, val_mse = _val_metrics(model, head, val_loader, device)
    with torch.no_grad():
        _ = model.adj(model.embedding.weight)
    deg = model.adj.mean_degree(args.edge_threshold)
    return val_nll, val_mse, deg


def main():
    ap = argparse.ArgumentParser()
    # Lambdas
    ap.add_argument('-lambda_grid', type=str,
                    default='1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4',
                    help='Comma-separated lambdas (cold-start each).')
    # Per-job (matches train_gdeltauq_jointnll_main.py)
    ap.add_argument('-dataset', type=str, default='swat')
    ap.add_argument('-batch', type=int, default=128)
    ap.add_argument('-epoch', type=int, default=100)
    ap.add_argument('-slide_win', type=int, default=60)
    ap.add_argument('-slide_stride', type=int, default=1)
    ap.add_argument('-dim', type=int, default=64)
    ap.add_argument('-out_layer_num', type=int, default=1)
    ap.add_argument('-out_layer_inter_dim', type=int, default=128)
    ap.add_argument('-topk', type=int, default=15)
    ap.add_argument('-n_gnn_layers', type=int, default=2)
    ap.add_argument('-K_anchors', type=int, default=10)
    ap.add_argument('-decay', type=float, default=0.0)
    ap.add_argument('-random_seed', type=int, default=42)
    ap.add_argument('-split_path', type=str,
                    default='data/swat/gdeltauq_split.json')
    ap.add_argument('-save_path_pattern', type=str,
                    default='swat_gdeltauq_jointnll_lsa')
    ap.add_argument('-lsa_tau', type=float, default=1.0)
    ap.add_argument('-logvar_l2', type=float, default=0.0)
    # Multi-GPU dispatch
    ap.add_argument('--max-parallel', type=int, default=4)
    ap.add_argument('--util-threshold', type=int, default=20,
                    help='Consider a GPU free when utilisation <= this percent.')
    ap.add_argument('--mem-threshold-mib', type=int, default=20000,
                    help='Consider a GPU free when free mem >= this many MiB.')
    ap.add_argument('--poll-secs', type=int, default=30)
    # Selection
    ap.add_argument('--edge-threshold', type=float, default=1e-3)
    ap.add_argument('--tie-pct', type=float, default=0.01)
    ap.add_argument('--python', type=str,
                    default='/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python')
    args = ap.parse_args()

    lambdas = [float(x.strip()) for x in args.lambda_grid.split(',') if x.strip()]
    print(f'[lsweep] grid: {lambdas}  max_parallel={args.max_parallel}',
          flush=True)

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    save_dir = REPO_ROOT / 'pretrained' / args.save_path_pattern
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'[lsweep] save_dir={save_dir}', flush=True)

    pending = list(lambdas)
    running: list[Job] = []
    completed: list[dict] = []
    sweep_start_ts = time.time()

    while pending or running:
        # Reap finished jobs
        still_running = []
        for j in running:
            rc = j.proc.poll()
            if rc is None:
                still_running.append(j)
            else:
                elapsed = time.time() - j.start_ts
                ok = (rc == 0)
                # Find the actual saved files for this lambda (the spawned
                # main uses its own datestr).
                mdl, head, hp = _scan_outputs(save_dir, j.lam, j.start_ts)
                print(f'[lsweep] FINISH lambda={j.lam:.4g}  gpu={j.gpu}  '
                      f'rc={rc}  elapsed={elapsed/60:.1f}m  '
                      f'ckpt={mdl.name if mdl else "MISSING"}', flush=True)
                rec = {'lambda_': j.lam, 'gpu': j.gpu, 'returncode': rc,
                       'elapsed_s': elapsed, 'log': str(j.log_path),
                       'checkpoint': str(mdl) if mdl else None,
                       'head_checkpoint': str(head) if head else None,
                       'hp': str(hp) if hp else None}
                completed.append(rec)
        running = still_running

        # Dispatch new jobs as GPUs free up
        if pending and len(running) < args.max_parallel:
            busy = {j.gpu for j in running}
            free = _free_gpus(args.util_threshold, args.mem_threshold_mib, busy)
            for gpu in free:
                if not pending or len(running) >= args.max_parallel:
                    break
                lam = pending.pop(0)
                running.append(_spawn_one(lam, gpu, args, datestr, save_dir))

        if pending or running:
            time.sleep(args.poll_secs)

    print(f'[lsweep] all {len(completed)} jobs complete in '
          f'{(time.time()-sweep_start_ts)/60:.1f}m; running val + selection',
          flush=True)

    # Post-process: val NLL + mean degree per lambda, then pick lambda*.
    rows = []
    for rec in sorted(completed, key=lambda r: -r['lambda_']):
        if not rec['checkpoint']:
            print(f'[lsweep] SKIP lambda={rec["lambda_"]} (no checkpoint)',
                  flush=True)
            continue
        hp_dict = json.loads(Path(rec['hp']).read_text())
        try:
            val_nll, val_mse, deg = _val_one(
                Path(rec['checkpoint']), Path(rec['head_checkpoint']),
                hp_dict, args)
        except Exception as e:
            print(f'[lsweep] val failed for lambda={rec["lambda_"]}: {e}',
                  flush=True)
            continue
        row = dict(lambda_=rec['lambda_'], val_nll=val_nll,
                   val_mse=val_mse, mean_degree=deg,
                   checkpoint=rec['checkpoint'],
                   head_checkpoint=rec['head_checkpoint'],
                   hp=rec['hp'], elapsed_s=rec['elapsed_s'])
        rows.append(row)
        print(f'[lsweep]   lambda={rec["lambda_"]:.4g}  '
              f'val_nll={val_nll:.6f}  val_mse={val_mse:.6f}  '
              f'mean_degree={deg:.2f}', flush=True)

    if not rows:
        sys.exit('[lsweep] no successful runs; nothing to select')

    # Filter out degenerate runs whose adjacency fully collapsed: those have
    # mean_degree ~= 0 and game val_NLL via overconfident sigma^2 (an artefact
    # of Gaussian NLL on a self-loop-only model). They should not be eligible
    # for lambda_star.
    candidates = [r for r in rows if r['mean_degree'] >= 1.0] or rows
    best = min(candidates, key=lambda r: r['val_nll'])
    # Signed-safe tie band: relax the threshold by abs(best)*tie_pct in the
    # 'worse' direction. The naive best * (1 + tie_pct) is mis-signed when
    # best < 0 (it tightens instead of loosens).
    threshold = best['val_nll'] + abs(best['val_nll']) * args.tie_pct
    qualifying = [r for r in candidates if r['val_nll'] <= threshold] or [best]
    lambda_star = min(qualifying, key=lambda r: r['lambda_'])

    summary = dict(
        datestr=datestr,
        lambda_grid=lambdas,
        max_parallel=args.max_parallel,
        best_val_nll=best['val_nll'],
        best_lambda=best['lambda_'],
        tie_threshold=threshold,
        lambda_star=lambda_star['lambda_'],
        lambda_star_checkpoint=lambda_star['checkpoint'],
        per_lambda=rows,
        wall_time_s=time.time() - sweep_start_ts,
    )
    out = save_dir / f'lambda_sweep_parallel_{datestr}.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f'\n[lsweep] wrote {out}', flush=True)
    print(f'[lsweep] LAMBDA_STAR={lambda_star["lambda_"]:.4g}  '
          f'val_nll={lambda_star["val_nll"]:.6f}  '
          f'mean_degree={lambda_star["mean_degree"]:.2f}  '
          f'ckpt={lambda_star["checkpoint"]}', flush=True)


if __name__ == '__main__':
    main()
