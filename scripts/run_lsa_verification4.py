"""Verification 4: M=5 seeds x {lambda=0, lambda*=1e-2} on the LSA +
G-DeltaUQ joint-NLL pipeline.

Per (seed, lambda) task, run end-to-end in a single subprocess:
  1. train_gdeltauq_jointnll_main.py -use_learnable_adj 1 -lambda_adj LAM
     -random_seed SEED -save_path_pattern <pattern>
  2. scripts/calibrate_gdeltauq.py -checkpoint ... -hyperparameters ...
  3. scripts/eval_paper_protocol_gdeltauq.py -checkpoint ... -bundle_dir ...
     (writes report.json with paper F1, P, R, AUC and arrays.npz)
  4. (post) compute PA%K (Kim et al. 2022) on the saved scores via
     scripts/pa_k_metric.py utilities.

Tasks dispatched across GPUs identically to scripts/run_lsa_lambda_sweep_parallel.py
(poll for util<=threshold AND mem free>=threshold). Per-task log captures all 3
stages; per-task report.json keys are aggregated at the end.

Final per-condition rollup (mean +/- std across the 5 seeds) is written to:
  pretrained/<pattern>/verification4_summary_<datestr>.json
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


# --------------------------------------------------------------------------- #
# GPU dispatch helpers (copied from run_lsa_lambda_sweep_parallel.py)
# --------------------------------------------------------------------------- #

def _query_gpu_state() -> list[dict]:
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.free',
             '--format=csv,noheader,nounits'], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        sys.exit(f'[v4] nvidia-smi failed: {e}')
    gpus = []
    for line in out.strip().splitlines():
        idx, util, mem_free = [s.strip() for s in line.split(',')]
        gpus.append({'index': int(idx),
                     'util_pct': int(util),
                     'mem_free_mib': int(mem_free)})
    return gpus


def _free_gpus(util_threshold: int, mem_threshold_mib: int,
               busy_gpus: set[int]) -> list[int]:
    return [g['index'] for g in _query_gpu_state()
            if g['index'] not in busy_gpus
            and g['util_pct'] <= util_threshold
            and g['mem_free_mib'] >= mem_threshold_mib]


# --------------------------------------------------------------------------- #
# Per-task script (chains train + calibrate + eval into one bash command)
# --------------------------------------------------------------------------- #

PER_TASK_SH = r'''
set -euo pipefail

PY="{python}"
SEED="{seed}"
LAM="{lam}"
DEVICE="cuda:{gpu}"
PATTERN="{pattern}"
SPLIT_PATH="{split_path}"
RESULTS_DIR="{results_dir}"
SAVE_DIR="pretrained/${{PATTERN}}"

echo "[task] seed=${{SEED}} lambda=${{LAM}} gpu=${{DEVICE}} pattern=${{PATTERN}}"
mkdir -p "$SAVE_DIR" "$RESULTS_DIR"

# --- 1. train ---
echo "[task] === train ==="
"$PY" -u train_gdeltauq_jointnll_main.py \
  -dataset {dataset} \
  -epoch {epoch} -batch {batch} \
  -slide_win {slide_win} -slide_stride {slide_stride} \
  -dim {dim} -out_layer_num {out_layer_num} -out_layer_inter_dim {out_layer_inter_dim} \
  -topk {topk} -n_gnn_layers {n_gnn_layers} -K_anchors {K_anchors} \
  -decay {decay} -random_seed "$SEED" \
  -split_path "$SPLIT_PATH" \
  -save_path_pattern "$PATTERN" \
  -device "$DEVICE" \
  -use_learnable_adj 1 -lambda_adj "$LAM" -lsa_tau {lsa_tau} \
  -logvar_l2 {logvar_l2} \
  -comment "v4_seed${{SEED}}_lambda${{LAM}}" \
  | tee "${{SAVE_DIR}}/train_seed${{SEED}}_lambda${{LAM}}.log"

# Recover checkpoint paths from the training log (train_gdeltauq_jointnll_main
# prints CHECKPOINT_PATH=... and HEAD_PATH=...).
CKPT=$(grep '^CHECKPOINT_PATH=' "${{SAVE_DIR}}/train_seed${{SEED}}_lambda${{LAM}}.log" | tail -1 | cut -d= -f2)
HEAD=$(grep '^HEAD_PATH='       "${{SAVE_DIR}}/train_seed${{SEED}}_lambda${{LAM}}.log" | tail -1 | cut -d= -f2)
STEM=$(basename "$CKPT" .pt | sed 's/^best_//')
HP="${{SAVE_DIR}}/hyperparameters_${{STEM}}.json"
echo "[task] CKPT=$CKPT  HP=$HP  HEAD=$HEAD"

# --- 2. calibrate ---
echo "[task] === calibrate ==="
BUNDLE_DIR="${{SAVE_DIR}}/calibration_bundle_seed${{SEED}}_lambda${{LAM}}"
"$PY" -u scripts/calibrate_gdeltauq.py \
  -checkpoint "$CKPT" \
  -hyperparameters "$HP" \
  -split_path "$SPLIT_PATH" \
  -device "$DEVICE" \
  -save_dir "$BUNDLE_DIR" \
  2>&1 | tee "${{SAVE_DIR}}/calibrate_seed${{SEED}}_lambda${{LAM}}.log"

# --- 3. eval (paper protocol F1/P/R/AUC + arrays for PA%K) ---
echo "[task] === eval ==="
EVAL_RESULTS_DIR="${{RESULTS_DIR}}/eval_seed${{SEED}}_lambda${{LAM}}"
"$PY" -u scripts/eval_paper_protocol_gdeltauq.py \
  -checkpoint "$CKPT" \
  -hyperparameters "$HP" \
  -bundle_dir "$BUNDLE_DIR" \
  -split_path "$SPLIT_PATH" \
  -device "$DEVICE" \
  -results_dir "$EVAL_RESULTS_DIR" \
  2>&1 | tee "${{SAVE_DIR}}/eval_seed${{SEED}}_lambda${{LAM}}.log"

# Find the eval run dir (newest under EVAL_RESULTS_DIR).
EVAL_RUN=$(ls -td "$EVAL_RESULTS_DIR"/*/ | head -1)
echo "[task] EVAL_RUN=$EVAL_RUN"

# --- 4. PA%K + combined.json post-step (extracted to its own script to
#     avoid Python-format-vs-bash-braces conflicts in the orchestrator). ---
echo "[task] === PA%K ==="
"$PY" -u scripts/_v4_per_task_postproc.py \
  "$EVAL_RUN" "$SEED" "$LAM" "$BUNDLE_DIR" "$HP" "$CKPT" \
  2>&1 | tee "${{SAVE_DIR}}/postproc_seed${{SEED}}_lambda${{LAM}}.log"

echo "[task] DONE seed=${{SEED}} lambda=${{LAM}}"
'''.lstrip()


@dataclass
class Job:
    seed: int
    lam: float
    gpu: int
    proc: subprocess.Popen
    log_path: Path
    start_ts: float = field(default_factory=time.time)


def _spawn(seed: int, lam: float, gpu: int, args, datestr: str,
           pattern: str) -> Job:
    log_dir = REPO_ROOT / f'pretrained/{pattern}'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'task_seed{seed}_lambda{lam:.0e}_{datestr}.log'
    results_dir = REPO_ROOT / f'results/{pattern}'
    sh = PER_TASK_SH.format(
        python=args.python,
        seed=seed, lam=lam, gpu=gpu,
        pattern=pattern,
        split_path=args.split_path,
        results_dir=str(results_dir),
        dataset=args.dataset,
        epoch=args.epoch, batch=args.batch,
        slide_win=args.slide_win, slide_stride=args.slide_stride,
        dim=args.dim, out_layer_num=args.out_layer_num,
        out_layer_inter_dim=args.out_layer_inter_dim,
        topk=args.topk, n_gnn_layers=args.n_gnn_layers,
        K_anchors=args.K_anchors, decay=args.decay,
        lsa_tau=args.lsa_tau, logvar_l2=args.logvar_l2,
    )
    log_f = open(log_path, 'w')
    proc = subprocess.Popen(['bash', '-c', sh], stdout=log_f,
                             stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    log_f.close()
    print(f'[v4] LAUNCH  seed={seed}  lambda={lam:.0e}  gpu={gpu}  '
          f'log={log_path}', flush=True)
    return Job(seed=seed, lam=lam, gpu=gpu, proc=proc, log_path=log_path)


def _aggregate(pattern: str, datestr: str, seeds: list[int],
                lambdas: list[float]) -> dict:
    """Collect every combined.json under results/<pattern>/eval_*/ and
    bucket by lambda."""
    eval_root = REPO_ROOT / f'results/{pattern}'
    rows = []
    for combined in eval_root.glob('eval_seed*_lambda*/*/combined.json'):
        try:
            rows.append(json.loads(combined.read_text()))
        except Exception as e:
            print(f'[v4] skip {combined}: {e}', flush=True)
    print(f'[v4] aggregated {len(rows)} per-task combined.json files',
          flush=True)

    import statistics as stats
    by_lam: dict[float, list[dict]] = {}
    for r in rows:
        by_lam.setdefault(float(r['lambda']), []).append(r)
    per_condition = []
    for lam in sorted(by_lam):
        members = by_lam[lam]
        agg = {'lambda': lam, 'n_seeds': len(members),
               'seeds': sorted(int(m['seed']) for m in members)}

        # Paper-protocol metrics.
        for k in ('F1', 'precision', 'recall', 'AUC'):
            vals = [m['paper_protocol'][k] for m in members]
            agg[f'paper_{k}_mean'] = float(stats.mean(vals))
            agg[f'paper_{k}_std']  = float(stats.pstdev(vals)) if len(vals) > 1 else 0.0
        # PA%K metrics.
        for K_pct in (0, 5, 10, 20, 50, 100):
            for k in ('F1', 'P', 'R'):
                vals = [m['pa_k'][f'PA%K_{K_pct}'][k] for m in members]
                agg[f'PA%K_{K_pct}_{k}_mean'] = float(stats.mean(vals))
                agg[f'PA%K_{K_pct}_{k}_std']  = float(stats.pstdev(vals)) if len(vals) > 1 else 0.0
        per_condition.append(agg)

    summary = {'datestr': datestr, 'pattern': pattern,
                'seeds_requested': seeds, 'lambdas_requested': lambdas,
                'per_seed': rows, 'per_condition': per_condition}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-lambdas',  type=str, default='0.0,1e-2',
                    help='Comma-separated lambdas (default: re-baseline + lambda*).')
    ap.add_argument('-seeds',    type=str, default='1,2,3,42,100',
                    help='Comma-separated seeds.')
    ap.add_argument('-pattern',  type=str, default='swat_gdeltauq_jointnll_lsa_v4',
                    help='Subdir under pretrained/ and results/ to write artifacts to.')
    # Production hyperparameters.
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
    ap.add_argument('-lsa_tau', type=float, default=1.0)
    ap.add_argument('-logvar_l2', type=float, default=0.0)
    ap.add_argument('-split_path', type=str,
                    default='data/swat/gdeltauq_split.json')
    # Dispatch.
    ap.add_argument('--max-parallel', type=int, default=4)
    ap.add_argument('--util-threshold', type=int, default=20)
    ap.add_argument('--mem-threshold-mib', type=int, default=10000)
    ap.add_argument('--poll-secs', type=int, default=30)
    ap.add_argument('--python', type=str,
                    default='/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python')
    args = ap.parse_args()

    lambdas = [float(x.strip()) for x in args.lambdas.split(',') if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    tasks = [(s, l) for l in lambdas for s in seeds]
    print(f'[v4] {len(tasks)} tasks: {len(seeds)} seeds x {len(lambdas)} lambdas',
          flush=True)
    print(f'[v4] seeds={seeds}  lambdas={lambdas}', flush=True)
    print(f'[v4] pattern={args.pattern}', flush=True)

    datestr = datetime.now().strftime('%m%d-%H%M%S')
    pending = list(tasks)
    running: list[Job] = []
    t0 = time.time()

    while pending or running:
        still = []
        for j in running:
            rc = j.proc.poll()
            if rc is None:
                still.append(j)
            else:
                elapsed = time.time() - j.start_ts
                print(f'[v4] FINISH  seed={j.seed}  lambda={j.lam:.0e}  '
                      f'gpu={j.gpu}  rc={rc}  elapsed={elapsed/60:.1f}m',
                      flush=True)
        running = still

        if pending and len(running) < args.max_parallel:
            busy = {j.gpu for j in running}
            free = _free_gpus(args.util_threshold, args.mem_threshold_mib, busy)
            for gpu in free:
                if not pending or len(running) >= args.max_parallel:
                    break
                seed, lam = pending.pop(0)
                running.append(_spawn(seed, lam, gpu, args, datestr, args.pattern))

        if pending or running:
            time.sleep(args.poll_secs)

    wall = time.time() - t0
    print(f'\n[v4] all tasks complete in {wall/60:.1f}m; aggregating',
          flush=True)

    summary = _aggregate(args.pattern, datestr, seeds, lambdas)
    summary['wall_time_s'] = wall

    out_dir = REPO_ROOT / 'pretrained' / args.pattern
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f'verification4_summary_{datestr}.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'[v4] wrote {summary_path}', flush=True)

    # Pretty table on stdout.
    print('\n=== Verification 4 final table (mean +/- std over seeds) ===')
    hdr = ['lambda', 'n', 'paper_F1', 'paper_P', 'paper_R', 'paper_AUC',
           'PA%K_0_F1', 'PA%K_20_F1', 'PA%K_100_F1']
    print('  '.join(f'{h:>14s}' for h in hdr))
    for cond in summary['per_condition']:
        row = [
            f'{cond["lambda"]:.0e}',
            f'{cond["n_seeds"]}',
            f'{cond["paper_F1_mean"]:.4f}+/-{cond["paper_F1_std"]:.4f}',
            f'{cond["paper_precision_mean"]:.4f}+/-{cond["paper_precision_std"]:.4f}',
            f'{cond["paper_recall_mean"]:.4f}+/-{cond["paper_recall_std"]:.4f}',
            f'{cond["paper_AUC_mean"]:.4f}+/-{cond["paper_AUC_std"]:.4f}',
            f'{cond["PA%K_0_F1_mean"]:.4f}+/-{cond["PA%K_0_F1_std"]:.4f}',
            f'{cond["PA%K_20_F1_mean"]:.4f}+/-{cond["PA%K_20_F1_std"]:.4f}',
            f'{cond["PA%K_100_F1_mean"]:.4f}+/-{cond["PA%K_100_F1_std"]:.4f}',
        ]
        print('  '.join(f'{c:>14s}' for c in row))


if __name__ == '__main__':
    main()
