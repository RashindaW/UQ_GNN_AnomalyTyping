"""Per-task post-processing for scripts/run_lsa_verification4.py.

Reads the eval run directory's arrays.npz + report.json, computes PA%K
at several K values, and writes:
  <run_dir>/pa_k.json
  <run_dir>/combined.json

Usage:
  python scripts/_v4_per_task_postproc.py <run_dir> <seed> <lam> \
      <bundle_dir> <hp_path> <ckpt>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pa_k_metric import best_f1_pa_k


def main():
    if len(sys.argv) != 7:
        sys.exit(f'usage: {sys.argv[0]} run_dir seed lam bundle_dir hp_path ckpt')
    run_dir = Path(sys.argv[1])
    seed = int(sys.argv[2])
    lam = float(sys.argv[3])
    bundle_dir = sys.argv[4]
    hp_path = sys.argv[5]
    ckpt = sys.argv[6]

    arrays = np.load(run_dir / 'arrays.npz')
    labels = arrays['test_attack_label'].astype(np.int32)
    full_scores = arrays['full_scores']         # (V, T) per-feature err scores
    # Top-1 aggregation across features (paper protocol uses topk=1).
    scores = full_scores.max(axis=0)

    pa_k = {}
    for K_pct in (0, 5, 10, 20, 50, 100):
        m = best_f1_pa_k(scores, labels, K_pct=K_pct, n_thresholds=400)
        pa_k[f'PA%K_{K_pct}'] = {
            'F1': float(m['F1']),
            'P':  float(m['P']),
            'R':  float(m['R']),
            'tau': float(m['tau']),
        }
    (run_dir / 'pa_k.json').write_text(json.dumps(pa_k, indent=2))
    print(f'[pa_k] wrote {run_dir / "pa_k.json"}', flush=True)

    report = json.loads((run_dir / 'report.json').read_text())
    combined = {
        'seed': seed,
        'lambda': lam,
        'checkpoint': ckpt,
        'hyperparameters': hp_path,
        'bundle_dir': bundle_dir,
        'paper_protocol': report['paper_protocol'],
        'pa_k': pa_k,
        'run_dir': str(run_dir),
    }
    out = run_dir / 'combined.json'
    out.write_text(json.dumps(combined, indent=2))
    print(f'[task] wrote combined report {out}', flush=True)


if __name__ == '__main__':
    main()
