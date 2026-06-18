"""Build pretrained/dualstgf_ensemble/manifest.json after the ensemble finishes.

Mirrors scripts/build_manifest.py but writes to dualstgf_ensemble/. Records
`model: 'dualstgf_uq'` so calibrate.py / detect.py can dispatch on the manifest.
"""
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENSEMBLE_ROOT = REPO_ROOT / 'pretrained' / 'dualstgf_ensemble'

CHECKPOINT_RE = re.compile(r'^CHECKPOINT_PATH=(.+)$')
MEMBER_DIR_RE = re.compile(r'^member_(\d+)_seed_(\d+)$')

HYPERPARAMETERS = {
    'model': 'dualstgf_uq',
    'dataset': 'swat',
    'window_size': 60,
    'train_stride': 1,
    'val_stride': 5,
    'batch': 32,
    'epoch': 50,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'early_stop_patience': 15,
    'gnn_embed_dim': 64,
    'temp_node_embed_dim': 16,
    'recon_hidden_dim': 10,
    'topk': 15,
    'num_gnn_layers': 1,
    'with_variance_head': True,
    'logvar_clamp': [-10.0, 10.0],
    'aug_control': False,
    'use_spectral_view': False,
    'lambda_div': 0.0,
    'anomaly_weight': 0.0,
}


def parse_member_dir(d: Path):
    m = MEMBER_DIR_RE.match(d.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def extract_checkpoint(log_path: Path):
    if not log_path.is_file():
        return None
    with log_path.open() as f:
        for line in f:
            m = CHECKPOINT_RE.match(line.strip())
            if m:
                return m.group(1).strip()
    return None


def main():
    if not ENSEMBLE_ROOT.is_dir():
        print(f'[manifest-dualstgf] ENSEMBLE_ROOT does not exist: {ENSEMBLE_ROOT}', file=sys.stderr)
        sys.exit(1)

    members = []
    for entry in sorted(ENSEMBLE_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_member_dir(entry)
        if parsed is None:
            continue
        idx, seed = parsed

        log_path = entry / 'train.log'
        ckpt = extract_checkpoint(log_path)
        if ckpt is None:
            print(f'[manifest-dualstgf] WARN: no CHECKPOINT_PATH in {log_path}', file=sys.stderr)
            continue

        ckpt_abs = (REPO_ROOT / ckpt).resolve() if not os.path.isabs(ckpt) else Path(ckpt)
        ckpt_exists = ckpt_abs.is_file()
        if not ckpt_exists:
            print(f'[manifest-dualstgf] WARN: checkpoint missing for member {idx}: {ckpt}',
                  file=sys.stderr)

        members.append({
            'index': idx,
            'seed': seed,
            'directory': str(entry.relative_to(REPO_ROOT)),
            'checkpoint': ckpt,
            'checkpoint_exists': ckpt_exists,
            'log': str(log_path.relative_to(REPO_ROOT)),
        })

    members.sort(key=lambda d: d['index'])

    manifest = {
        'model': 'dualstgf_uq',
        'dataset': 'swat',
        'M': len(members),
        'members': members,
        'hyperparameters': HYPERPARAMETERS,
    }
    out_path = ENSEMBLE_ROOT / 'manifest.json'
    with out_path.open('w') as f:
        json.dump(manifest, f, indent=2)
    print(f'[manifest-dualstgf] wrote {out_path} ({len(members)} members)')

    if any(not m['checkpoint_exists'] for m in members):
        sys.exit(2)


if __name__ == '__main__':
    main()
