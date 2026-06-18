"""Build pretrained/swat_ensemble/manifest.json after train_ensemble.sh completes.

Walks each member_*_seed_*/ folder, extracts the CHECKPOINT_PATH=... line that
main.py prints early in training, sanity-checks that the file exists, and writes
a JSON manifest used by downstream UQ readouts.
"""
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENSEMBLE_ROOT = REPO_ROOT / 'pretrained' / 'swat_ensemble'

CHECKPOINT_RE = re.compile(r'^CHECKPOINT_PATH=(.+)$')
MEMBER_DIR_RE = re.compile(r'^member_(\d+)_seed_(\d+)$')

# Defaults — used as a fallback if no per-member hyperparameters.json file
# is found (older runs predate that artefact). Newer runs READ from the
# per-member hyperparameters.json produced by main.py at training time.
DEFAULT_HYPERPARAMETERS = {
    'model': 'gdn_uq',
    'dataset': 'swat',
    'batch': 32,
    'epoch': 30,
    'slide_win': 5,
    'slide_stride': 1,
    'dim': 64,
    'out_layer_num': 1,
    'out_layer_inter_dim': 128,
    'val_ratio': 0.2,
    'decay': 0,
    'topk': 15,
    'logvar_clamp': [-10.0, 10.0],
    'logvar_l2': 0.0,
    'optimizer': 'Adam',
    'lr': 1e-3,
    'betas': [0.9, 0.99],
    'early_stop_patience': 15,
}


def load_member_hyperparameters(member_dir: Path) -> dict:
    """Read per-member hyperparameters.json if present, else return defaults."""
    hp_path = member_dir / 'hyperparameters.json'
    if hp_path.is_file():
        with hp_path.open() as f:
            hp = json.load(f)
        # Merge over defaults so any missing keys fall back gracefully.
        merged = {**DEFAULT_HYPERPARAMETERS, **hp}
        # Normalise logvar_clamp to a list (json doesn't preserve tuples).
        if 'logvar_clamp' in merged and merged['logvar_clamp'] is not None:
            merged['logvar_clamp'] = list(merged['logvar_clamp'])
        return merged
    return dict(DEFAULT_HYPERPARAMETERS)


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
        print(f'[manifest] ENSEMBLE_ROOT does not exist: {ENSEMBLE_ROOT}', file=sys.stderr)
        sys.exit(1)

    members = []
    per_member_hps = []     # accumulator to detect mismatched configs
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
            print(
                f'[manifest] WARN: could not find CHECKPOINT_PATH in {log_path}',
                file=sys.stderr,
            )
            continue

        ckpt_abs = (REPO_ROOT / ckpt).resolve() if not os.path.isabs(ckpt) else Path(ckpt)
        ckpt_exists = ckpt_abs.is_file()
        if not ckpt_exists:
            print(
                f'[manifest] WARN: checkpoint not on disk for member {idx}: {ckpt}',
                file=sys.stderr,
            )

        members.append({
            'index': idx,
            'seed': seed,
            'directory': str(entry.relative_to(REPO_ROOT)),
            'checkpoint': ckpt,
            'checkpoint_exists': ckpt_exists,
            'log': str(log_path.relative_to(REPO_ROOT)),
        })
        per_member_hps.append(load_member_hyperparameters(entry))

    members.sort(key=lambda d: d['index'])

    # Hyperparameters: prefer the per-member files, but warn if they disagree.
    if per_member_hps:
        canonical_hp = per_member_hps[0]
        # Quick sanity: keys that should match across members.
        check_keys = ['logvar_clamp', 'logvar_l2', 'epoch', 'topk', 'dim',
                      'slide_win', 'slide_stride', 'batch']
        for i, hp in enumerate(per_member_hps[1:], start=1):
            for k in check_keys:
                if hp.get(k) != canonical_hp.get(k):
                    print(f'[manifest] WARN: member {i} {k}={hp.get(k)} but '
                          f'member 0 {k}={canonical_hp.get(k)}', file=sys.stderr)
    else:
        canonical_hp = dict(DEFAULT_HYPERPARAMETERS)

    manifest = {
        'model': 'gdn_uq',
        'dataset': 'swat',
        'M': len(members),
        'members': members,
        'hyperparameters': canonical_hp,
    }

    out_path = ENSEMBLE_ROOT / 'manifest.json'
    with out_path.open('w') as f:
        json.dump(manifest, f, indent=2)
    print(f'[manifest] wrote {out_path} ({len(members)} members)')

    if any(not m['checkpoint_exists'] for m in members):
        sys.exit(2)


if __name__ == '__main__':
    main()
