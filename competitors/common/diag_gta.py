"""Diagnose GTA's suspiciously low baseline F1 (0.464, PA%K 0.534)."""
import numpy as np, glob, os
np.set_printoptions(suppress=True)

def load2d(p):
    a = np.asarray(np.load(p), dtype=np.float64)
    if a.ndim == 3:
        a = a[:, 0, :] if a.shape[1] == 1 else (a[:, :, 0] if a.shape[2] == 1 else a.reshape(a.shape[0], -1))
    return a

ref = np.load('results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655/arrays.npz')
ref_mu = ref['test_mu_bar']; ref_gt = ref['test_ground_truth']; ref_lab = ref['test_attack_label']
print(f"GDN ref: mu rng [{ref_mu.min():.2f},{ref_mu.max():.2f}] gt rng [{ref_gt.min():.2f},{ref_gt.max():.2f}] "
      f"MSE={np.mean((ref_mu-ref_gt)**2):.4f}")
print()
print(f"{'seed':>4} | {'metrics.npy [mae,mse,..]':>34} | {'pred rng':>20} {'true rng':>20} {'MSE':>8} {'corr':>6}")
for S in [1, 2, 3, 42, 100]:
    D = sorted(glob.glob(f'competitors/GTA/results/*seed{S}_*/'))[0]
    m = np.load(D + 'metrics.npy')
    pred = load2d(D + 'pred.npy'); true = load2d(D + 'true.npy')
    n = min(len(pred), len(true)); pred, true = pred[:n], true[:n]
    mse = np.mean((pred - true) ** 2)
    corr = np.corrcoef(pred.ravel(), true.ravel())[0, 1]
    print(f"{S:>4} | {np.array2string(m, precision=4, max_line_width=200):>34} | "
          f"[{pred.min():8.2f},{pred.max():7.2f}] [{true.min():8.2f},{true.max():7.2f}] {mse:8.4f} {corr:6.3f}")

print("\n=== GTA scale check: is pred/true standardized (mean~0,std~1) or raw? ===")
D = sorted(glob.glob('competitors/GTA/results/*seed1_*/'))[0]
pred = load2d(D + 'pred.npy'); true = load2d(D + 'true.npy')
print(f"  true: mean={true.mean():.3f} std={true.std():.3f} | pred: mean={pred.mean():.3f} std={pred.std():.3f}")
print(f"  GDN ref gt: mean={ref_gt.mean():.3f} std={ref_gt.std():.3f}  <- comparison scale")

print("\n=== residual separation: does GTA err discriminate attack vs normal? ===")
# emulate harness: per-feature abs err, but compare attack vs normal magnitude
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
lab = ref_lab[:len(pred)].astype(bool)
err = np.abs(pred - true)              # (T,V)
err_t = err.mean(axis=1)              # per-timestep mean abs err
print(f"  GTA mean|err|  normal={err_t[~lab].mean():.4f}  attack={err_t[lab].mean():.4f}  "
      f"ratio={err_t[lab].mean()/max(err_t[~lab].mean(),1e-9):.2f}x")
err_max = err.max(axis=1)
print(f"  GTA max|err|   normal={err_max[~lab].mean():.4f}  attack={err_max[lab].mean():.4f}  "
      f"ratio={err_max[lab].mean()/max(err_max[~lab].mean(),1e-9):.2f}x")
# GDN ref for comparison
refl = ref_lab.astype(bool); refe = np.abs(ref_mu - ref_gt)
refe_t = refe.mean(axis=1)
print(f"  GDN mean|err|  normal={refe_t[~refl].mean():.4f}  attack={refe_t[refl].mean():.4f}  "
      f"ratio={refe_t[refl].mean()/max(refe_t[~refl].mean(),1e-9):.2f}x")
