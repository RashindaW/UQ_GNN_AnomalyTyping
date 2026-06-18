"""Diagnose TopoGDN baseline>M10 reversal: where does MC variance explode?"""
import numpy as np, glob, os
np.set_printoptions(suppress=True)

def load2d(p):
    a = np.asarray(np.load(p), dtype=np.float64)
    if a.ndim == 3:
        a = a[:, 0, :] if a.shape[1] == 1 else (a[:, :, 0] if a.shape[2] == 1 else a.reshape(a.shape[0], -1))
    return a

print(f"{'seed':>4} {'detF1?':>6} | {'det_pred rng':>22} {'mc_pred rng':>22} | "
      f"{'mc_upar mean':>13} {'mc_upar max':>13} {'#nan':>5} {'#inf':>5} {'det~mc MAE':>10}")
for S in [1, 2, 3, 42, 100]:
    D = f'competitors/TopoGDN/results_canon/swat_seed{S}'
    det = load2d(f'{D}/test_pred.npy')      # deterministic (dropout OFF)
    tru = load2d(f'{D}/test_true.npy')
    mc  = load2d(f'{D}/mc_pred.npy')         # MC mean (dropout ON)
    up  = load2d(f'{D}/mc_upar.npy')
    # align lengths (det may be full T, mc same)
    n = min(len(det), len(mc), len(tru), len(up))
    det, tru, mc, up = det[:n], tru[:n], mc[:n], up[:n]
    nan = int(np.isnan(mc).sum() + np.isnan(up).sum())
    inf = int(np.isinf(mc).sum() + np.isinf(up).sum())
    det_mc_mae = np.nanmean(np.abs(det - mc))
    print(f"{S:>4} {'':>6} | [{det.min():9.2f},{det.max():9.2f}] [{mc.min():9.2f},{mc.max():9.2f}] | "
          f"{np.nanmean(up):13.3e} {np.nanmax(up):13.3e} {nan:5d} {inf:5d} {det_mc_mae:10.3f}")

print("\n=== per-seed: which sensors carry the U_par blow-up (top-3 by mean var) ===")
for S in [2, 3]:
    D = f'competitors/TopoGDN/results_canon/swat_seed{S}'
    up = load2d(f'{D}/mc_upar.npy')
    per_sensor = np.nanmean(up, axis=0)
    top = np.argsort(per_sensor)[::-1][:5]
    print(f"  seed{S}: top sensors {list(top)}  var means {per_sensor[top].round(1)}")
    # what fraction of timesteps are 'normal' magnitude vs blown up?
    ts_max = np.nanmax(up, axis=1)
    print(f"          ts var max: median={np.median(ts_max):.2e}  p99={np.percentile(ts_max,99):.2e}  max={ts_max.max():.2e}")
    print(f"          frac timesteps with max-var > 100: {(ts_max>100).mean():.4f}")

print("\n=== compare MC-mean forecast quality vs deterministic (per-seed test MSE) ===")
for S in [1, 2, 3, 42, 100]:
    D = f'competitors/TopoGDN/results_canon/swat_seed{S}'
    det = load2d(f'{D}/test_pred.npy'); tru = load2d(f'{D}/test_true.npy'); mc = load2d(f'{D}/mc_pred.npy')
    n = min(len(det), len(mc), len(tru)); det, tru, mc = det[:n], tru[:n], mc[:n]
    mse_det = np.nanmean((det - tru) ** 2); mse_mc = np.nanmean((mc - tru) ** 2)
    print(f"  seed{S}: MSE deterministic={mse_det:.4f}  MSE MC-mean={mse_mc:.4f}  ratio={mse_mc/max(mse_det,1e-9):.2f}x")
