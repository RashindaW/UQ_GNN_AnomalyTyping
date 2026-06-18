import numpy as np, glob, os
ref = np.load('results/swat_gdeltauq_sw60_paper_protocol_K100/0516-031655/arrays.npz')
ref_label = ref['test_attack_label'].astype(np.int8)
Tref = ref_label.shape[0]
d = os.path.dirname(sorted(glob.glob('competitors/GTA/results/*seed1_*/pred.npy'))[0])
print('GTA dir:', d)
print('files:', sorted(os.listdir(d)))
lab = np.asarray(np.load(d + '/label.npy')).reshape(-1).astype(np.int8)
L = len(lab)
print('gta label len:', L, ' ref:', Tref, ' diff:', Tref - L)
best = None
for off in range(0, Tref - L + 1):
    if np.array_equal(lab, ref_label[off:off + L]):
        best = off
        break
print('exact-match offset =', best)
if best is not None:
    print('  front_pad =', best, ' back_pad =', Tref - L - best)
print('matches ref[0:L]? ', bool(np.array_equal(lab, ref_label[:L])))
print('matches ref[-L:]? ', bool(np.array_equal(lab, ref_label[Tref - L:])))
print('gta attack_rate=%.4f  ref attack_rate=%.4f' % (lab.mean(), ref_label.mean()))
