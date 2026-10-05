"""Classical (non-learned) baseline: top-hat filtering of the green channel -> pseudo-probability maps, evaluated with the same protocol."""
import json, numpy as np, cv2
from metrics_seg import full_eval
W = '/home/user/work/'
K = dict(MA=7, HE=21, EX=15, SE=41)
def maps(X):
    out = np.zeros((len(X), 4, 512, 512), np.uint8)
    for n, im in enumerate(X):
        g = im[..., 1]; fov = cv2.erode((im.max(2) > 12).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        for c, (nme, dark) in enumerate([('MA', 1), ('HE', 1), ('EX', 0), ('SE', 0)]):
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K[nme], K[nme]))
            th = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT if dark else cv2.MORPH_TOPHAT, ker).astype(np.float32)
            th = cv2.GaussianBlur(th, (0, 0), 1.0); th[~fov] = 0
            out[n, c] = np.clip(th / (np.percentile(th[fov], 99.7) + 1e-6), 0, 1) * 255
    return out
Pv = maps(np.load(W + 'les_valid_X.npy')); Pt = maps(np.load(W + 'les_test_X.npy'))
Yv = np.load(W + 'les_valid_Y.npy'); Yt = np.load(W + 'les_test_Y.npy')
res = full_eval(Pv, Yv, Pt, Yt, np.load(W + 'les_test_B.npy', allow_pickle=True)); res['variant'] = 'morph'
json.dump(res, open(W + 'res/seg_morph.json', 'w'), indent=1); np.save(W + 'probs_test_morph.npy', Pt)
print('AUPR', [round(a, 3) for a in res['aupr']], 'Dice', [round(a, 3) for a in res['dice']])
