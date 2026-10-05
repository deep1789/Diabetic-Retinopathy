"""Evidence ablation: remove zones or lesion classes from the evidence tensor (3 seeds, full LesionRule head). Writes res/ablate_evidence.json"""
import sys, os, json, numpy as np, torch
_a = sys.argv; sys.argv = ['x', 'ftlbg', 'convnext_tiny']; import heads as H; sys.argv = _a
from sklearn.metrics import cohen_kappa_score
W = '/home/user/work/'; SEG, BB = 'ftlbg', 'convnext_tiny'
meta = np.load(W + 'g_ddr_meta.npy', allow_pickle=True); sp = np.array([m[0] for m in meta]); yy = np.array([m[2] for m in meta]); nm = np.array([m[1].rsplit('.', 1)[0] for m in meta])
P = np.load(W + f'g_ddr_{SEG}_maps.npy'); E = np.load(W + f'g_ddr_feat_{BB}.npy'); Z = H.evidence(P); good = yy < 5
mu, sd = E[sp == 'train'].mean(0), E[sp == 'train'].std(0) + 1e-6; E = torch.from_numpy((E - mu) / sd).float()
seen = set(np.load(W + 'les_train_names.npy')) | set(np.load(W + 'les_valid_names.npy')); leak = np.array([n in seen for n in nm])
def sub(s, Zm): k = (sp == s) & good; return E[k], Zm[k], torch.from_numpy(yy[k]).long()
Evx = sub('valid', Z)[0]; pm = torch.from_numpy(np.random.default_rng(0).permutation(len(Evx))); h = len(pm) // 2
cfg = dict(arch='et', loss='ord', sigma=0.5, lam=1.0)
ABL = {'Full evidence (reference)': ([0, 1, 2, 3], [0, 1, 2, 3, 4, 5]), 'Whole-field zone only': ([0, 1, 2, 3], [5]), 'Quadrants + centre (no whole field)': ([0, 1, 2, 3], [0, 1, 2, 3, 4]),
       'Without MA': ([1, 2, 3], [0, 1, 2, 3, 4, 5]), 'Without HE': ([0, 2, 3], [0, 1, 2, 3, 4, 5]), 'Without EX': ([0, 1, 3], [0, 1, 2, 3, 4, 5]), 'Without SE': ([0, 1, 2], [0, 1, 2, 3, 4, 5]),
       'MA only': ([0], [0, 1, 2, 3, 4, 5]), 'HE only': ([1], [0, 1, 2, 3, 4, 5]), 'EX only': ([2], [0, 1, 2, 3, 4, 5]), 'SE only': ([3], [0, 1, 2, 3, 4, 5])}
res = {}
for name, (cl, zn) in ABL.items():
    mask = torch.zeros(1, 4, 6, 1); [mask.__setitem__((0, c, z, 0), 1.) for c in cl for z in zn]; Zm = Z * mask
    Ev_, Zv_, yv_ = sub('valid', Zm)
    D = dict(tr=sub('train', Zm), va=(Ev_[pm[:h]], Zv_[pm[:h]], yv_[pm[:h]]), vb=(Ev_[pm[h:]], Zv_[pm[h:]], yv_[pm[h:]]), te=sub('test', Zm), clean=torch.from_numpy(~leak[(sp == 'test') & good]))
    D['R'] = H.fit_rules(D['tr'][1], D['tr'][2]); runs = []
    for s in range(3):
        m, ov, ot, ob = H.run(cfg, s, D); T = H.fit_T(ov, D['va'][2], cfg); pt = H.class_probs(ot, cfg, T)[D['clean']]; yt = D['te'][2][D['clean']].numpy()
        runs.append(dict(qwk=cohen_kappa_score(yt, pt.argmax(1).numpy(), weights='quadratic'), acc=float((pt.argmax(1).numpy() == yt).mean()), ece=H.ece(pt, torch.from_numpy(yt))))
    res[name] = runs; print(name, {k: round(float(np.mean([r[k] for r in runs])), 4) for k in runs[0]}, flush=True)
json.dump(res, open(W + 'res/ablate_evidence.json', 'w'))
