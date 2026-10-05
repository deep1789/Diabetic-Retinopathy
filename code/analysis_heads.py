"""Aggregate head results: mean/SD over seeds, bootstrap CIs, paired differences, calibration, conformal referral. usage: python analysis_heads.py <seg> <bb> [main_model_name]"""
import sys, os, json, numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
W = os.environ.get('DRW', '/home/user/work/'); SEG, BB = sys.argv[1], sys.argv[2]; MAIN = sys.argv[3] if len(sys.argv) > 3 else 'Ours (full: C1+C2+C5)'
R = json.load(open(W + f'res/heads_{SEG}_{BB}.json')); res = R['res']; yt = np.array(R['yt']); yb = np.array(R['yb'])
KEYS = ['qwk', 'acc', 'f1', 'auc_ref', 'ece', 'viol', 'qwk_full', 'acc6', 'f1_6', 'gate_auc']
KEYS += [k for k in ['ex_qwk', 'ex_acc', 'ex_f1', 'ex_auc', 'ex_viol'] if k in next(iter(res.values()))[0]]
agg = {n: {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))] for k in KEYS} for n, runs in res.items()}
P = {n: np.mean([np.array(r['probs']) for r in runs], 0) for n, runs in res.items()}
PB = {n: np.mean([np.array(r['pb']) for r in runs], 0) for n, runs in res.items()}
rng = np.random.default_rng(0); N = len(yt); IDX = rng.integers(0, N, (2000, N))

def qwk(y, p, K=5):
    O = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float); a = O.sum(1); b = O.sum(0); E = np.outer(a, b) / O.sum()
    w = (np.arange(K)[:, None] - np.arange(K)[None]) ** 2 / (K - 1) ** 2
    return 1 - (w * O).sum() / (w * E).sum()
pred = {n: P[n].argmax(1) for n in P}
boot = {n: np.array([qwk(yt[i], pred[n][i]) for i in IDX]) for n in P}
ci = {n: [float(np.percentile(boot[n], 2.5)), float(np.percentile(boot[n], 97.5))] for n in P}
diff = {n: dict(mean=float((boot[MAIN] - boot[n]).mean()), lo=float(np.percentile(boot[MAIN] - boot[n], 2.5)), hi=float(np.percentile(boot[MAIN] - boot[n], 97.5))) for n in P if n != MAIN}
# paired bootstrap on AUC difference is omitted; DeLong-type comparisons are not made.
# confusion matrix / per-class report for main model
cm = np.zeros((5, 5), int)
for a, b in zip(yt, pred[MAIN]): cm[a, b] += 1
pr, rc, f1, sup = precision_recall_fscore_support(yt, pred[MAIN], labels=range(5), zero_division=0)
adj = float(np.mean(np.abs(yt - pred[MAIN]) <= 1))
# reliability (main + B2)
def rel(p, y, nb=10):
    conf = p.max(1); acc = (p.argmax(1) == y); out = []
    for lo in np.linspace(0, 1, nb + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / nb)
        out.append([float(conf[m].mean()) if m.any() else None, float(acc[m].mean()) if m.any() else None, int(m.sum())])
    return out
reliab = {n: rel(P[n], yt) for n in [MAIN, 'B1 CE (img emb.)', 'B2 Ordinal (img emb.)'] if n in P}
# split-conformal (calibration = validation half B, using seed-averaged calibrated probs)
def conformal(pb, pt, alpha):
    n = len(yb); s = 1 - pb[np.arange(n), yb]; k = int(np.ceil((n + 1) * (1 - alpha))); q = np.sort(s)[min(k, n) - 1]
    sets = (1 - pt) <= q; cov = sets[np.arange(len(yt)), yt]; size = sets.sum(1)
    ref_pred = sets[:, 2:].any(1); ref_true = yt >= 2
    return dict(alpha=alpha, coverage=float(cov.mean()), size=float(size.mean()), singleton=float((size == 1).mean()),
                cov_by_class=[float(cov[yt == c].mean()) for c in range(5)], ref_sens=float(ref_pred[ref_true].mean()), ref_spec=float((~ref_pred[~ref_true]).mean()),
                refer_rate=float(ref_pred.mean()))
conf = {n: [conformal(PB[n], P[n], a) for a in (0.05, 0.1, 0.2)] for n in [MAIN, 'B1 CE (img emb.)', 'B2 Ordinal (img emb.)', 'B4 Concat (emb.+evidence)'] if n in P}
# hard-decision referral for reference (argmax >= 2)
hard = {n: dict(sens=float((pred[n][yt >= 2] >= 2).mean()), spec=float((pred[n][yt < 2] < 2).mean())) for n in P}
# partition shift: calibration half B vs test, and an exchangeable check (random halves of the test set)
shift = {n: dict(valB_acc=float((PB[n].argmax(1) == yb).mean()), test_acc=float((P[n].argmax(1) == yt).mean()), valB_nll=float(-np.log(PB[n][np.arange(len(yb)), yb]).mean()),
                 test_nll=float(-np.log(P[n][np.arange(len(yt)), yt]).mean()), valB_qwk=float(qwk(yb, PB[n].argmax(1))), test_qwk=float(qwk(yt, P[n].argmax(1)))) for n in [MAIN, 'B2 Ordinal (img emb.)']}
def conf_split(p, alpha, reps=300):
    rg = np.random.default_rng(1); cov, size, sens, spec = [], [], [], []
    for _ in range(reps):
        perm = rg.permutation(len(yt)); c, e = perm[:len(perm) // 2], perm[len(perm) // 2:]
        s_ = 1 - p[c, yt[c]]; k = int(np.ceil((len(c) + 1) * (1 - alpha))); q = np.sort(s_)[min(k, len(c)) - 1]; sets = (1 - p[e]) <= q
        cov.append(sets[np.arange(len(e)), yt[e]].mean()); size.append(sets.sum(1).mean()); rp = sets[:, 2:].any(1); rt = yt[e] >= 2
        sens.append(rp[rt].mean()); spec.append((~rp[~rt]).mean())
    return dict(alpha=alpha, coverage=[float(np.mean(cov)), float(np.std(cov))], size=float(np.mean(size)), ref_sens=float(np.mean(sens)), ref_spec=float(np.mean(spec)))
conf_exch = {n: [conf_split(P[n], a_) for a_ in (0.05, 0.1, 0.2)] for n in [MAIN, 'B2 Ordinal (img emb.)', 'B4 Concat (emb.+evidence)'] if n in P}
json.dump(dict(shift=shift, conf_exch=conf_exch, agg=agg, ci=ci, diff=diff, cm=cm.tolist(), per_class=dict(p=pr.tolist(), r=rc.tolist(), f1=f1.tolist(), n=sup.tolist()), adjacent_acc=adj, reliab=reliab, conformal=conf, hard_ref=hard, main=MAIN, n_test=int(N), n_calib=int(len(yb))),
          open(W + f'res/analysis_{SEG}_{BB}.json', 'w'), indent=1)
print('main', MAIN, agg[MAIN]); print('CI', ci[MAIN]); print({n: round(v['mean'], 4) for n, v in diff.items()})
