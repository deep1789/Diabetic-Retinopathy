"""Compare the end-to-end fine-tuned ResNet-18 reference with the frozen-embedding models (paired bootstrap on the shared test set)."""
import json, numpy as np
W = '/home/user/work/'; P = '/home/user/Diabetic-Retinopathy/paper/'
ft = json.load(open(W + 'res/ft_baseline.json')); H = json.load(open(W + 'res/heads_ftlbg_convnext_tiny.json')); yt = np.array(H['yt'])
A = json.load(open(W + 'res/analysis_ftlbg_convnext_tiny.json'))
assert ft['n'] == len(yt)
def qwk(y, p, K=5):
    O = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float); a = O.sum(1); b = O.sum(0); E = np.outer(a, b) / O.sum()
    w = (np.arange(K)[:, None] - np.arange(K)[None]) ** 2 / (K - 1) ** 2; return 1 - (w * O).sum() / (w * E).sum()
pf = np.array(ft['probs']).argmax(1); names = {'main': 'Ours (full: C1+C2+C5)', 'b2': 'B2 Ordinal (img emb.)'}
pm = np.mean([np.array(r['probs']) for r in H['res'][names['main']]], 0).argmax(1); pb = np.mean([np.array(r['probs']) for r in H['res'][names['b2']]], 0).argmax(1)
rng = np.random.default_rng(0); I = rng.integers(0, len(yt), (2000, len(yt)))
q = {k: np.array([qwk(yt[i], p[i]) for i in I]) for k, p in [('ft', pf), ('main', pm), ('b2', pb)]}
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
d1 = q['main'] - q['ft']; d2 = q['b2'] - q['ft']
rows = [('End-to-end fine-tuned ResNet-18 (1 run, 8 epochs, $224^2$)', f"{ft['params'] / 1e6:.1f}", ft['qwk'], ci(q['ft']), ft['acc'], ft['f1'], ft['auc_ref'], ft['ece']),
        (r'\mname\ (frozen ConvNeXt-T + evidence, 5 seeds)', '0.14 (head)', A['agg'][names['main']]['qwk'][0], A['ci'][names['main']], A['agg'][names['main']]['acc'][0], A['agg'][names['main']]['f1'][0], A['agg'][names['main']]['auc_ref'][0], A['agg'][names['main']]['ece'][0]),
        ('B2: frozen ConvNeXt-T, ordinal (5 seeds)', '0.30 (head)', A['agg'][names['b2']]['qwk'][0], A['ci'][names['b2']], A['agg'][names['b2']]['acc'][0], A['agg'][names['b2']]['f1'][0], A['agg'][names['b2']]['auc_ref'][0], A['agg'][names['b2']]['ece'][0])]
L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{End-to-end fine-tuned CNN reference versus frozen-embedding models on the leak-controlled test set. The fine-tuned model is a single run (trainable parameters in millions); the others are seed averages. Intervals: 95\% bootstrap over test images.}', r'\label{tab:ft}', r'\small', r'\resizebox{\textwidth}{!}{%',
     r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule', r'Model & Params (M) & QWK [95\% CI] & Acc. & F1 & AUC$_{\ge2}$ & ECE \\', r'\midrule']
for n, p, qk, c, a, f, u, e in rows: L.append(f"{n} & {p} & {qk:.3f} [{c[0]:.3f}, {c[1]:.3f}] & {a:.3f} & {f:.3f} & {u:.3f} & {e:.3f} \\\\")
L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']; open(P + 'tables/ft.tex', 'w').write('\n'.join(L) + '\n')
N = dict(FTQWK=f"{ft['qwk']:.3f}", FTAcc=f"{100 * ft['acc']:.1f}", FTAUC=f"{ft['auc_ref']:.3f}", FTECE=f"{ft['ece']:.3f}", FTdMainMean=f"{d1.mean():+.3f}", FTdMainLo=f"{np.percentile(d1, 2.5):+.3f}", FTdMainHi=f"{np.percentile(d1, 97.5):+.3f}",
         FTdBtwoMean=f"{d2.mean():+.3f}", FTdBtwoLo=f"{np.percentile(d2, 2.5):+.3f}", FTdBtwoHi=f"{np.percentile(d2, 97.5):+.3f}")
open(P + 'numbers_ft.tex', 'w').write('\n'.join(r'\newcommand{\n' + k + '}{' + v + '}' for k, v in N.items()) + '\n'); print(N)
