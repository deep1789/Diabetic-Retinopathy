import json, numpy as np
R = json.load(open('/home/user/work/res/ablate_evidence.json')); ref = np.array([r['qwk'] for r in R['Full evidence (reference)']])
L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Evidence ablation: the full \mname\ head trained with parts of the evidence tensor removed (zones or lesion classes set to zero everywhere, including in the rule scores). Three seeds (0--2), mean $\pm$ SD, leak-controlled test set. $\Delta$ is the mean QWK difference to the reference. With three seeds and SDs of 0.005--0.02, differences below about 0.01 are not distinguishable from noise.}', r'\label{tab:evablate}', r'\small', r'\resizebox{\textwidth}{!}{%',
     r'\begin{tabular}{@{}lcccc@{}}', r'\toprule', r'Evidence kept & QWK & $\Delta$QWK & Acc. & ECE \\', r'\midrule']
for n, runs in R.items():
    q = np.array([r['qwk'] for r in runs]); a = np.array([r['acc'] for r in runs]); e = np.array([r['ece'] for r in runs])
    L.append(f"{n} & {q.mean():.3f}$\\pm${q.std():.3f} & {'--' if 'reference' in n else f'{q.mean() - ref.mean():+.3f}'} & {a.mean():.3f}$\\pm${a.std():.3f} & {e.mean():.3f}$\\pm${e.std():.3f} \\\\")
L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']; open('/home/user/Diabetic-Retinopathy/paper/tables/evablate.tex', 'w').write('\n'.join(L) + '\n')
print({n: round(float(np.mean([r['qwk'] for r in runs])), 3) for n, runs in R.items()}, {n: round(float(np.std([r['qwk'] for r in runs])), 3) for n, runs in R.items()})
