import json, numpy as np
d = json.load(open('/home/user/Diabetic-Retinopathy/paper/stats_ddr.json'))
W = '/home/user/work/'
g = d['grade_counts']; b = d['boxes']; im = d['imgs_with_lesion']
nl = {sp: len(np.load(W + f'les_{sp}_names.npy')) for sp in ['train', 'valid', 'test']}
f = lambda x: f'{x:,}'.replace(',', '{,}')
L = [r'\begin{table}[t]', r'\centering', r'\caption{DDR dataset statistics as verified from the downloaded archive. Top: grading labels per split (G0--G4 = ICDR grades, UG = ungradable). Bottom: lesion subset (unique boxes after duplicate removal; number of images containing each lesion class in parentheses).}', r'\label{tab:ddr}', r'\small',
     r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule', r'Split & G0 & G1 & G2 & G3 & G4 & UG & Total \\', r'\midrule']
for sp, nm in [('train', 'Train'), ('valid', 'Validation'), ('test', 'Test')]:
    row = [g[sp][str(k)] for k in range(6)]; L.append(f'{nm} & ' + ' & '.join(f(x) for x in row) + f' & {f(sum(row))} \\\\')
tot = [sum(g[sp][str(k)] for sp in g) for k in range(6)]
L += [r'\midrule', 'All & ' + ' & '.join(f(x) for x in tot) + f' & {f(sum(tot))} \\\\', r'\bottomrule', r'\end{tabular}', r'', r'\vspace{4pt}', r'', r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
      r'Lesion split & Images & MA & HE & EX & SE \\', r'\midrule']
for sp, nm in [('train', 'Train'), ('valid', 'Validation'), ('test', 'Test')]:
    L.append(f'{nm} & {nl[sp]} & ' + ' & '.join(f'{f(b[sp][c])} ({im[sp][c]})' for c in range(4)) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('/home/user/Diabetic-Retinopathy/paper/tables/ddr_stats.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
