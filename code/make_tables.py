"""Generate all LaTeX tables and number macros from result JSON files. usage: python make_tables.py <seg> <bb> <main_seg_variant>"""
import sys, json, os, numpy as np
W = '/home/user/work/'; T = '/home/user/Diabetic-Retinopathy/paper/tables/'
SEG, BB = sys.argv[1], sys.argv[2]; MAINSEG = sys.argv[3] if len(sys.argv) > 3 else SEG
CN = ['MA', 'HE', 'EX', 'SE']
SV = [('morph', 'Top-hat + threshold (classical)'), ('bce', r'U-Net, BCE'), ('bcedice', r'U-Net, BCE+Dice'), ('ftl', r'U-Net, focal Tversky (ours)'), ('ftlbg', r'U-Net, focal Tversky, GF input (ours)')]
f3 = lambda x: f'{x:.3f}'
def bold(vals, i, fmt=f3, hi=True):
    best = max(vals) if hi else min(vals); s = fmt(vals[i]); return r'\textbf{' + s + '}' if abs(vals[i] - best) < 1e-12 else s
def write(name, s): open(T + name, 'w').write(s); print('wrote', name)

# ------------------------------------------------------------------ segmentation
S = {v: json.load(open(W + f'res/seg_{v}.json')) for v, _ in SV if os.path.exists(W + f'res/seg_{v}.json')}
rows = [(v, n) for v, n in SV if v in S]
mean = lambda l: float(np.mean(l))
cols = {'AP': [[*S[v]['aupr'], mean(S[v]['aupr'])] for v, _ in rows], 'D': [[*S[v]['dice'], mean(S[v]['dice'])] for v, _ in rows], 'I': [[*S[v]['iou'], mean(S[v]['iou'])] for v, _ in rows]}
L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Lesion segmentation on the DDR test split (225 images, $512\times512$). AUPR is threshold-free; Dice and IoU are computed at the per-class threshold that maximises validation Dice and accumulated over all test pixels. GF = Gaussian-filtered input. Single training run per row (seed 0); best value per column in bold.}', r'\label{tab:seg}', r'\small', r'\setlength{\tabcolsep}{3.2pt}',
     r'\begin{tabular}{@{}lccccc|ccccc|c@{}}', r'\toprule', r' & \multicolumn{5}{c|}{AUPR} & \multicolumn{5}{c|}{Dice} & mIoU \\', r'Method & MA & HE & EX & SE & Mean & MA & HE & EX & SE & Mean & \\', r'\midrule']
for r, (v, n) in enumerate(rows):
    cells = [bold([cols['AP'][q][k] for q in range(len(rows))], r) for k in range(5)] + [bold([cols['D'][q][k] for q in range(len(rows))], r) for k in range(5)] + [bold([cols['I'][q][4] for q in range(len(rows))], r)]
    L.append(n + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']; write('seg.tex', '\n'.join(L) + '\n')

# ------------------------------------------------------------------ detection
L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Lesion detection on the DDR test split from segmentation-derived proposals (Alg.~\ref{alg:det}). AP$_{0.5}$ / AP$_{0.3}$ are average precision at IoU 0.5 / 0.3; the last column is the mean of AP$_{0.3}$ over the four classes. The ground truth contains 2041 MA, 6457 HE, 8320 EX and 214 SE boxes.}', r'\label{tab:det}', r'\small', r'\setlength{\tabcolsep}{3.5pt}',
     r'\begin{tabular}{@{}lcccccccc|c@{}}', r'\toprule', r' & \multicolumn{2}{c}{MA} & \multicolumn{2}{c}{HE} & \multicolumn{2}{c}{EX} & \multicolumn{2}{c|}{SE} & \\', r'Proposals from & AP$_{.5}$ & AP$_{.3}$ & AP$_{.5}$ & AP$_{.3}$ & AP$_{.5}$ & AP$_{.3}$ & AP$_{.5}$ & AP$_{.3}$ & mAP$_{.3}$ \\', r'\midrule']
m3 = [mean([S[v]['det'][c]['ap30'] for c in CN]) for v, _ in rows]
for r, (v, n) in enumerate(rows):
    cells = []
    for c in CN: cells += [f3(S[v]['det'][c]['ap50']), f3(S[v]['det'][c]['ap30'])]
    L.append(n + ' & ' + ' & '.join(cells) + ' & ' + bold(m3, r) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']; write('det.tex', '\n'.join(L) + '\n')
fp = ['1', '2', '4', '8', '16']
L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Lesion-level FROC sensitivity of segmentation-derived detections (a prediction is a hit if its centre lies inside a ground-truth box) at 1, 2, 4, 8 and 16 false positives per image (FP/img), DDR test split.}', r'\label{tab:froc}', r'\small',
     r'\begin{tabular}{@{}llccccc@{}}', r'\toprule', r'Proposals from & Class & ' + ' & '.join(fp) + r' \\', r'\midrule']
for v, n in rows:
    if v not in ('morph', 'bce', MAINSEG): continue
    for k, c in enumerate(CN): L.append((n if k == 0 else '') + f' & {c} & ' + ' & '.join(f3(S[v]['det'][c]['froc'][x]) for x in fp) + r' \\')
    L.append(r'\addlinespace')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']; write('froc.tex', '\n'.join(L) + '\n')

# ------------------------------------------------------------------ grading
AP = W + f'res/analysis_{SEG}_{BB}.json'
if os.path.exists(AP):
    A = json.load(open(AP)); names = list(A['agg']); main = A['main']
    pm = lambda n, k, d=3: f"{A['agg'][n][k][0]:.{d}f}$\\pm${A['agg'][n][k][1]:.{d}f}"
    short = {'B1 CE (img emb.)': 'B1: embedding, CE', 'B2 Ordinal (img emb.)': 'B2: embedding, ordinal', 'B3 Lesion evidence only': 'B3: evidence only', 'B4 Concat (emb.+evidence)': 'B4: embedding + evidence (concat.)',
             'Ours: tokens, no rule/soft': 'T0: evidence tokens', 'Ours: + soft labels (C5)': 'T1: T0 + soft labels', 'Ours: + rule loss (C2)': 'T2: T0 + rule loss',
             'Ours (full: C1+C2+C5)': r'\textbf{\mname\ (full)}', 'Abl: full w/o evidence tokens': r'A1: \mname\ w/o evidence tokens'}
    L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Five-class DR grading on the leak-controlled DDR test set (' + str(A['n_test']) + r' gradable images), mean $\pm$ SD over 5 seeds. QWK 95\% CI: bootstrap over test images of the seed-averaged probabilities. $\Delta$QWK: paired bootstrap difference (\mname\ minus model), with 95\% interval; intervals excluding 0 are marked $^{*}$. Rule viol.: fraction of test images whose predicted grade is not licensed by the frozen rule scores (Section~\ref{sec:rule}).}', r'\label{tab:grading}', r'\footnotesize', r'\setlength{\tabcolsep}{2.8pt}',
         r'\begin{tabular}{@{}lcccccccc@{}}', r'\toprule', r'Model & QWK & 95\% CI & Acc. & F1 & AUC$_{\ge2}$ & ECE & Rule viol. & $\Delta$QWK \\', r'\midrule']
    for n in names:
        d = A['diff'].get(n); dd = '--' if d is None else f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" + ('$^{*}$' if (d['lo'] > 0 or d['hi'] < 0) else '')
        ci = A['ci'][n]; L.append((short.get(n, n.replace('&', r'\&').replace('_', r'\_'))) + f" & {pm(n, 'qwk')} & [{ci[0]:.3f}, {ci[1]:.3f}] & {pm(n, 'acc')} & {pm(n, 'f1')} & {pm(n, 'auc_ref')} & {pm(n, 'ece')} & {pm(n, 'viol')} & {dd} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']; write('grading.tex', '\n'.join(L) + '\n')
    # six-class
    L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Six-class protocol (ungradable images included) on the leak-controlled test set: a separate quality gate decides ``ungradable''; otherwise the ordinal grade is output. Mean $\pm$ SD over 5 seeds.}', r'\label{tab:six}', r'\small',
         r'\begin{tabular}{@{}lccc@{}}', r'\toprule', r'Model & Accuracy & Macro-F1 & Gate AUC \\', r'\midrule']
    for n in names: L.append(short.get(n, n) + f" & {pm(n, 'acc6')} & {pm(n, 'f1_6')} & {pm(n, 'gate_auc')} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']; write('six.tex', '\n'.join(L) + '\n')
    # conformal
    L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Split-conformal prediction sets (calibration: validation half B, ' + str(A['n_calib']) + r' images) evaluated on the leak-controlled test set. Coverage should be $\ge1-\alpha$. ``Refer'' = set contains any grade $\ge2$; sensitivity/specificity are for referable DR ($y\ge2$). Class-wise coverage shows that the guarantee is marginal.}', r'\label{tab:conformal}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{@{}llcccccc@{}}', r'\toprule', r'Model & $\alpha$ & Coverage & Set size & Singleton & Refer sens. & Refer spec. & Cov. G0/G1/G2/G3/G4 \\', r'\midrule']
    for n, cv in A['conformal'].items():
        for i, c in enumerate(cv): L.append((short.get(n, n.replace('&', r'\&')) if i == 0 else '') + f" & {c['alpha']} & {c['coverage']:.3f} & {c['size']:.2f} & {c['singleton']:.2f} & {c['ref_sens']:.3f} & {c['ref_spec']:.3f} & {'/'.join(f'{x:.2f}' for x in c['cov_by_class'])} \\\\")
        L.append(r'\addlinespace')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']; write('conformal.tex', '\n'.join(L) + '\n')
    # leak sensitivity
    L = [r'\begin{table}[h]', r'\centering', r'\caption{Sensitivity to the leak control: QWK (mean $\pm$ SD over 5 seeds) on the leak-controlled test set (3633 images) versus the full official gradable test partition (3759 images).}', r'\label{tab:sens}', r'\small',
         r'\begin{tabular}{@{}lcc@{}}', r'\toprule', r'Model & Leak-controlled & Full official test \\', r'\midrule']
    for n in names: L.append(short.get(n, n.replace('&', r'\&')) + f" & {pm(n, 'qwk')} & {pm(n, 'qwk_full')} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']; write('sens.tex', '\n'.join(L) + '\n')
    # external
    if 'ex_qwk' in A['agg'][main]:
        L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Zero-shot external test on the Gaussian-filtered APTOS-derived Kaggle set (3664 images, $224\times224$, grades 0--4): models trained on DDR only, no adaptation (mean $\pm$ SD over 5 seeds). Lesion evidence is computed by the segmentation network applied to the up-sampled images.}', r'\label{tab:external}', r'\small',
             r'\begin{tabular}{@{}lccccc@{}}', r'\toprule', r'Model & QWK & Acc. & F1 & AUC$_{\ge2}$ & Rule viol. \\', r'\midrule']
        for n in names: L.append(short.get(n, n.replace('&', r'\&')) + f" & {pm(n, 'ex_qwk')} & {pm(n, 'ex_acc')} & {pm(n, 'ex_f1')} & {pm(n, 'ex_auc')} & {pm(n, 'ex_viol')} \\\\")
        L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']; write('external.tex', '\n'.join(L) + '\n')
    # numbers macros
    g = A['agg'][main]; b2 = A['agg'].get('B2 Ordinal (img emb.)'); b1 = A['agg'].get('B1 CE (img emb.)')
    N = {'MainQWK': f"{g['qwk'][0]:.3f}", 'MainQWKsd': f"{g['qwk'][1]:.3f}", 'MainCIlo': f"{A['ci'][main][0]:.3f}", 'MainCIhi': f"{A['ci'][main][1]:.3f}", 'MainAcc': f"{100 * g['acc'][0]:.1f}", 'MainAUC': f"{g['auc_ref'][0]:.3f}",
         'MainECE': f"{g['ece'][0]:.3f}", 'MainViol': f"{100 * g['viol'][0]:.1f}", 'BtwoQWK': f"{b2['qwk'][0]:.3f}", 'BtwoViol': f"{100 * b2['viol'][0]:.1f}", 'BoneQWK': f"{b1['qwk'][0]:.3f}", 'BtwoECE': f"{b2['ece'][0]:.3f}",
         'BoneECE': f"{b1['ece'][0]:.3f}", 'NTest': str(A['n_test'])}
    cv = A['conformal'][main]; N.update({'ConfCovTen': f"{cv[1]['coverage']:.3f}", 'ConfSizeTen': f"{cv[1]['size']:.2f}", 'ConfSensTen': f"{cv[1]['ref_sens']:.3f}", 'ConfSpecTen': f"{cv[1]['ref_spec']:.3f}"})
    if 'ex_qwk' in g: N.update({'ExQWK': f"{g['ex_qwk'][0]:.3f}", 'ExAUC': f"{g['ex_auc'][0]:.3f}", 'ExBtwoQWK': f"{b2['ex_qwk'][0]:.3f}"})
    for v in S:
        key = {'morph': 'Morph', 'bce': 'Bce', 'bcedice': 'Bcedice', 'ftl': 'Ftl', 'ftlbg': 'Ftlbg'}[v]
        N['SegAUPR' + key] = f"{mean(S[v]['aupr']):.3f}"; N['SegDice' + key] = f"{mean(S[v]['dice']):.3f}"; N['DetMap' + key] = f"{mean([S[v]['det'][c]['ap30'] for c in CN]):.3f}"
    write('../numbers.tex', '\n'.join(r'\newcommand{\n' + k + '}{' + v + '}' for k, v in N.items()) + '\n')
