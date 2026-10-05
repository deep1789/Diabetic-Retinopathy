"""Result figures. usage: python figs_results.py <fig> [args]   (figs: seg_curves, seg_qual, froc_pr, det_qual, grading, evidence, example)"""
import sys, json, os, numpy as np, cv2, torch
from figstyle import *
W = '/home/user/work/'
VAR = [('morph', 'Top-hat (classical)', C['mute']), ('bce', 'U-Net, BCE', C['orange']), ('bcedice', 'U-Net, BCE+Dice', C['aqua']),
       ('ftl', 'U-Net, focal Tversky', C['blue']), ('ftlbg', 'U-Net, focal Tversky + GF', C['violet'])]
CN = ['MA', 'HE', 'EX', 'SE']
which = sys.argv[1]
have = lambda v: os.path.exists(W + f'res/seg_{v}.json')

def seg_curves():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for v, nm, col in VAR:
        p = W + f'seg_{v}_s0_log.json'
        if not os.path.exists(p): continue
        L = json.load(open(p)); ax[0].plot([r['ep'] + 1 for r in L], [r['loss'] for r in L], color=col, lw=1.4, label=nm)
        e = [(r['ep'] + 1, r['val_mean']) for r in L if 'val_mean' in r]; ax[1].plot(*zip(*e), color=col, lw=1.4, marker='o', ms=3.5, label=nm)
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Training loss'); ax[0].set_title('(a) Training loss (objective-specific)')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Mean AUPR (lesion-valid)'); ax[1].set_title('(b) Validation AUPR'); ax[1].legend(frameon=False, fontsize=6.5)
    plt.tight_layout(); plt.savefig(OUT + 'fig_seg_curves.pdf'); plt.close()

def seg_qual():
    X = np.load(W + 'les_test_X.npy', mmap_mode='r'); Y = np.load(W + 'les_test_Y.npy', mmap_mode='r')
    use = [v for v in ['bce', 'bcedice', 'ftl', 'ftlbg'] if os.path.exists(W + f'probs_test_{v}.npy')]
    Pm = {v: np.load(W + f'probs_test_{v}.npy', mmap_mode='r') for v in use}; R = {v: json.load(open(W + f'res/seg_{v}.json')) for v in use}
    burden = Y.reshape(len(Y), 4, -1).sum(2); sc = (burden > 0).sum(1) * 1e7 + np.minimum(burden, 4e4).sum(1)
    pick = list(np.argsort(-sc)[[2, 12, 25, 40]])
    cols = np.array([[27, 175, 122], [227, 73, 72], [237, 161, 0], [42, 120, 214]], np.uint8)
    nC = 2 + len(use); fig, ax = plt.subplots(len(pick), nC, figsize=(1.35 * nC, 1.35 * len(pick)))
    def over(im, M):
        o = im.copy()
        for c in range(4): o[M[c]] = cols[c]
        return o
    names = {'bce': 'BCE', 'bcedice': 'BCE+Dice', 'ftl': 'Focal Tversky', 'ftlbg': 'Focal Tversky\n+GF'}
    for r, i in enumerate(pick):
        im = np.array(X[i]); gt = np.array(Y[i]).astype(bool)
        m = gt.any(0).astype(np.float32); ii = cv2.integral(m); best = (-1, 0, 0)
        for yy in range(0, 512 - 192, 16):
            for xx in range(0, 512 - 192, 16):
                s = ii[yy + 192, xx + 192] - ii[yy, xx + 192] - ii[yy + 192, xx] + ii[yy, xx]
                if s > best[0]: best = (s, yy, xx)
        _, y0, x0 = best; sl = (slice(y0, y0 + 192), slice(x0, x0 + 192))
        panels = [im[sl], over(im, gt)[sl]] + [over(im, np.stack([np.array(Pm[v][i][c]) >= R[v]['thr'][c] * 255 for c in range(4)]))[sl] for v in use]
        for k, p in enumerate(panels):
            ax[r, k].imshow(p); ax[r, k].axis('off')
            if r == 0: ax[r, k].set_title((['Image', 'Ground truth'] + [names[v] for v in use])[k], fontsize=6.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=LES[k], label=k) for k in CN], loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, -.02), fontsize=7)
    plt.subplots_adjust(wspace=.02, hspace=.02); plt.savefig(OUT + 'fig_seg_qual.pdf'); plt.close()

def froc_pr():
    Yt = np.load(W + 'les_test_Y.npy', mmap_mode='r')
    fig, ax = plt.subplots(2, 4, figsize=(7.4, 3.6))
    for v, nm, col in VAR:
        if not have(v): continue
        R = json.load(open(W + f'res/seg_{v}.json')); P = np.load(W + f'probs_test_{v}.npy', mmap_mode='r')
        for c in range(4):
            p = torch.from_numpy(np.ascontiguousarray(P[:, c])).flatten().long(); y = torch.from_numpy(np.ascontiguousarray(Yt[:, c])).flatten().bool()
            pos = torch.bincount(p[y], minlength=256).double(); neg = torch.bincount(p[~y], minlength=256).double()
            tp = pos.flip(0).cumsum(0); fp = neg.flip(0).cumsum(0); prec = (tp / (tp + fp).clamp(min=1)).numpy(); rec = (tp / pos.sum()).numpy()
            ax[0, c].plot(rec, prec, color=col, lw=1.3, label=nm if c == 0 else None)
            cu = np.array(R['det'][CN[c]]['curve']);
            if len(cu): ax[1, c].plot(cu[:, 0], cu[:, 1], color=col, lw=1.3)
    for c in range(4):
        ax[0, c].set_title(CN[c]); ax[0, c].set_xlabel('Recall'); ax[0, c].set_xlim(0, 1); ax[0, c].set_ylim(0, 1)
        ax[1, c].set_xscale('symlog', linthresh=1); ax[1, c].set_xlim(0, 100); ax[1, c].set_ylim(0, 1); ax[1, c].set_xlabel('False positives per image')
    ax[0, 0].set_ylabel('Precision'); ax[1, 0].set_ylabel('Lesion sensitivity (FROC)')
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.04), fontsize=6.5)
    plt.tight_layout(); plt.savefig(OUT + 'fig_froc_pr.pdf'); plt.close()

def det_qual(v='ftlbg'):
    from metrics_seg import components, dedupe
    X = np.load(W + 'les_test_X.npy', mmap_mode='r'); P = np.load(W + f'probs_test_{v}.npy', mmap_mode='r'); B = np.load(W + 'les_test_B.npy', allow_pickle=True)
    R = json.load(open(W + f'res/seg_{v}.json')); Yt = np.load(W + 'les_test_Y.npy', mmap_mode='r')
    pick = [int(i) for i in np.argsort(-np.array([len(b) for b in B]))[[3, 15, 40]]]
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.9))
    for k, i in enumerate(pick):
        im = np.array(X[i]); gt = dedupe(B[i]); y0, x0 = 128, 128
        for r, cls in enumerate([[1, 2], [0]]):
            a = ax[r, k]; a.imshow(im); a.axis('off')
            for c in cls:
                for b in gt[gt[:, 0] == c]: a.add_patch(plt.Rectangle((b[1], b[2]), b[3] - b[1], b[4] - b[2], fill=False, ec='white', lw=.7))
                pr = components(np.array(P[i][c]), max(R['thr'][c] * 255 * .6, 20)); pr = pr[pr[:, 4] > (0.45 if c != 0 else 0.25)]
                for b in pr: a.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=LES[CN[c]], lw=.9))
            if r == 1:
                m = np.array(Yt[i][0]); ys, xs = np.nonzero(m)
                if len(ys): cy, cx = int(np.median(ys)), int(np.median(xs)); a.set_xlim(cx - 64, cx + 64); a.set_ylim(cy + 64, cy - 64)
            a.set_title(f'Test image #{i}: ' + ('HE + EX' if r == 0 else 'MA (zoom)'), fontsize=7)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc='none', ec='white', label='Ground-truth box'), Patch(fc='none', ec=LES['MA'], label='Predicted MA'), Patch(fc='none', ec=LES['HE'], label='Predicted HE'), Patch(fc='none', ec=LES['EX'], label='Predicted EX')],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, -.02), fontsize=7)
    plt.subplots_adjust(wspace=.03, hspace=.12); plt.savefig(OUT + 'fig_det_qual.pdf'); plt.close()

def grading(seg='ftlbg', bb='convnext_tiny'):
    A = json.load(open(W + f'res/analysis_{seg}_{bb}.json')); R = json.load(open(W + f'res/heads_{seg}_{bb}.json')); main = A['main']
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    cm = np.array(A['cm']); cmn = cm / cm.sum(1, keepdims=True); im = ax[0].imshow(cmn, cmap='Blues', vmin=0, vmax=1); ax[0].grid(False)
    for i in range(5):
        for j in range(5): ax[0].text(j, i, f'{cmn[i, j]:.2f}\n({cm[i, j]})', ha='center', va='center', fontsize=5.5, color='white' if cmn[i, j] > .5 else C['ink'])
    ax[0].set_xticks(range(5)); ax[0].set_yticks(range(5)); ax[0].set_xlabel('Predicted grade'); ax[0].set_ylabel('True grade'); ax[0].set_title('(a) Confusion (row-normalised)')
    for n, col, mk in [(main, C['blue'], 'o'), ('B1 CE (img emb.)', C['orange'], 's'), ('B2 Ordinal (img emb.)', C['aqua'], '^')]:
        if n not in A['reliab']: continue
        r = [(a, b) for a, b, c in A['reliab'][n] if a is not None]; ax[1].plot(*zip(*r), color=col, marker=mk, ms=3.5, lw=1.2, label=n.replace('Ours (full: C1+C2+C5)', 'LesionRule (ours)'))
    ax[1].plot([0, 1], [0, 1], color=C['mute'], lw=.8, ls='--'); ax[1].set_xlabel('Confidence'); ax[1].set_ylabel('Accuracy'); ax[1].set_title('(b) Reliability'); ax[1].legend(frameon=False, fontsize=6)
    for n, col, mk in [(main, C['blue'], 'o'), ('B2 Ordinal (img emb.)', C['aqua'], '^')]:
        if n not in A['conformal']: continue
        cv = A['conformal'][n]; ax[2].plot([1 - c['alpha'] for c in cv], [c['size'] for c in cv], color=col, marker=mk, ms=4, lw=1.2, label=n.replace('Ours (full: C1+C2+C5)', 'LesionRule (ours)'))
        for c in cv: ax[2].annotate(f"{c['coverage']:.3f}", ((1 - c['alpha']), c['size']), fontsize=5.5, xytext=(2, 3), textcoords='offset points', color=col)
    ax[2].set_xlabel('Target coverage $1-\\alpha$'); ax[2].set_ylabel('Mean set size'); ax[2].set_title('(c) Conformal sets (labels: empirical coverage)'); ax[2].legend(frameon=False, fontsize=6)
    plt.tight_layout(); plt.savefig(OUT + 'fig_grading.pdf'); plt.close()

def ablation(seg='ftlbg', bb='convnext_tiny'):
    A = json.load(open(W + f'res/analysis_{seg}_{bb}.json')); names = list(A['agg']); LABS = {'B1 CE (img emb.)': 'B1 embedding, CE', 'B2 Ordinal (img emb.)': 'B2 embedding, ordinal', 'B3 Lesion evidence only': 'B3 evidence only', 'B4 Concat (emb.+evidence)': 'B4 concat.',
            'Ours: tokens, no rule/soft': 'T0 tokens', 'Ours: + soft labels (C5)': 'T1 + soft labels', 'Ours: + rule loss (C2)': 'T2 + rule loss', 'Ours (full: C1+C2+C5)': 'LesionRule (full)', 'Abl: full w/o evidence tokens': 'A1 w/o evidence tokens'}
    lab = [LABS.get(n, n) for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    y = np.arange(len(names))[::-1]
    for a, (k, t) in zip(ax, [('qwk', 'QWK (higher better)'), ('ece', 'ECE (lower better)'), ('viol', 'Rule-violation rate (lower better)')]):
        m = [A['agg'][n][k][0] for n in names]; s = [A['agg'][n][k][1] for n in names]
        a.barh(y, m, xerr=s, color=[C['blue'] if 'full' in n else C['mute'] if n.startswith('B') else C['aqua'] for n in names], height=.65, error_kw=dict(lw=.8, capsize=2))
        a.set_xlabel(t); a.set_yticks(y); a.set_yticklabels(lab)
        if k == 'qwk': a.set_xlim(min(m) - .08, max(m) + .05)
    plt.tight_layout(); plt.savefig(OUT + 'fig_ablation.pdf'); plt.close()

def evidence(seg='ftlbg'):
    _a = sys.argv; sys.argv = ['x', 's', 'b']; import heads as H; sys.argv = _a
    meta = np.load(W + 'g_ddr_meta.npy', allow_pickle=True); sp = np.array([m[0] for m in meta]); y = np.array([m[2] for m in meta])
    P = np.load(W + f'g_ddr_{seg}_maps.npy', mmap_mode='r'); k = np.where((sp == 'test') & (y < 5))[0]
    Z = H.evidence(np.array(P[k])).numpy(); yk = y[k]
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.1)); zn = ['Q1', 'Q2', 'Q3', 'Q4', 'Centre', 'All']
    for c in range(4):
        M = np.stack([Z[yk == g][:, c, :, 1].mean(0) for g in range(5)])      # mean log(1+soft count at tau=.5) per grade and zone
        im = ax[c].imshow(M, cmap='viridis', aspect='auto'); ax[c].grid(False); ax[c].set_xticks(range(6)); ax[c].set_xticklabels(zn, rotation=60, fontsize=6)
        ax[c].set_yticks(range(5)); ax[c].set_yticklabels(range(5) if c == 0 else []); ax[c].set_title(CN[c])
        plt.colorbar(im, ax=ax[c], fraction=.05, pad=.03)
    ax[0].set_ylabel('True grade'); plt.tight_layout(); plt.savefig(OUT + 'fig_evidence.pdf'); plt.close()

def example(seg='ftlbg'):
    """Four test images (one per grade 0,2,3,4): image, lesion maps, zone-wise evidence."""
    meta = np.load(W + 'g_ddr_meta.npy', allow_pickle=True); sp = np.array([m[0] for m in meta]); y = np.array([m[2] for m in meta])
    P = np.load(W + f'g_ddr_{seg}_maps.npy', mmap_mode='r'); X = np.load(W + 'g_ddr_rgb256.npy', mmap_mode='r')
    rng = np.random.default_rng(3); fig, ax = plt.subplots(2, 5, figsize=(7.4, 3.1)); cols = np.array([[27, 175, 122], [227, 73, 72], [237, 161, 0], [42, 120, 214]], np.float32)
    for j, g in enumerate(range(5)):
        cand = np.where((sp == 'test') & (y == g))[0]; i = int(cand[rng.integers(len(cand))])
        im = np.array(X[i]); ax[0, j].imshow(im); ax[0, j].axis('off'); ax[0, j].set_title(f'Grade {g}', fontsize=7)
        p = np.array(P[i]).astype(np.float32) / 255; ov = np.zeros((128, 128, 3), np.float32)
        for c in range(4): ov = np.maximum(ov, p[c][..., None] * cols[c] / 255)
        ax[1, j].imshow(ov); ax[1, j].axis('off')
    ax[0, 0].text(-.05, .5, 'Image (GF)', transform=ax[0, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
    ax[1, 0].text(-.05, .5, 'Lesion maps', transform=ax[1, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=LES[k], label=k) for k in CN], loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, -.02), fontsize=7)
    plt.subplots_adjust(wspace=.03, hspace=.03); plt.savefig(OUT + 'fig_example_maps.pdf'); plt.close()

def failures(seg='ftlbg', bb='convnext_tiny'):
    """Confidently wrong predictions of LesionRule on the leak-controlled test set, with lesion maps."""
    H = json.load(open(W + f'res/heads_{seg}_{bb}.json')); P = np.mean([np.array(r['probs']) for r in H['res']['Ours (full: C1+C2+C5)']], 0); yt = np.array(H['yt'])
    meta = np.load(W + 'g_ddr_meta.npy', allow_pickle=True); sp = np.array([m[0] for m in meta]); y = np.array([m[2] for m in meta]); nm = np.array([m[1].rsplit('.', 1)[0] for m in meta])
    seen = set(np.load(W + 'les_train_names.npy')) | set(np.load(W + 'les_valid_names.npy')); leak = np.array([n in seen for n in nm])
    gi = np.where((sp == 'test') & (y < 5) & ~leak)[0]; assert len(gi) == len(yt) and (y[gi] == yt).all()
    maps = np.load(W + f'g_ddr_{seg}_maps.npy', mmap_mode='r'); X = np.load(W + 'g_ddr_rgb256.npy', mmap_mode='r'); pred = P.argmax(1); conf = P.max(1)
    kinds = [(2, 0, 'Moderate called normal'), (0, 2, 'Normal called moderate'), (3, 2, 'Severe called moderate'), (4, 2, 'PDR called moderate')]
    cols = np.array([[27, 175, 122], [227, 73, 72], [237, 161, 0], [42, 120, 214]], np.float32); rng = np.random.default_rng(5)
    fig, ax = plt.subplots(2, 4, figsize=(7.4, 3.4))
    for j, (t, p, title) in enumerate(kinds):
        idx = np.where((yt == t) & (pred == p))[0]; top = idx[np.argsort(-conf[idx])[:15]]; k = int(top[rng.integers(len(top))]); g = gi[k]
        ax[0, j].imshow(np.array(X[g])); ax[0, j].axis('off'); ax[0, j].set_title(f'{title}\n(true {t}, pred. {p}, conf. {conf[k]:.2f})', fontsize=6.3)
        mp = np.array(maps[g]).astype(np.float32) / 255; ov = np.zeros((128, 128, 3), np.float32)
        for c in range(4): ov = np.maximum(ov, mp[c][..., None] * cols[c] / 255)
        ax[1, j].imshow(ov); ax[1, j].axis('off')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=LES[k], label=k) for k in CN], loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, -.03), fontsize=7)
    plt.subplots_adjust(wspace=.03, hspace=.12); plt.savefig(OUT + 'fig_failures.pdf'); plt.close()

globals()[which](*sys.argv[2:]); print('done', which)
