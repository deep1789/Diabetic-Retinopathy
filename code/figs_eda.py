import numpy as np, cv2, json, zipfile, collections
from figstyle import *
from multizip import open_ddr
from common import crop_square, bg_filter
rng = np.random.default_rng(7)
z = open_ddr('/tmp/claude-0/s/ddr'); R = 'DDR-dataset/DR_grading/'
rows = {sp: [l.split() for l in z.read(R + sp + '.txt').decode().split('\n') if l.strip()] for sp in ['train', 'valid', 'test']}
# ---------- Fig: samples per grade (raw + Ben-Graham) ----------
fig, ax = plt.subplots(2, 6, figsize=(7.2, 2.6)); stats = {}
for g in range(6):
    cand = [n for n, y in rows['train'] if int(y) == g]
    for _try in range(60):                       # pick a well-framed example (FOV fills the cropped square)
        n = cand[rng.integers(len(cand))]
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(R + 'train/' + n), np.uint8), 1), cv2.COLOR_BGR2RGB)
        sq, _ = crop_square(img)
        if g == 5 or (sq.max(2) > 12).mean() > 0.72: break
    sq = cv2.resize(sq, (320, 320), interpolation=cv2.INTER_AREA)
    for r, im in enumerate([sq, bg_filter(sq)]):
        ax[r, g].imshow(im); ax[r, g].axis('off')
    ax[0, g].set_title(f'Grade {g}: {GRADES[g]}' if g < 5 else 'Ungradable', fontsize=7)
ax[0, 0].text(-0.06, .5, 'Original', transform=ax[0, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
ax[1, 0].text(-0.06, .5, 'Gaussian-filtered', transform=ax[1, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
plt.subplots_adjust(wspace=.03, hspace=.03); plt.savefig(OUT + 'fig_ddr_samples.pdf'); plt.close()
# ---------- Fig: lesion annotations ----------
W = '/home/user/work/'; X = np.load(W + 'les_train_X.npy', mmap_mode='r'); Y = np.load(W + 'les_train_Y.npy', mmap_mode='r'); B = np.load(W + 'les_train_B.npy', allow_pickle=True)
tot = np.array([[Y[i, c].sum() for c in range(4)] for i in range(len(X))])
pick = [int(np.argsort(-(tot > 0).sum(1) * 1e9 - tot.sum(1))[k]) for k in (0, 3, 9, 20)]
pick = [i for i in np.argsort(-((tot > 0).sum(1) * 1e7 + np.minimum(tot, 2e4).sum(1)))[:40]]
pick = [pick[k] for k in (0, 7, 14, 21)]
fig, ax = plt.subplots(3, 4, figsize=(7.2, 5.5))
cols = np.array([[27, 175, 122], [227, 73, 72], [237, 161, 0], [42, 120, 214]], np.uint8)
for j, i in enumerate(pick):
    im = np.array(X[i]); ov = im.copy()
    for c in range(4): ov[Y[i, c] > 0] = cols[c]
    ax[0, j].imshow(im); ax[1, j].imshow(ov)
    # zoom: densest 128x128 window of MA/HE
    m = (Y[i, 0] + Y[i, 1] + Y[i, 2] + Y[i, 3] > 0).astype(np.float32); ii = cv2.integral(m)
    best = (-1, 0, 0)
    for yy in range(0, 512 - 128, 16):
        for xx in range(0, 512 - 128, 16):
            s = ii[yy + 128, xx + 128] - ii[yy, xx + 128] - ii[yy + 128, xx] + ii[yy, xx]
            if s > best[0]: best = (s, yy, xx)
    _, yy, xx = best; z_ = ov[yy:yy + 128, xx:xx + 128].copy(); z_ = cv2.resize(z_, (384, 384), interpolation=cv2.INTER_NEAREST)
    for b in B[i]:
        x0, y0, x1, y1 = [(v - o) * 3 for v, o in zip(b[1:], [xx, yy, xx, yy])]
        if x1 > 0 and y1 > 0 and x0 < 384 and y0 < 384: cv2.rectangle(z_, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 1)
    ax[2, j].imshow(z_)
    for r in range(3): ax[r, j].axis('off')
    ax[0, j].set_title(f'DDR train #{i}', fontsize=7)
ax[0, 0].text(-.04, .5, 'Fundus (512²)', transform=ax[0, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
ax[1, 0].text(-.04, .5, 'Pixel masks', transform=ax[1, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
ax[2, 0].text(-.04, .5, 'Zoom + boxes', transform=ax[2, 0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=LES[k], label=n) for k, n in zip(['MA', 'HE', 'EX', 'SE'], ['Microaneurysm', 'Haemorrhage', 'Hard exudate', 'Soft exudate'])] +
           [Patch(facecolor='none', edgecolor='k', label='Bounding box')], loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.01))
plt.subplots_adjust(wspace=.03, hspace=.03); plt.savefig(OUT + 'fig_ddr_lesions.pdf'); plt.close()
# ---------- Fig: dataset statistics ----------
cnt = {sp: collections.Counter(int(y) for _, y in v) for sp, v in rows.items()}
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.3))
w = .27
for k, (sp, col) in enumerate(zip(['train', 'valid', 'test'], [C['blue'], C['orange'], C['aqua']])):
    ax[0].bar(np.arange(6) + (k - 1) * w, [cnt[sp][g] for g in range(6)], w * .92, color=col, label=sp)
ax[0].set_xticks(range(6)); ax[0].set_xticklabels(['0', '1', '2', '3', '4', 'UG']); ax[0].set_xlabel('Grade (UG = ungradable)'); ax[0].set_ylabel('Images'); ax[0].legend(frameon=False)
ax[0].set_title('(a) Grade distribution')
def _dd(b): return np.unique(np.round(b, 2), axis=0) if len(b) else b          # raw XML lists ~41% of boxes twice
allB = {sp: [_dd(b) for b in np.load(W + f'les_{sp}_B.npy', allow_pickle=True)] for sp in ['train', 'valid', 'test']}
Ypx = {sp: np.load(W + f'les_{sp}_Y.npy', mmap_mode='r') for sp in ['train', 'valid', 'test']}
names = ['MA', 'HE', 'EX', 'SE']
inst = np.array([[sum(int((b[:, 0] == c).sum()) for b in allB[sp]) for c in range(4)] for sp in allB])
imgs_with = np.array([[int(sum((Ypx[sp][i, c].any()) for i in range(len(Ypx[sp])))) for c in range(4)] for sp in allB])
for k, (sp, col) in enumerate(zip(allB, [C['blue'], C['orange'], C['aqua']])):
    ax[1].bar(np.arange(4) + (k - 1) * w, inst[k], w * .92, color=col)
ax[1].set_yscale('log'); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names); ax[1].set_ylabel('Annotated boxes (log)'); ax[1].set_title('(b) Lesion instances')
sizes = {c: [] for c in range(4)}
for b in allB['train']:
    for r in b: sizes[int(r[0])].append(max(r[3] - r[1], r[4] - r[2]) * 1956 / 512 / 1.0)  # approx native px (DDR images ~1.9k px)
for c in range(4):
    v = np.array(sizes[c]); ax[2].hist(v, bins=np.logspace(0.5, 3, 30), histtype='step', color=LES[names[c]], lw=1.4, label=names[c])
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('Longer box side (≈ native px)'); ax[2].set_ylabel('Count'); ax[2].legend(frameon=False); ax[2].set_title('(c) Lesion size')
per = [np.array([(sum(1 for r in b if int(r[0]) == c)) for b in allB['train']]) for c in range(4)]
ax[3].boxplot([np.log10(p + 1) for p in per], tick_labels=names, showfliers=False, medianprops=dict(color=C['ink']), boxprops=dict(color=C['mute']), whiskerprops=dict(color=C['mute']), capprops=dict(color=C['mute']))
ax[3].set_ylabel('log10(boxes per image + 1)'); ax[3].set_title('(d) Burden per image')
plt.tight_layout(w_pad=.8); plt.savefig(OUT + 'fig_dataset_stats.pdf'); plt.close()
json.dump(dict(grade_counts={sp: {str(g): cnt[sp][g] for g in range(6)} for sp in cnt}, boxes={sp: inst[k].tolist() for k, sp in enumerate(allB)},
               imgs_with_lesion={sp: imgs_with[k].tolist() for k, sp in enumerate(allB)},
               median_size={n: float(np.median(sizes[c])) for c, n in enumerate(names)}, p10_size={n: float(np.percentile(sizes[c], 10)) for c, n in enumerate(names)}),
          open('/home/user/Diabetic-Retinopathy/paper/stats_ddr.json', 'w'), indent=1)
# ---------- Fig: Kaggle samples ----------
k = zipfile.ZipFile('/tmp/claude-0/s/k.zip'); cl = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
fig, ax = plt.subplots(2, 5, figsize=(7.2, 3.0))
for j, c in enumerate(cl):
    fs = sorted(n for n in k.namelist() if f'/{c}/' in n and n.endswith('.png'))
    for r in range(2):
        im = cv2.cvtColor(cv2.imdecode(np.frombuffer(k.read(fs[rng.integers(len(fs))]), np.uint8), 1), cv2.COLOR_BGR2RGB)
        ax[r, j].imshow(im); ax[r, j].axis('off')
    ax[0, j].set_title(f'{GRADES[j]} (n={len(fs)})', fontsize=7)
plt.subplots_adjust(wspace=.03, hspace=.03); plt.savefig(OUT + 'fig_aptos_samples.pdf'); plt.close()
print('ok')
