"""Extract the 757 lesion-annotated DDR images (seg masks + boxes) into a 512x512 cache."""
import sys, io, numpy as np, cv2, xml.etree.ElementTree as ET
from multizip import open_ddr
from common import *
z = open_ddr('/tmp/claude-0/s/ddr'); R = 'DDR-dataset/lesion_segmentation/'
out = {}
for sp in ['train', 'valid', 'test']:
    lab = R + sp + ('/segmentation label/' if sp == 'valid' else '/label/')
    names = sorted(n for n in z.namelist() if n.startswith(R + sp + '/image/') and n.endswith('.jpg'))
    X, Y, B, nm = [], [], [], []
    for n in names:
        stem = n.split('/')[-1][:-4]
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(n), np.uint8), 1), cv2.COLOR_BGR2RGB)
        sq, geo = crop_square(img)
        X.append(to_size(sq, geo[2]))
        ms = []
        for c in CLASSES:
            m = cv2.imdecode(np.frombuffer(z.read(f'{lab}{c}/{stem}.tif'), np.uint8), cv2.IMREAD_UNCHANGED)
            if m.ndim == 3: m = m[..., 0]
            m = (m > 0).astype(np.uint8)
            if m.shape != img.shape[:2]: m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            msq, _ = crop_square(np.where(img.max(2) > 12, 1, 0).astype(np.uint8) * 0 + m) if False else (None, None)
            # apply identical crop geometry as the image
            x0, y0, side = geo; pad = np.zeros((side, side), np.uint8)
            xs0, ys0 = max(x0, 0), max(y0, 0); xs1, ys1 = min(x0 + side, m.shape[1]), min(y0 + side, m.shape[0])
            pad[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = m[ys0:ys1, xs0:xs1]
            ms.append(to_size(pad, side, mask=True))
        Y.append(np.stack(ms))
        t = ET.fromstring(z.read(f'DDR-dataset/lesion_detection/{sp}/{stem}.xml')) if f'DDR-dataset/lesion_detection/{sp}/{stem}.xml' in z.NameToInfo else None
        bl = []
        if t is not None:
            for o in t.findall('object'):
                cl = CLASSES.index(o.find('name').text.strip().upper())
                bb = o.find('bndbox'); b = [float(bb.find(k).text) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                bl.append([cl] + map_box(b, geo))
        B.append(np.array(bl, np.float32).reshape(-1, 5)); nm.append(stem)
    out[sp] = dict(X=np.stack(X), Y=np.stack(Y), B=np.array(B, dtype=object), names=np.array(nm))
    print(sp, len(nm), out[sp]['X'].shape, 'px/class', out[sp]['Y'].sum((0, 2, 3)), 'boxes', sum(len(b) for b in B), flush=True)
for sp, d in out.items():
    np.savez_compressed(f'/home/user/work/les_{sp}.npz', X=d['X'], Y=d['Y'], names=d['names'], B=d['B'], allow_pickle=True) if False else None
    np.save(f'/home/user/work/les_{sp}_X.npy', d['X']); np.save(f'/home/user/work/les_{sp}_Y.npy', d['Y'])
    np.save(f'/home/user/work/les_{sp}_names.npy', d['names']); np.save(f'/home/user/work/les_{sp}_B.npy', d['B'], allow_pickle=True)
