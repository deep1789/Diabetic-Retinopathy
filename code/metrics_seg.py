import numpy as np, torch, cv2

def aupr(P, Y, nb=512):
    """Exact AP of per-pixel probabilities (uint8 or float) against binary masks, quantised to nb bins. P,Y: (N,4,H,W)."""
    res = []
    for c in range(4):
        p = torch.from_numpy(np.ascontiguousarray(P[:, c])); p = p.float() / 255 if p.dtype == torch.uint8 else p.float()
        q = (p.flatten() * (nb - 1)).round().long(); y = torch.from_numpy(np.ascontiguousarray(Y[:, c])).flatten().bool()
        pos = torch.bincount(q[y], minlength=nb).double(); neg = torch.bincount(q[~y], minlength=nb).double()
        tp = pos.flip(0).cumsum(0); fp = neg.flip(0).cumsum(0); prec = tp / (tp + fp).clamp(min=1); rec = tp / pos.sum().clamp(min=1)
        res.append(float((prec[1:] * (rec[1:] - rec[:-1])).sum() + prec[0] * rec[0]))
    return res

def dice_iou(P, Y, thr):
    d, i = [], []
    for c in range(4):
        pb = (np.ascontiguousarray(P[:, c]) >= thr[c]); yb = np.ascontiguousarray(Y[:, c]).astype(bool)
        tp = (pb & yb).sum(); fp = (pb & ~yb).sum(); fn = (~pb & yb).sum()
        d.append(2 * tp / max(2 * tp + fp + fn, 1)); i.append(tp / max(tp + fp + fn, 1))
    return d, i

def best_thr(P, Y, grid=np.linspace(0.05, 0.95, 19)):
    out = []
    for c in range(4):
        pc = np.ascontiguousarray(P[:, c]); yb = np.ascontiguousarray(Y[:, c]).astype(bool); best = (-1, .5)
        for t in grid:
            pb = pc >= int(round(t * 255)); tp = (pb & yb).sum(); d = 2 * tp / max(pb.sum() + yb.sum(), 1)
            if d > best[0]: best = (d, t)
        out.append(int(round(best[1] * 255)))
    return out                                                  # thresholds on the uint8 scale

# ------------------------------------------------------------------ detection
def components(prob_c, thr, min_area=1):
    """Seg-guided proposals (Alg. 5): connected components of the thresholded map -> (x0,y0,x1,y1,score)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((prob_c >= thr).astype(np.uint8), connectivity=8)
    if n <= 1: return np.zeros((0, 5), np.float32)
    cnt = np.bincount(lab.ravel(), minlength=n); sm = np.bincount(lab.ravel(), weights=prob_c.ravel().astype(np.float64), minlength=n)
    keep = np.where(st[1:, 4] >= min_area)[0] + 1
    return np.concatenate([st[keep, :2], st[keep, :2] + st[keep, 2:4], (sm[keep] / cnt[keep] / 255.)[:, None]], 1).astype(np.float32)

def iou_mat(a, b):
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    A = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); B = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (A[:, None] + B[None] - inter + 1e-9)

def dedupe(b):
    return np.unique(np.round(b, 2), axis=0) if len(b) else b

def ap_class(preds, gts, iou_thr):
    """preds: list per image of (n,5); gts: list per image of (m,4). VOC all-point AP."""
    recs = []; ngt = sum(len(g) for g in gts)
    for p, g in zip(preds, gts):
        if len(p) == 0: continue
        order = np.argsort(-p[:, 4]); p = p[order]; used = np.zeros(len(g), bool)
        M = iou_mat(p[:, :4], g) if len(g) else np.zeros((len(p), 0))
        for j in range(len(p)):
            tp = 0
            if M.shape[1]:
                k = np.argmax(np.where(used, -1, M[j]))
                if not used[k] and M[j, k] >= iou_thr: used[k] = True; tp = 1
            recs.append((p[j, 4], tp))
    if not recs: return 0.
    recs = np.array(sorted(recs, key=lambda r: -r[0])); tp = np.cumsum(recs[:, 1]); fp = np.cumsum(1 - recs[:, 1])
    rec = tp / max(ngt, 1); prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]; return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())

def froc(preds, gts, fps_at=(1, 2, 4, 8, 16)):
    """Lesion-level FROC; a prediction is a hit if its centre lies inside a GT box. Returns sensitivity at given FP/image."""
    ev = []; ngt = sum(len(g) for g in gts); nimg = len(gts)
    for p, g in zip(preds, gts):
        if len(p) == 0: continue
        cx = (p[:, 0] + p[:, 2]) / 2; cy = (p[:, 1] + p[:, 3]) / 2
        for j in range(len(p)):
            hit = -1
            if len(g):
                inside = np.where((cx[j] >= g[:, 0]) & (cx[j] <= g[:, 2]) & (cy[j] >= g[:, 1]) & (cy[j] <= g[:, 3]))[0]
                if len(inside): hit = int(inside[0])
            ev.append((p[j, 4], hit, id(g)))
    ev.sort(key=lambda r: -r[0]); found = set(); tps = 0; fp = 0; curve = []
    for sc, hit, gid in ev:
        if hit >= 0:
            if (gid, hit) not in found: found.add((gid, hit)); tps += 1
        else: fp += 1
        curve.append((fp / nimg, tps / max(ngt, 1)))
    c = np.array(curve) if curve else np.zeros((1, 2))
    return {f: float(c[c[:, 0] <= f][:, 1].max()) if (c[:, 0] <= f).any() else 0. for f in fps_at}, c

SEEDS = (12, 25, 50, 75, 128, 180)          # candidate seed thresholds on the uint8 scale (0.05 .. 0.7)

def full_eval(Pv, Yv, Pt, Yt, Bt, Bv):
    """Pixel metrics (AUPR, Dice, IoU at val-optimal thresholds) and segmentation-derived detection metrics on the test split.
    The detection seed threshold of each class is selected on the validation split by AP at IoU 0.3."""
    thr = best_thr(Pv, Yv); d, i = dice_iou(Pt, Yt, thr); ap = aupr(Pt, Yt)
    gts_t = [[dedupe(b[b[:, 0] == c][:, 1:]) for b in Bt] for c in range(4)]; gts_v = [[dedupe(b[b[:, 0] == c][:, 1:]) for b in Bv] for c in range(4)]
    det = {}
    for c, nme in enumerate(['MA', 'HE', 'EX', 'SE']):
        sc = [ap_class([components(Pv[k, c], s) for k in range(len(Pv))], gts_v[c], .3) for s in SEEDS]; seed = SEEDS[int(np.argmax(sc))]
        preds = [components(Pt[k, c], seed) for k in range(len(Pt))]
        fr, curve = froc(preds, gts_t[c])
        det[nme] = dict(seed=seed / 255, val_ap30=float(max(sc)), ap50=ap_class(preds, gts_t[c], .5), ap30=ap_class(preds, gts_t[c], .3), froc=fr, n_pred=int(sum(len(p) for p in preds)),
                        n_gt=int(sum(len(g) for g in gts_t[c])), curve=[[float(a), float(b)] for a, b in curve[::max(1, len(curve) // 200)]])
    return dict(thr=[t / 255 for t in thr], aupr=ap, dice=[float(x) for x in d], iou=[float(x) for x in i], det=det)
