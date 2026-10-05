"""Lesion segmentation training on DDR (383 train / 149 valid) on CPU (bf16 autocast).
usage: python train_seg.py <variant: bce|bcedice|ftl> <seed> [epochs]"""
import sys, time, json, numpy as np, torch, torch.nn.functional as F
from models import UNet
from common import bg_filter
torch.set_num_threads(4)
variant, seed = sys.argv[1], int(sys.argv[2]); EP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
torch.manual_seed(seed); rng = np.random.default_rng(seed)
W = '/home/user/work/'
Xtr, Ytr = np.load(W + 'les_train_X.npy'), np.load(W + 'les_train_Y.npy')
Xva, Yva = np.load(W + 'les_valid_X.npy'), np.load(W + 'les_valid_Y.npy')
if variant.endswith('bg'):
    Xtr = np.stack([bg_filter(x) for x in Xtr]); Xva = np.stack([bg_filter(x) for x in Xva])
CROP, BS, IT = 256, 8, 48
pos_w = torch.tensor([20., 6., 6., 8.]).view(1, 4, 1, 1)
# image indices containing each lesion class (for lesion-centred crop sampling)
has = [np.where(Ytr[:, c].reshape(len(Ytr), -1).any(1))[0] for c in range(4)]

def batch():
    xs, ys = [], []
    for _ in range(BS):
        if rng.random() < 0.7:
            c = rng.integers(4); i = rng.choice(has[c]); yy, xx = np.nonzero(Ytr[i, c])
            k = rng.integers(len(yy)); cy, cx = yy[k], xx[k]
            y0 = int(np.clip(cy - rng.integers(32, CROP - 32), 0, 512 - CROP)); x0 = int(np.clip(cx - rng.integers(32, CROP - 32), 0, 512 - CROP))
        else:
            i = rng.integers(len(Xtr)); y0, x0 = rng.integers(0, 512 - CROP + 1, 2)
        x = Xtr[i, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255; y = Ytr[i, :, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32)
        x = x.transpose(2, 0, 1)
        if rng.random() < .5: x, y = x[:, :, ::-1], y[:, :, ::-1]
        if rng.random() < .5: x, y = x[:, ::-1], y[:, ::-1]
        k = rng.integers(4); x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
        x = np.clip((x - .5) * rng.uniform(.8, 1.25) + .5 + rng.uniform(-.08, .08), 0, 1)
        xs.append(x.copy()); ys.append(y.copy())
    return torch.from_numpy(np.stack(xs)).contiguous(memory_format=torch.channels_last), torch.from_numpy(np.stack(ys))

def ftl(p, y, a=.3, b=.7, g=.75):
    tp = (p * y).sum((0, 2, 3)); fn = ((1 - p) * y).sum((0, 2, 3)); fp = (p * (1 - y)).sum((0, 2, 3))
    t = (tp + 1) / (tp + a * fp + b * fn + 1)          # Tversky index (beta weights false negatives)
    return ((1 - t) ** g).mean()

def dice_loss(p, y):
    i = (p * y).sum((0, 2, 3)); return (1 - (2 * i + 1) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)) + 1)).mean()

def loss_fn(lg, y):
    p = torch.sigmoid(lg)
    if variant == 'bce': return F.binary_cross_entropy_with_logits(lg, y)
    bce = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pos_w)
    return bce + (dice_loss(p, y) if variant == 'bcedice' else 2 * ftl(p, y))

@torch.no_grad()
def predict(m, X, bs=4):
    m.eval(); out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).float().div(255).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cpu', dtype=torch.bfloat16): out.append(torch.sigmoid(m(x).float()))
    return torch.cat(out)

def aupr(P, Y, nb=512):
    """exact AP on probabilities quantised to nb bins; returns per-class AP"""
    res = []
    for c in range(4):
        q = (P[:, c].flatten() * (nb - 1)).round().long(); y = torch.from_numpy(Y[:, c]).flatten().bool()
        pos = torch.bincount(q[y], minlength=nb).double(); neg = torch.bincount(q[~y], minlength=nb).double()
        tp = pos.flip(0).cumsum(0); fp = neg.flip(0).cumsum(0); prec = tp / (tp + fp).clamp(min=1); rec = tp / pos.sum().clamp(min=1)
        ap = (prec[1:] * (rec[1:] - rec[:-1])).sum() + prec[0] * rec[0]; res.append(float(ap))
    return res

m = UNet().to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * IT, pct_start=.15)
best, log = -1, []
for ep in range(EP):
    m.train(); t = time.time(); tl = 0
    for _ in range(IT):
        x, y = batch()
        with torch.autocast('cpu', dtype=torch.bfloat16): lg = m(x)
        l = loss_fn(lg.float(), y); opt.zero_grad(); l.backward(); opt.step(); sched.step(); tl += l.item()
    rec = dict(ep=ep, loss=tl / IT, t=time.time() - t)
    if ep % 5 == 4 or ep == EP - 1:
        ap = aupr(predict(m, Xva), Yva); rec['val_aupr'] = ap; rec['val_mean'] = float(np.mean(ap))
        if rec['val_mean'] > best: best = rec['val_mean']; torch.save(m.state_dict(), W + f'seg_{variant}_s{seed}.pt')
    log.append(rec); print(json.dumps(rec), flush=True)
json.dump(log, open(W + f'seg_{variant}_s{seed}_log.json', 'w'))
