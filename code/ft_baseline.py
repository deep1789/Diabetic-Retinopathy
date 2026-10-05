"""End-to-end fine-tuned ResNet-18 ordinal baseline on Gaussian-filtered 256->224 crops (CPU, bf16). usage: python ft_baseline.py [epochs]"""
import sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0)
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 8; W = '/home/user/work/'
meta = np.load(W + 'g_ddr_meta.npy', allow_pickle=True); sp = np.array([m[0] for m in meta]); y = np.array([m[2] for m in meta]); nm = np.array([m[1].rsplit('.', 1)[0] for m in meta])
X = np.load(W + 'g_ddr_rgb256.npy', mmap_mode='r')
seen = set(np.load(W + 'les_train_names.npy')) | set(np.load(W + 'les_valid_names.npy')); clean = np.array([n not in seen for n in nm])
tr = np.where((sp == 'train') & (y < 5))[0]; va = np.where((sp == 'valid') & (y < 5))[0]; te = np.where((sp == 'test') & (y < 5) & clean)[0]
m = timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=4).to(memory_format=torch.channels_last)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def prep(xb, train):
    x = torch.from_numpy(np.stack(xb)).permute(0, 3, 1, 2).float() / 255
    if train:
        i, j = rng.integers(0, 33, 2); x = x[:, :, i:i + 224, j:j + 224]
        if rng.random() < .5: x = x.flip(3)
        if rng.random() < .5: x = x.flip(2)
    else: x = x[:, :, 16:240, 16:240]
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)
def tgt(yb, s=0.5):
    ks = torch.arange(5).float(); q = torch.softmax(-(ks[None] - yb[:, None].float()) ** 2 / (2 * s ** 2), 1); return q.flip(1).cumsum(1).flip(1)[:, 1:]
@torch.no_grad()
def logits(idx):
    m.eval(); out = []
    for i in range(0, len(idx), 64):
        with torch.autocast('cpu', dtype=torch.bfloat16): out.append(m(prep([X[k] for k in idx[i:i + 64]], False)).float())
    return torch.cat(out)
opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-2); BS = 32; steps = EP * (len(tr) // BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=.15)
best, st = -1, None
for ep in range(EP):
    m.train(); perm = rng.permutation(tr); t = time.time(); tl = 0
    for i in range(0, len(perm) - BS + 1, BS):
        b = np.sort(perm[i:i + BS]); xb = prep([X[k] for k in b], True); yb = torch.from_numpy(y[b]).long()
        with torch.autocast('cpu', dtype=torch.bfloat16): o = m(xb)
        l = F.binary_cross_entropy_with_logits(o.float(), tgt(yb)); opt.zero_grad(); l.backward(); opt.step(); sched.step(); tl += l.item()
    ov = logits(va); pv = (torch.sigmoid(ov) > .5).sum(1).numpy(); q = cohen_kappa_score(y[va], pv, weights='quadratic')
    print(json.dumps(dict(ep=ep, loss=tl / (len(perm) // BS), val_qwk=q, t=time.time() - t)), flush=True)
    if q > best: best = q; st = {k: v.clone() for k, v in m.state_dict().items()}
m.load_state_dict(st); ot = logits(te); ov = logits(va)
def cp(o, T=1.):
    c = torch.cummin(torch.sigmoid(o / T), 1).values; full = torch.cat([torch.ones_like(c[:, :1]), c, torch.zeros_like(c[:, :1])], 1); p = (full[:, :-1] - full[:, 1:]).clamp(min=1e-6); return p / p.sum(1, keepdim=True)
bt = min(np.linspace(.5, 3, 26), key=lambda T: -torch.log(cp(ov, T)[torch.arange(len(va)), torch.from_numpy(y[va])]).mean().item())
pt = cp(ot, bt); pc = pt.argmax(1).numpy(); yt = y[te]
conf, pr = pt.max(1); ece = 0.
for lo in np.linspace(0, 1, 16)[:-1]:
    mk = (conf > lo) & (conf <= lo + 1 / 15)
    if mk.any(): ece += mk.float().mean().item() * abs((pr[mk] == torch.from_numpy(yt)[mk]).float().mean().item() - conf[mk].mean().item())
res = dict(qwk=cohen_kappa_score(yt, pc, weights='quadratic'), acc=float((pc == yt).mean()), f1=f1_score(yt, pc, average='macro'), auc_ref=roc_auc_score(yt >= 2, pt[:, 2:].sum(1).numpy()), ece=ece, n=len(te), params=sum(p.numel() for p in m.parameters()), probs=pt.numpy().round(4).tolist(), T=float(bt))
json.dump(res, open(W + 'res/ft_baseline.json', 'w')); print({k: v for k, v in res.items() if k != 'probs'})
