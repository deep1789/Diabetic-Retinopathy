"""Grading heads on frozen backbone embeddings + lesion-evidence from the segmentation network.
usage: python heads.py <seg_variant> <backbone> [seeds]
Writes /home/user/work/res/heads_<seg_variant>_<backbone>.json and per-run test probabilities."""
import sys, json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
torch.set_num_threads(4)
W = '/home/user/work/'; os.makedirs(W + 'res', exist_ok=True)
SEG, BB = sys.argv[1], sys.argv[2]; SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5

# ---------------------------------------------------------------- lesion evidence (C1)
def zone_masks(n=128):
    yy, xx = np.mgrid[:n, :n]; c = (n - 1) / 2; r = np.hypot(yy - c, xx - c); fov = r <= n / 2 - 2
    q = [fov & (yy < c) & (xx < c), fov & (yy < c) & (xx >= c), fov & (yy >= c) & (xx < c), fov & (yy >= c) & (xx >= c)]
    return torch.tensor(np.stack(q + [fov & (r < 0.35 * n / 2), fov]).astype(np.float32))      # 4 quadrants, centre, whole  -> (6,n,n)

TAUS = (0.25, 0.5, 0.75)
def evidence(P, bs=512):
    """P (N,4,128,128) in [0,1] -> Z (N,4,6,5): 3 soft counts at thresholds, #local peaks, max prob, per class and zone."""
    M = zone_masks(); out = []
    for i in range(0, len(P), bs):
        p = torch.from_numpy(P[i:i + bs]).float() / 255
        cnt = torch.stack([torch.sigmoid((p - t) / 0.05) for t in TAUS], 2)                  # (b,4,3,H,W)
        pk = ((p == F.max_pool2d(p, 3, 1, 1)) & (p > 0.5)).float()                           # peaks
        feats = torch.einsum('bcthw,zhw->bczt', cnt, M)                                     # (b,4,6,3)
        pks = torch.einsum('bchw,zhw->bcz', pk, M).unsqueeze(-1)
        mx = (p.unsqueeze(2) * M.view(1, 1, 6, *M.shape[-2:])).amax((3, 4)).unsqueeze(-1)
        out.append(torch.log1p(torch.cat([feats, pks, mx * 10], -1)))
    return torch.cat(out)

# ---------------------------------------------------------------- rule templates (C2)
class Rules(nn.Module):
    """Soft-logic necessary conditions R_k(Z) for 'grade >= k', learnable thresholds (product t-norm)."""
    def __init__(s):
        super().__init__()
        s.t = nn.Parameter(torch.tensor([1.0, 1.4, 1.4, 1.4, 2.0, 1.4, 1.4, 1.4, 1.4, 2.0]))
        s.s = 0.35
    def forward(s, Z):                                  # Z (N,4,6,5); stat 1 = soft count at tau=.5 ; zones 0-3 quadrants, 5 whole
        u = Z[..., 1]; a = lambda x, i: torch.sigmoid((x - s.t[i]) / s.s)
        ma, he, ex, se = u[:, 0, 5], u[:, 1, 5], u[:, 2, 5], u[:, 3, 5]
        any_ = 1 - (1 - a(ma, 0)) * (1 - a(he, 1)) * (1 - a(ex, 2)) * (1 - a(se, 3))
        nonma = 1 - (1 - a(ma, 4)) * (1 - a(he, 1)) * (1 - a(ex, 2)) * (1 - a(se, 3))
        q4 = torch.stack([a(u[:, 1, q], 5 + q % 4) for q in range(4)], 1).prod(1)          # HE in all 4 quadrants ("4" of 4-2-1)
        sev = 1 - (1 - q4) * (1 - a(se, 9))
        return torch.stack([any_, nonma, sev, nonma], 1)                                   # k = 1..4

def fit_rules(Z, y, steps=400):
    R = Rules(); opt = torch.optim.Adam(R.parameters(), 0.05); ks = torch.arange(1, 5)
    for _ in range(steps):
        r = R(Z); pos = (y[:, None] >= ks).float()
        loss = ((1 - r) * pos).sum(0) / pos.sum(0).clamp(min=1) + 0.5 * (r * (1 - pos)).sum(0) / (1 - pos).sum(0).clamp(min=1)
        opt.zero_grad(); loss.sum().backward(); opt.step()
    for p in R.parameters(): p.requires_grad_(False)
    return R

# ---------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(s, din, nout=4, h=256, p=0.3):
        super().__init__(); s.f = nn.Sequential(nn.Dropout(p), nn.Linear(din, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, nout))
    def forward(s, E, Z): return s.f(E)
class LesionMLP(MLP):
    def __init__(s, nout=4): super().__init__(120, nout, 128, .1)
    def forward(s, E, Z): return s.f(Z.flatten(1))
class ConcatMLP(MLP):
    def __init__(s, din, nout=4): super().__init__(din + 120, nout)
    def forward(s, E, Z): return s.f(torch.cat([E, Z.flatten(1)], 1))
class EvidenceTransformer(nn.Module):
    """Grade query attends to image token + 24 lesion-evidence tokens (class x zone)  (C1)."""
    def __init__(s, din, nout=4, d=64, use_ev=True, use_img=True, L=2):
        super().__init__(); s.use_ev, s.use_img = use_ev, use_img
        s.img = nn.Sequential(nn.Dropout(.3), nn.Linear(din, d)); s.ev = nn.Linear(5, d)
        s.ce = nn.Parameter(torch.randn(4, 1, d) * .02); s.ze = nn.Parameter(torch.randn(1, 6, d) * .02); s.cls = nn.Parameter(torch.randn(1, 1, d) * .02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True), L)
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, nout))
    def forward(s, E, Z):
        toks = [s.cls.expand(len(E), -1, -1)]
        if s.use_img: toks.append(s.img(E).unsqueeze(1))
        if s.use_ev: toks.append((s.ev(Z) + s.ce.unsqueeze(0) + s.ze.unsqueeze(0)).flatten(1, 2))
        return s.out(s.enc(torch.cat(toks, 1))[:, 0])

# ---------------------------------------------------------------- losses / decoding
def soft_cum_targets(y, sigma):
    ks = torch.arange(5).float()
    if sigma <= 0: q = F.one_hot(y, 5).float()
    else: q = torch.softmax(-(ks[None] - y[:, None].float()) ** 2 / (2 * sigma ** 2), 1)
    return q.flip(1).cumsum(1).flip(1)[:, 1:]                                              # P(y>=k), k=1..4

def probs_from_cum(c):                                                                      # c = P(y>=k) (N,4) -> class probs (N,5)
    c = torch.cummin(c, 1).values; full = torch.cat([torch.ones_like(c[:, :1]), c, torch.zeros_like(c[:, :1])], 1)
    p = (full[:, :-1] - full[:, 1:]).clamp(min=1e-6); return p / p.sum(1, keepdim=True)

def make(cfg, din):
    k = cfg['arch']
    if k == 'mlp': return MLP(din, 5 if cfg['loss'] == 'ce' else 4)
    if k == 'lesion': return LesionMLP()
    if k == 'concat': return ConcatMLP(din)
    if k == 'et': return EvidenceTransformer(din, use_ev=cfg.get('ev', True))

def run(cfg, seed, D):
    torch.manual_seed(seed); np.random.seed(seed)
    E, Z, y = D['tr']; Ev, Zv, yv = D['va']; Et, Zt, yt = D['te']; R = D['R']
    m = make(cfg, E.shape[1]); lr = 5e-4 if cfg['arch'] == 'et' else 1e-3
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-2); EP = 60
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    sigma, lam = cfg.get('sigma', 0.0), cfg.get('lam', 0.0); best, state = -1, None
    def predict(E_, Z_):
        m.eval()
        with torch.no_grad(): o = m(E_, Z_)
        return o
    for ep in range(EP):
        m.train(); perm = torch.randperm(len(E))
        for i in range(0, len(E), 128):
            b = perm[i:i + 128]; o = m(E[b], Z[b])
            if cfg['loss'] == 'ce': loss = F.cross_entropy(o, y[b])
            else:
                tgt = soft_cum_targets(y[b], sigma); loss = F.binary_cross_entropy_with_logits(o, tgt)
                if lam > 0: loss = loss + lam * (F.relu(torch.sigmoid(o) - R(Z[b])) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 3 == 2 or ep == EP - 1:
            ov = predict(Ev, Zv); pv = pred_class(ov, cfg); q = cohen_kappa_score(yv.numpy(), pv.numpy(), weights='quadratic')
            if q > best: best = q; state = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    return m, predict(Ev, Zv), predict(Et, Zt), predict(*D['vb'][:2])

def pred_class(o, cfg):
    return o.argmax(1) if cfg['loss'] == 'ce' else (torch.sigmoid(o) > 0.5).sum(1)

def class_probs(o, cfg, T=1.0):
    return torch.softmax(o / T, 1) if cfg['loss'] == 'ce' else probs_from_cum(torch.sigmoid(o / T))

def fit_T(o, y, cfg):
    best, bt = 1e9, 1.0
    for T in np.linspace(0.5, 3, 26):
        p = class_probs(o, cfg, T); nll = -torch.log(p[torch.arange(len(y)), y]).mean().item()
        if nll < best: best, bt = nll, T
    return bt

def ece(p, y, nb=15):
    conf, pred = p.max(1); acc = (pred == y).float(); e = 0.
    for lo in np.linspace(0, 1, nb + 1)[:-1]:
        mk = (conf > lo) & (conf <= lo + 1 / nb)
        if mk.any(): e += mk.float().mean().item() * abs(acc[mk].mean().item() - conf[mk].mean().item())
    return e

if __name__ == '__main__':
    meta = np.load(W + 'g_ddr_meta.npy', allow_pickle=True); sp = np.array([m[0] for m in meta]); yy = np.array([m[2] for m in meta])
    P = np.load(W + f'g_ddr_{SEG}_maps.npy'); E = np.load(W + f'g_ddr_feat_{BB}.npy')
    Z = evidence(P); good = yy < 5
    mu, sd = E[sp == 'train'].mean(0), E[sp == 'train'].std(0) + 1e-6; E = torch.from_numpy((E - mu) / sd).float()
    seen = set(np.load(W + 'les_train_names.npy')) | set(np.load(W + 'les_valid_names.npy'))
    nm = np.array([m[1].rsplit('.', 1)[0] for m in meta]); leak = np.array([n in seen for n in nm])
    def sub(s): k = (sp == s) & good; return E[k], Z[k], torch.from_numpy(yy[k]).long()
    Ev_, Zv_, yv_ = sub('valid'); pm = torch.from_numpy(np.random.default_rng(0).permutation(len(Ev_))); h = len(pm) // 2
    # validation split: half A = model selection + temperature scaling, half B = split-conformal calibration (keeps exchangeability)
    D = dict(tr=sub('train'), va=(Ev_[pm[:h]], Zv_[pm[:h]], yv_[pm[:h]]), vb=(Ev_[pm[h:]], Zv_[pm[h:]], yv_[pm[h:]]), te=sub('test'), clean=torch.from_numpy(~leak[(sp == 'test') & good]))
    def suba(s): k = (sp == s); return E[k], Z[k], torch.from_numpy(yy[k]).long()
    Atr, Ava, Ate = suba('train'), suba('valid'), suba('test'); clean_all = torch.from_numpy(~leak[sp == 'test'])
    def train_gate(seed):
        torch.manual_seed(seed); g = ConcatMLP(E.shape[1], 1); opt = torch.optim.AdamW(g.parameters(), 1e-3, weight_decay=1e-2); yb = (Atr[2] == 5).float()
        for ep in range(30):
            g.train(); perm = torch.randperm(len(yb))
            for i in range(0, len(yb), 128):
                bi = perm[i:i + 128]; l = F.binary_cross_entropy_with_logits(g(Atr[0][bi], Atr[1][bi]).squeeze(1), yb[bi]); opt.zero_grad(); l.backward(); opt.step()
        g.eval()
        with torch.no_grad(): pv = torch.sigmoid(g(Ava[0], Ava[1]).squeeze(1)); pt = torch.sigmoid(g(Ate[0], Ate[1]).squeeze(1))
        yv = (Ava[2] == 5).numpy(); best = (-1, .5)
        for t in np.linspace(.05, .95, 19):
            f = f1_score(yv, (pv.numpy() > t), zero_division=0)
            if f > best[0]: best = (f, t)
        return pt.numpy(), best[1], roc_auc_score((Ate[2] == 5).numpy()[clean_all.numpy()], pt.numpy()[clean_all.numpy()])
    GATES = [train_gate(sd) for sd in range(SEEDS)]
    EXT = None   # zero-shot external set (APTOS-derived Kaggle), standardised with DDR training statistics
    if os.path.exists(W + f'g_aptos_{SEG}_maps.npy') and os.path.exists(W + f'g_aptos_feat_{BB}.npy'):
        Pa = np.load(W + f'g_aptos_{SEG}_maps.npy'); Ea = np.load(W + f'g_aptos_feat_{BB}.npy'); ya = np.array([m[2] for m in np.load(W + 'g_aptos_meta.npy', allow_pickle=True)])
        EXT = (torch.from_numpy((Ea - mu) / sd).float(), evidence(Pa), torch.from_numpy(ya).long())
    D['R'] = fit_rules(D['tr'][1], D['tr'][2]); torch.save(D['R'].state_dict(), W + f'res/rules_{SEG}.pt')
    CFG = {
        'B1 CE (img emb.)':            dict(arch='mlp', loss='ce'),
        'B2 Ordinal (img emb.)':       dict(arch='mlp', loss='ord'),
        'B3 Lesion evidence only':     dict(arch='lesion', loss='ord'),
        'B4 Concat (emb.+evidence)':   dict(arch='concat', loss='ord'),
        'Ours: tokens, no rule/soft':  dict(arch='et', loss='ord'),
        'Ours: + soft labels (C5)':    dict(arch='et', loss='ord', sigma=0.5),
        'Ours: + rule loss (C2)':      dict(arch='et', loss='ord', lam=1.0),
        'Ours (full: C1+C2+C5)':       dict(arch='et', loss='ord', sigma=0.5, lam=1.0),
        'Abl: full w/o evidence tokens': dict(arch='et', loss='ord', sigma=0.5, lam=1.0, ev=False),
    }
    res = {}
    for name, cfg in CFG.items():
        runs = []
        for s in range(SEEDS):
            m, ov, ot, ob = run(cfg, s, D); T = fit_T(ov, D['va'][2], cfg)
            cl = D['clean']; yt_full = D['te'][2].numpy(); pt_full = class_probs(ot, cfg, T); pc_full = pt_full.argmax(1).numpy()
            qwk_full = cohen_kappa_score(yt_full, pc_full, weights='quadratic')
            yt = yt_full[cl.numpy()]; pt = pt_full[cl]; pc = pc_full[cl.numpy()]; Rt = D['R'](D['te'][1][cl]).numpy()
            viol = float(np.mean([any(pc[i] >= k and Rt[i, k - 1] < 0.5 for k in range(1, 5)) for i in range(len(pc))]))
            ref = (pt[:, 2:].sum(1)).numpy()
            m.eval()
            with torch.no_grad(): pa = pred_class(m(Ate[0], Ate[1]), cfg).numpy()
            ext = {}
            if EXT is not None:
                with torch.no_grad(): oe = m(EXT[0], EXT[1])
                pe = class_probs(oe, cfg, T); ce = pe.argmax(1).numpy(); ye = EXT[2].numpy(); Re = D['R'](EXT[1]).numpy()
                ext = dict(ex_qwk=cohen_kappa_score(ye, ce, weights='quadratic'), ex_acc=float((ce == ye).mean()), ex_f1=f1_score(ye, ce, average='macro'),
                           ex_auc=roc_auc_score(ye >= 2, pe[:, 2:].sum(1).numpy()), ex_viol=float(np.mean([any(ce[i] >= k and Re[i, k - 1] < 0.5 for k in range(1, 5)) for i in range(len(ce))])),
                           ex_probs=pe.numpy().round(3).tolist() if s == 0 else None)
            gp, gt, gauc = GATES[s]; p6 = np.where(gp > gt, 5, pa); ya = Ate[2].numpy(); cm = clean_all.numpy()
            runs.append(dict(acc6=float((p6 == ya)[cm].mean()), f1_6=f1_score(ya[cm], p6[cm], average='macro'), gate_auc=float(gauc), **ext))
            runs[-1].update(dict(qwk=cohen_kappa_score(yt, pc, weights='quadratic'), qwk_full=qwk_full, acc=float((pc == yt).mean()), f1=f1_score(yt, pc, average='macro'),
                             auc_ref=roc_auc_score(yt >= 2, ref), ece=ece(pt, torch.from_numpy(yt)), viol=viol,
                             probs=pt.numpy().round(4).tolist(), pb=class_probs(ob, cfg, T).numpy().round(4).tolist(), T=float(T)))
            print(name, s, {k: round(v, 4) for k, v in runs[-1].items() if k not in ("probs", "pb", "ex_probs")}, flush=True)
        res[name] = runs
    json.dump(dict(res=res, yt=D['te'][2][D['clean']].tolist(), yb=D['vb'][2].tolist()), open(W + f'res/heads_{SEG}_{BB}.json', 'w'))
