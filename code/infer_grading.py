"""One pass over every grading image: crop->512, lesion segmentation (best seg model), save 128x128 lesion maps (uint8) and a 256x256 Ben-Graham RGB.
usage: python infer_grading.py <seg_variant> <dataset: ddr|aptos>"""
import sys, io, re, glob, zipfile, numpy as np, cv2, torch
from concurrent.futures import ThreadPoolExecutor
from models import UNet
from common import crop_square, bg_filter
torch.set_num_threads(4)
variant, ds = sys.argv[1], sys.argv[2]
W = '/home/user/work/'
m = UNet(pretrained=False).to(memory_format=torch.channels_last); m.load_state_dict(torch.load(W + f'seg_{variant}_s0.pt')); m.eval()
use_bg = variant.endswith('bg')

def items():
    if ds == 'ddr':
        from multizip import open_ddr
        z = open_ddr('/tmp/claude-0/s/ddr'); R = 'DDR-dataset/DR_grading/'
        for sp in ['train', 'valid', 'test']:
            for l in z.read(R + sp + '.txt').decode().split('\n'):
                if l.strip():
                    n, y = l.split(); yield sp, n, int(y), (lambda n=n, sp=sp: z.read(R + sp + '/' + n))
    else:
        z = zipfile.ZipFile('/tmp/claude-0/s/k.zip'); lab = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Proliferate_DR': 4}
        for n in sorted(z.namelist()):
            if n.endswith('.png'): yield 'all', n.split('/')[-1][:-4], lab[n.split('/')[-2]], (lambda n=n: z.read(n))

def prep(b):
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)
    if ds == 'ddr': sq, _ = crop_square(img)
    else: sq = img                                    # APTOS-Kaggle images are already cropped, filtered, 224x224
    r512 = cv2.resize(sq, (512, 512), interpolation=cv2.INTER_AREA if sq.shape[0] > 512 else cv2.INTER_CUBIC)
    r256 = cv2.resize(sq, (256, 256), interpolation=cv2.INTER_AREA if sq.shape[0] > 256 else cv2.INTER_CUBIC)
    if ds == 'ddr':
        r256 = bg_filter(r256)
        if use_bg: r512 = bg_filter(r512)
    return r512, r256                                  # APTOS-Kaggle is already Ben-Graham filtered, so it matches the *bg model

meta, maps, rgb = [], [], []
pool = ThreadPoolExecutor(3); BS = 8; buf = []; futs = []
def flush():
    global buf, futs
    if not futs: return
    res = [f.result() for f in futs]
    x = torch.from_numpy(np.stack([r[0] for r in res])).permute(0, 3, 1, 2).float().div(255).contiguous(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        p = torch.sigmoid(m(x).float())
    p = torch.nn.functional.max_pool2d(p, 4)         # 128x128 maps (max-pool keeps micro-aneurysm peaks)
    maps.append((p * 255).round().byte().numpy()); rgb.extend(r[1] for r in res); futs = []
n = 0
for sp, name, y, rd in items():
    futs.append(pool.submit(prep, rd())); meta.append((sp, name, y)); n += 1
    if len(futs) == BS: flush()
    if n % 800 == 0: print(n, flush=True)
flush()
np.save(W + f'g_{ds}_{variant}_maps.npy', np.concatenate(maps)); np.save(W + f'g_{ds}_rgb256.npy', np.stack(rgb))
np.save(W + f'g_{ds}_meta.npy', np.array(meta, dtype=object), allow_pickle=True)
print('done', n)
