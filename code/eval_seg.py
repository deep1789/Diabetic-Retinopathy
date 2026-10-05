"""Evaluate a trained segmentation variant on DDR valid/test: AUPR, Dice, IoU, + segmentation-derived detection. usage: python eval_seg.py <variant>"""
import sys, json, numpy as np, torch
from models import UNet
from common import bg_filter
from metrics_seg import *
torch.set_num_threads(4)
v = sys.argv[1]; W = '/home/user/work/'
m = UNet(pretrained=False).to(memory_format=torch.channels_last); m.load_state_dict(torch.load(W + f'seg_{v}_s0.pt')); m.eval()
def predict(X):
    out = []
    for i in range(0, len(X), 4):
        xb = X[i:i + 4]
        if v.endswith('bg'): xb = np.stack([bg_filter(x) for x in xb])
        x = torch.from_numpy(xb).permute(0, 3, 1, 2).float().div(255).contiguous(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16): out.append((torch.sigmoid(m(x).float()) * 255).round().byte().numpy())
    return np.concatenate(out)
Pv = predict(np.load(W + 'les_valid_X.npy')); Yv = np.load(W + 'les_valid_Y.npy')
Pt = predict(np.load(W + 'les_test_X.npy')); Yt = np.load(W + 'les_test_Y.npy')
np.save(W + f'probs_test_{v}.npy', Pt)
res = full_eval(Pv, Yv, Pt, Yt, np.load(W + 'les_test_B.npy', allow_pickle=True)); res['variant'] = v
json.dump(res, open(W + f'res/seg_{v}.json', 'w'), indent=1)
print(v, 'AUPR', [round(a, 3) for a in res['aupr']], 'Dice', [round(a, 3) for a in res['dice']])
