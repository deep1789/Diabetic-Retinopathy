"""Frozen ImageNet backbone embeddings (multi-scale GAP) from the 256x256 Ben-Graham images. usage: python extract_feats.py <ddr|aptos> [timm_name]"""
import sys, numpy as np, torch, timm
torch.set_num_threads(4)
ds = sys.argv[1]; name = sys.argv[2] if len(sys.argv) > 2 else 'convnext_tiny.fb_in1k'
W = '/home/user/work/'
X = np.load(W + f'g_{ds}_rgb256.npy', mmap_mode='r')
m = timm.create_model(name, pretrained=True, features_only=True, out_indices=(2, 3)).eval().to(memory_format=torch.channels_last)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
out = []
with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
    for i in range(0, len(X), 32):
        x = ((torch.from_numpy(np.array(X[i:i + 32])).permute(0, 3, 1, 2).float() / 255 - mean) / std).contiguous(memory_format=torch.channels_last)
        f = m(x); out.append(torch.cat([t.float().mean((2, 3)) for t in f], 1))
        if i % 1600 == 0: print(i, flush=True)
np.save(W + f'g_{ds}_feat_{name.split(".")[0]}.npy', torch.cat(out).numpy().astype(np.float32)); print('done')
