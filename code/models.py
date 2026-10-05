import torch, torch.nn as nn, torch.nn.functional as F, timm

def cbr(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class UNet(nn.Module):
    """U-Net with pretrained timm encoder and a full-resolution shallow branch (keeps micro-aneurysms visible)."""
    def __init__(self, enc='resnet18.a1_in1k', n_out=4, pretrained=True):
        super().__init__()
        self.enc = timm.create_model(enc, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        ch = self.enc.feature_info.channels()            # strides 2,4,8,16,32
        dec = [128, 96, 64, 48, 32]
        self.up = nn.ModuleList(); prev = ch[-1]
        for i in range(4, 0, -1):
            self.up.append(nn.Sequential(cbr(prev + ch[i - 1], dec[i]), cbr(dec[i], dec[i]))); prev = dec[i]
        self.full = nn.Sequential(cbr(3, 16), cbr(16, 16))
        self.fin = nn.Sequential(cbr(prev + 16, 32), nn.Conv2d(32, n_out, 1))
        self.register_buffer('m', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('s', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                                  # x in [0,1]
        x = (x - self.m) / self.s
        f = self.enc(x); y = f[-1]
        for k, i in enumerate(range(4, 0, -1)):
            y = F.interpolate(y, size=f[i - 1].shape[-2:], mode='bilinear', align_corners=False)
            y = self.up[k](torch.cat([y, f[i - 1]], 1))
        y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.fin(torch.cat([y, self.full(x)], 1))
