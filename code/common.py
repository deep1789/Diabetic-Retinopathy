import numpy as np, cv2
CLASSES = ['MA', 'HE', 'EX', 'SE']
S = 512

def crop_square(img, thr=12):
    """Crop black borders of a fundus photo, pad to square. Returns img, (x0,y0,side,padx,pady)."""
    g = img.max(2) if img.ndim == 3 else img
    ys, xs = np.where(g > thr)
    if len(xs) < 100: y0, y1, x0, x1 = 0, img.shape[0], 0, img.shape[1]
    else: y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = img[y0:y1, x0:x1]
    h, w = c.shape[:2]; side = max(h, w)
    ph, pw = (side - h) // 2, (side - w) // 2
    out = np.zeros((side, side) + img.shape[2:], img.dtype)
    out[ph:ph + h, pw:pw + w] = c
    return out, (x0 - pw, y0 - ph, side)

def to_size(img, side, size=S, mask=False):
    if mask:
        m = cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        return (m > 0.15).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def map_box(b, geo, size=S):
    x0, y0, side = geo; s = size / side
    return [(b[0] - x0) * s, (b[1] - y0) * s, (b[2] - x0) * s, (b[3] - y0) * s]

def bg_filter(img, sigma_frac=1 / 22.4):
    """Ben-Graham style Gaussian filtering used by the Kaggle 'gaussian_filtered' set: 4*I - 4*G*I + 128, outside FOV = 128."""
    s = img.shape[0] * sigma_frac
    blur = cv2.GaussianBlur(img, (0, 0), s)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    fov = cv2.erode((img.max(2) > 12).astype(np.uint8), np.ones((7, 7), np.uint8))
    out[fov == 0] = 128
    return out
