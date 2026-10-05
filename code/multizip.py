"""Seekable reader over split zip parts (.001..N) so the archive can be read without concatenating."""
import glob, io, zipfile

class MultiPartFile(io.RawIOBase):
    def __init__(self, paths):
        import os
        self.f = [open(p, 'rb') for p in paths]
        self.sz = [os.path.getsize(p) for p in paths]
        self.off = [0]
        for s in self.sz: self.off.append(self.off[-1] + s)
        self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, o, w=0):
        self.pos = o if w == 0 else self.pos + o if w == 1 else self.off[-1] + o
        return self.pos
    def readinto(self, b):
        n = len(b); out = 0
        while out < n and self.pos < self.off[-1]:
            i = max(k for k in range(len(self.sz)) if self.off[k] <= self.pos)
            self.f[i].seek(self.pos - self.off[i])
            chunk = self.f[i].read(min(n - out, self.off[i+1] - self.pos))
            if not chunk: break
            b[out:out+len(chunk)] = chunk; out += len(chunk); self.pos += len(chunk)
        return out

def open_ddr(root):
    parts = sorted(glob.glob(root + '/DDR-dataset.zip.0*'))
    return zipfile.ZipFile(io.BufferedReader(MultiPartFile(parts), 1 << 20))
