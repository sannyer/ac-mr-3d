"""Slab forward model: each native pixel is the mean over its 3.5 mm slice thickness (NS samples through the slab)
of the trilinearly interpolated 3D volume, after the stack's rigid transform. fwd: volume -> pixels; adj: transpose."""
import numpy as np
from .geometry import N, HALF, rigid, world_to_vox


def stack_points(st, margin=4.0):
    Z, R, C = st['img'].shape
    jj, ii = np.meshgrid(np.arange(R), np.arange(C), indexing='ij')
    out_P, out_v, out_k = [], [], []
    for k in range(Z):
        P = st['pos'][k] + st['r'] * (ii[..., None] * st['sp']) + st['c'] * (jj[..., None] * st['sp'])
        P = P.reshape(-1, 3); v = st['img'][k].reshape(-1)
        m = np.all(np.abs(P) < HALF + margin, axis=1)
        out_P.append(P[m]); out_v.append(v[m]); out_k.append(np.full(m.sum(), k, np.int16))
    return np.concatenate(out_P), np.concatenate(out_v), np.concatenate(out_k)


def build(st, ns=7, img=None):
    """Sparse operator rows for one stack: idx/w (rows x ns*8), y (pixel values), P (ref-frame centre), k (slice)."""
    st = dict(st)
    if img is not None:
        st['img'] = img
    Rm, t = rigid(st['ser'])
    Q, y, kidx = stack_points(st)
    P0 = (Q - t) @ Rm
    n = st['n'] @ Rm
    offs = (np.arange(ns) + 0.5) / ns * st['th'] - st['th'] / 2
    rows = len(P0)
    idx = np.zeros((rows, ns * 8), np.int32)
    wts = np.zeros((rows, ns * 8), np.float32)
    col = 0
    for s in offs:
        v = world_to_vox(P0 + s * n)
        f = np.floor(v).astype(np.int32); fr = (v - f).astype(np.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix, iy, iz = f[:, 0] + dx, f[:, 1] + dy, f[:, 2] + dz
                    w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dz else 1 - fr[:, 2])
                    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (iz >= 0) & (iz < N)
                    idx[:, col] = np.where(ok, (ix * N + iy) * N + iz, 0)
                    wts[:, col] = np.where(ok, w, 0) / ns
                    col += 1
    keep = wts.sum(1) > 0.6
    if img is not None:
        keep &= ~np.isnan(y)
    return dict(ser=st['ser'], idx=idx[keep], w=wts[keep], y=y[keep].astype(np.float32), P=P0[keep], k=kidx[keep])


def subset(op, m):
    return {kk: (vv[m] if isinstance(vv, np.ndarray) else vv) for kk, vv in op.items()}


class Ops:
    """Operators of several stacks moved to a backend."""
    def __init__(self, B, ops):
        self.B = B
        self.ops = [dict(ser=o['ser'], idx=B.index(o['idx']), w=B.asarray(o['w']), y=B.asarray(o['y'])) for o in ops]
    def fwd(self, o, x):
        return self.B.fwd(x, o['idx'], o['w'])
    def adj(self, o, r):
        return self.B.adj(r, o['idx'], o['w'], N ** 3)
    def AtA(self, x, wst=None):
        out = 0
        for o in self.ops:
            out = out + (wst or {}).get(o['ser'], 1.0) * self.adj(o, self.fwd(o, x))
        return out
    def Aty(self, wst=None):
        out = 0
        for o in self.ops:
            out = out + (wst or {}).get(o['ser'], 1.0) * self.adj(o, o['y'])
        return out
