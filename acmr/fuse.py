"""Fuse per-slice 2D bone contours from several series into one 3D signed-distance field (SDF) per bone.

Each contour becomes a truncated 2D SDF (negative inside, optionally Gaussian-smoothed in-plane); it enters through the
same slab forward model as the images. The 3D field minimises  sum ||A phi - sdf||^2 + lam ||Laplacian phi||^2 + mu ||phi - T||^2,
started from shape-based interpolation along each stack's normal and solved by CG on the selected backend."""
import time
import numpy as np
from scipy import ndimage as ndi
from . import backend, slab, reg
from .geometry import N, grid_points, stack_coords

T = 5.0
MU = 2e-4


def sdf2d(m, sp):
    if not m.any():
        return np.full(m.shape, T, np.float32)
    d = ndi.distance_transform_edt(~m) * sp - ndi.distance_transform_edt(m) * sp
    return np.clip(d, -T, T).astype(np.float32)


def fuse_bone(stacks, seg, info, bone, lam=1.0, iters=160, smooth=0.0, min_iou=0.55, outside_neighbours=True,
              hold=(), B=None, log=print):
    B = B or backend.get()
    GP = grid_points()
    ops, inits, used = [], [], {}
    for ser, st in stacks.items():
        if ser in hold or ser not in seg:
            continue
        ks = [k for k in seg[ser] if bone in seg[ser][k] and info[ser][k][bone]['iou'] >= min_iou]
        if not ks:
            continue
        allk = [k for k in seg[ser] if bone in seg[ser][k]]
        img = np.full(st['img'].shape, np.nan, np.float32)
        for k in ks:
            q = sdf2d(seg[ser][k][bone], st['sp'])
            img[k] = ndi.gaussian_filter(q, smooth / st['sp']) if smooth > 0 else q
        if outside_neighbours:
            for k in {min(allk) - 1, max(allk) + 1} - set(allk):
                if 0 <= k < len(img):
                    img[k] = T
        kk, jj, ii = stack_coords(st, GP)
        inits.append(ndi.map_coordinates(img, [kk, jj, ii], order=1, mode='constant', cval=np.nan))
        ops.append(slab.build(st, ns=3, img=img))
        used[ser] = sorted(ks)
    if not ops:
        return None, {}
    with np.errstate(all='ignore'):
        x0 = np.nanmean(np.stack(inits), 0)
    x0 = ndi.gaussian_filter(np.where(np.isnan(x0), T, x0).astype(np.float32).reshape(N, N, N), 1).ravel()
    O = slab.Ops(B, ops)
    b = O.Aty() + MU * T
    A = lambda v: O.AtA(v) + lam * reg.lap(B, reg.lap(B, v.reshape(N, N, N))).reshape(-1) + MU * v
    t0 = time.time()
    x = backend.cg(B, A, b, B.asarray(x0), iters)
    rms = {o['ser']: round(float(np.sqrt(((B.numpy(O.fwd(o, x)) - B.numpy(o['y'])) ** 2).mean())), 3) for o in O.ops}
    phi = B.numpy(x).reshape(N, N, N).astype(np.float32)
    log(f'  {bone}: slices {used}  rms misfit (mm) {rms}  volume {(phi < 0).sum() * 0.125 / 1000:.2f} ml  ({time.time() - t0:.0f} s, {B.name})')
    return phi, rms


def labels_from_phis(phis, vals=None):
    from .seg2d import VAL
    vals = vals or VAL
    lab = np.zeros((N, N, N), np.uint8); best = np.full((N, N, N), np.inf, np.float32)
    for b, phi in phis.items():
        m = (phi < 0) & (phi < best)
        lab[m] = vals[b]; best[m] = phi[m]
    return lab
