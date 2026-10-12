"""2D bone contours on native slices (random walker at native in-plane resolution).

initial(): from hand-read seeds on 601 + 801 (no prior model needed).
refine():  on every slice of every series, seeded by a 3D label prior projected into the slice
           (prior eroded = bone marker, outside the dilated prior = background marker)."""
import os
import numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.segmentation import random_walker
from .geometry import world_to_vox, slice_world
from . import seeds as SD

BONES = ['clav', 'acro', 'cora', 'hum']
VAL = dict(clav=2, acro=3, cora=4, hum=5)
NPROC = int(os.environ.get('ACMR_PROCS', os.cpu_count() or 2))


def disk(r):
    r = max(1, int(round(r))); y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


# ---------- initial contours from hand seeds ----------
HALFC = 30.0


def _crop_corner(st, k):
    d = -st['pos'][k]
    return d @ st['r'] / st['sp'] - HALFC / st['sp'], d @ st['c'] / st['sp'] - HALFC / st['sp']


def _local_to_pix(st, k, u, v):
    i0, j0 = _crop_corner(st, k)
    return i0 + u / st['sp'], j0 + v / st['sp']


def _body_depth(sm, sp):
    hi = np.percentile(sm, 99)
    body = ndi.binary_fill_holes(ndi.binary_closing(sm > 0.12 * hi, iterations=4))
    l, n = ndi.label(body)
    if n:
        body = l == (np.argmax(ndi.sum(body, l, range(1, n + 1))) + 1)
    return ndi.distance_transform_edt(body) * sp


def _rw_seeded(sm, seeds, sp, rel, R, depth, skin, open_mm, cortex_mm, beta=2500, others=()):
    H, Wd = sm.shape
    pts = np.asarray(seeds, float)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    pad = R / sp + 4
    i0, i1 = int(max(0, x0 - pad)), int(min(Wd, x1 + pad)); j0, j1 = int(max(0, y0 - pad)), int(min(H, y1 + pad))
    a = sm[j0:j1, i0:i1]; dp = depth[j0:j1, i0:i1]
    yy, xx = np.mgrid[j0:j1, i0:i1]
    d = np.min([np.hypot(xx - x, yy - y) for x, y in pts], 0) * sp
    an = np.clip(a / np.percentile(a[d < R], 99), 0, 1.2)
    mk = np.zeros(a.shape, np.int32)
    mk[(d > R) | (dp < skin)] = 2
    for x, y in others:
        mk[np.hypot(xx - x, yy - y) * sp < 2.0] = 2
    mk[d < 1.2] = 1
    m = random_walker(an, mk, beta=beta, mode='cg_j', tol=1e-3) == 1
    m = ndi.binary_opening(ndi.binary_fill_holes(m), iterations=max(1, int(round(open_mm / sp))))
    l, n = ndi.label(m)
    if n > 1:
        ids = {l[int(round(y)) - j0, int(round(x)) - i0] for x, y in pts} - {0}
        m = np.isin(l, list(ids)) if ids else l == (np.argmax(ndi.sum(m, l, range(1, n + 1))) + 1)
    out = np.zeros(sm.shape, bool); out[j0:j1, i0:i1] = m
    return out


def initial(stacks):
    """-> (seg, info): seg[ser][k][bone] = bool mask; info[...] = dict(iou=1, known=True). Absent slices get empty masks."""
    seg, info = {}, {}
    for ser in ['601', '801']:
        st = stacks[ser]
        ks = sorted({k for b in SD.SEEDS[ser] for k in SD.SEEDS[ser][b]})
        for k in ks:
            sm = ndi.gaussian_filter(st['img'][k], 0.6)
            depth = _body_depth(sm, st['sp'])
            H, Wd = sm.shape; yy, xx = np.mgrid[:H, :Wd]
            masks, dist = {}, {}
            for b, kv in SD.SEEDS[ser].items():
                if k not in kv:
                    continue
                pts = np.array([_local_to_pix(st, k, u, v) for u, v in kv[k]])
                p = dict(SD.PARAMS[b]); p.update(SD.OVR.get(f'{ser}:{k}:{b}', {}))
                others = [_local_to_pix(st, k, u, v) for o, kv2 in SD.SEEDS[ser].items() if o != b and k in kv2 for u, v in kv2[k]]
                others += [_local_to_pix(st, k, u, v) for u, v in SD.BG.get(f'{ser}:{k}', [])]
                masks[b] = _rw_seeded(sm, pts, st['sp'], depth=depth, others=others, **p)
                dist[b] = np.min([np.hypot(xx - x, yy - y) for x, y in pts], 0)
            for b in masks:
                m = masks[b].copy()
                for o in masks:
                    if o != b:
                        m &= ~(masks[o] & (dist[o] < dist[b]))
                seg.setdefault(ser, {}).setdefault(k, {})[b] = m
                info.setdefault(ser, {}).setdefault(k, {})[b] = dict(iou=1.0)
        for b, kl in SD.ABSENT[ser].items():
            for k in kl:
                seg.setdefault(ser, {}).setdefault(k, {})[b] = np.zeros(st['img'].shape[1:], bool)
                info.setdefault(ser, {}).setdefault(k, {})[b] = dict(iou=1.0)
    return seg, info


# ---------- refinement on all series, seeded by a 3D prior ----------
_G = {}


def project_labels(lab, st, k):
    P = slice_world(st, k); v = world_to_vox(P.reshape(-1, 3))
    return ndi.map_coordinates(lab, v.T, order=0, cval=0).reshape(P.shape[:2])


def _job(a):
    ser, k = a
    st, lab, bones, ero, dil = _G['S'][ser], _G['lab'], _G['bones'], _G['ero'], _G['dil']
    sp = st['sp']
    L = project_labels(lab, st, k)
    if not (L >= 2).any():
        return ser, k, {}, {}
    img = ndi.gaussian_filter(st['img'][k].astype(np.float32), 0.5)
    out, info = {}, {}
    for b in bones:
        m0 = L == VAL[b]
        if m0.sum() * sp * sp < 6:
            continue
        ys, xs = np.nonzero(m0); pad = int(max(12, dil + 6) / sp)
        j0, j1 = max(0, ys.min() - pad), min(L.shape[0], ys.max() + pad)
        i0, i1 = max(0, xs.min() - pad), min(L.shape[1], xs.max() + pad)
        a_ = img[j0:j1, i0:i1]; mm = m0[j0:j1, i0:i1]; Lc = L[j0:j1, i0:i1]
        ins = ndi.binary_erosion(mm, disk(ero / sp))
        if ins.sum() * sp * sp < 1.5:
            ins = ndi.binary_erosion(mm, disk(1.0 / sp))
        if ins.sum() * sp * sp < 0.5:
            continue
        near = ndi.binary_dilation(mm, disk(dil / sp))
        outm = ~near | ndi.binary_erosion((Lc >= 2) & (Lc != VAL[b]), disk(0.8 / sp))
        an = np.clip(a_ / max(np.percentile(a_[near], 99), 1e-3), 0, 1.2)
        mk = np.zeros(a_.shape, np.int32); mk[outm] = 2; mk[ins] = 1
        try:
            m = random_walker(an, mk, beta=1500, mode='cg_j', tol=1e-3) == 1
        except Exception:
            continue
        m = ndi.binary_opening(ndi.binary_fill_holes(m), disk(0.6 / sp))
        l, n = ndi.label(m)
        if n > 1:
            ids = set(np.unique(l[ins])) - {0}
            if ids:
                m = np.isin(l, list(ids))
        full = np.zeros(L.shape, bool); full[j0:j1, i0:i1] = m
        out[b] = full
        info[b] = dict(iou=float((m & mm).sum() / max(1, (m | mm).sum())), a0=float(mm.sum() * sp * sp), a=float(m.sum() * sp * sp))
    return ser, k, out, info


def refine(stacks, lab, bones=BONES, ero=2.0, dil=2.5):
    _G.update(S=stacks, lab=lab, bones=list(bones), ero=ero, dil=dil)
    jobs = [(s, k) for s, st in stacks.items() for k in range(st['img'].shape[0])]
    if NPROC > 1:
        with Pool(NPROC) as p:
            res = p.map(_job, jobs, chunksize=2)
    else:
        res = [_job(j) for j in jobs]
    seg, info = {}, {}
    for ser, k, o, i in res:
        if o:
            seg.setdefault(ser, {})[k] = o; info.setdefault(ser, {})[k] = i
    return seg, info


def merge(a, b, bones):
    """Replace the contours of `bones` in (seg, info) a by those in b."""
    seg, info = a
    for ser in list(seg):
        for k in list(seg[ser]):
            for bb in bones:
                seg[ser][k].pop(bb, None); info[ser][k].pop(bb, None)
    for ser, kk in b[0].items():
        for k, bm in kk.items():
            for bb, m in bm.items():
                if bb in bones:
                    seg.setdefault(ser, {}).setdefault(k, {})[bb] = m
                    info.setdefault(ser, {}).setdefault(k, {})[bb] = b[1][ser][k][bb]
    return seg, info
