"""Super-resolution reconstruction of thick-slice stacks onto the 0.5 mm ROI grid.

data term   sum_s || A_s x - y_s ||^2   (slab forward model, per-stack rigid, per-stack smooth bias field removed)
regulariser lam * sum |grad x|_Huber    (edge preserving, solved by IRLS + CG)
Fat-bright volume (T2 coronal + T1 sagittal): the two contrasts disagree, so a joint solve leaves a grid artefact.
Each stack is reconstructed alone with Huber weights that also stop at the edges of a guide volume (the
three-direction PD), blurred along its own slice normal (where it has no information) and the two are averaged."""
import time
import numpy as np
from . import backend, slab, reg
from scipy import ndimage as ndi
from .geometry import N, HALF, VOX, rigid


def poly_basis(P):
    p = P / HALF
    cols = [np.ones(len(p))] + [p[:, i] for i in range(3)] + [p[:, i] * p[:, j] for i in range(3) for j in range(i, 3)]
    return np.stack(cols, 1)


def prepare(stacks, holdout=None):
    ops, val = [], []
    for st in stacks:
        op = slab.build(st)
        op['y'] = op['y'] / np.percentile(op['y'], 99)
        if holdout:
            m = (op['k'] % 3) == 0
            val.append(slab.subset(op, m)); op = slab.subset(op, ~m)
        ops.append(op)
    return ops, val


def _np_fwd(op, x):
    return (x[op['idx']] * op['w']).sum(1)


def _np_adj(op, r):
    return np.bincount(op['idx'].ravel(), (op['w'] * r[:, None]).ravel(), minlength=N ** 3).astype(np.float32)


def bias_correct(ops, rounds=2):
    """Per-stack quadratic log-polynomial bias field, estimated against the back-projected average (CPU)."""
    den = sum(_np_adj(o, np.ones_like(o['y'])) for o in ops)
    def bp():
        num = sum(_np_adj(o, o['y']) for o in ops)
        return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
    x0 = bp(); fields = {}
    for _ in range(rounds):
        for o in ops:
            pred = _np_fwd(o, x0)
            m = (pred > np.percentile(pred, 30)) & (o['y'] > 0.05)
            coef, *_ = np.linalg.lstsq(poly_basis(o['P'][m]), np.log(o['y'][m] / pred[m]), rcond=None)
            fields[o['ser']] = fields.get(o['ser'], 0) + coef
            o['y'] = (o['y'] / np.exp(poly_basis(o['P']) @ coef)).astype(np.float32)
        x0 = bp()
    return x0, fields


def solve(ops, x0, lam=0.6, delta=0.02, outer=5, inner=25, guide=None, gdelta=0.02, guide_only=False, B=None, log=print):
    B = B or backend.get()
    O = slab.Ops(B, ops)
    b = O.Aty()
    x = B.asarray(x0)
    gw = None
    if guide is not None:
        gw = reg.huber_weight(B, B.asarray(guide).reshape(N, N, N), gdelta)
    wgt = B.asarray(np.ones((N, N, N), np.float32)) if gw is None else gw
    t0 = time.time()
    for oi in range(outer):
        if oi > 0 and not (gw is not None and guide_only):
            w = reg.huber_weight(B, x.reshape(N, N, N), delta)
            wgt = w if gw is None else w * gw  # smoothing stops at edges of either volume
        A = lambda v: O.AtA(v) + lam * reg.divT(B, reg.grads(B, v.reshape(N, N, N)), wgt).reshape(-1) + 1e-4 * v
        x = backend.cg(B, A, b, x, inner)
        r = {o['ser']: round(float(np.corrcoef(B.numpy(O.fwd(o, x)), B.numpy(o['y']))[0, 1]), 4) for o in O.ops}
        log(f'  outer {oi} fit corr {r} ({time.time() - t0:.0f} s, {B.name})')
    return np.clip(B.numpy(x).reshape(N, N, N), 0, None).astype(np.float32)


def evaluate(val, x, fields):
    out = {}
    for o in val:
        y = (o['y'] / np.exp(poly_basis(o['P']) @ fields[o['ser']])).astype(np.float32)
        p = _np_fwd(o, x.ravel())
        out[o['ser']] = round(float(np.corrcoef(p, y)[0, 1]), 4)
    return out


def reconstruct(stacks, guide=None, holdout=False, log=print, **kw):
    ops, val = prepare(stacks, holdout)
    x0, fields = bias_correct(ops)
    x = solve(ops, x0, guide=guide, log=log, **kw)
    res = {'volume': x}
    if holdout:
        res['holdout_corr'] = evaluate(val, x, fields)
        res['holdout_corr_backprojection'] = evaluate(val, x0, fields)
    return res


def normal_blur(v, n, sigma):
    """Gaussian blur of volume v along unit direction n (mm)."""
    idx = np.indices(v.shape).astype(np.float32); out = 0; ws = 0
    for t in np.arange(-2.5, 2.51, 0.5) * sigma:
        w = np.exp(-0.5 * (t / sigma) ** 2); d = n * t / VOX
        out = out + w * ndi.map_coordinates(v, [idx[0] + d[0], idx[1] + d[1], idx[2] + d[2]], order=1, mode='nearest'); ws += w
    return (out / ws).astype(np.float32)


def reconstruct_guided_pair(stacks, guide, sigma=1.5, log=print, **kw):
    acc = 0
    for st in stacks:
        x = reconstruct([st], guide=guide, log=log, outer=4, **kw)['volume']
        x = x / np.percentile(x[x > 0.02], 99.5)
        R, _ = rigid(st['ser'])
        acc = acc + normal_blur(x, st['n'] @ R, sigma)
    return acc / len(stacks)
