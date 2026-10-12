"""Super-resolution reconstruction from several thick-slice stacks onto the 0.5 mm ROI grid.
Forward model: slice-profile (box, NS samples through the slab) x trilinear interpolation, per-stack rigid transform.
Regulariser: Tikhonov gradient ('lap') or edge-preserving Huber-TV solved by IRLS ('huber').
usage: python srr2.py OUT STACKS [lam] [delta] [outer] [holdout]
  STACKS: comma list, e.g. 301,501,701
  holdout: 'k3' -> slices with k%3==0 are left out of the fit and reported as validation"""
import sys, time, pickle
from srr_common import *
from scipy.sparse.linalg import LinearOperator, cg

NS = 7


def build_op(st, inplane_keep=None):
    Rm, t = rigid(st['ser'])
    Q, y, flat = stack_points(st, crop=True, margin=4.0)
    kidx = flat // (st['img'].shape[1] * st['img'].shape[2])
    P0 = (Q - t) @ Rm  # into reference frame
    n = st['n'] @ Rm
    offs = (np.arange(NS) + 0.5) / NS * st['th'] - st['th'] / 2
    rows = len(P0)
    idx = np.zeros((rows, NS * 8), np.int32)
    wts = np.zeros((rows, NS * 8), np.float32)
    k = 0
    for s in offs:
        v = world_to_vox(P0 + s * n)
        f = np.floor(v).astype(np.int32); fr = (v - f).astype(np.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix, iy, iz = f[:, 0] + dx, f[:, 1] + dy, f[:, 2] + dz
                    w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dz else 1 - fr[:, 2])
                    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & (iz >= 0) & (iz < N)
                    idx[:, k] = np.where(ok, (ix * N + iy) * N + iz, 0)
                    wts[:, k] = np.where(ok, w, 0) / NS
                    k += 1
    keep = wts.sum(1) > 0.6
    return dict(ser=st['ser'], idx=idx[keep], w=wts[keep], y=y[keep].astype(np.float32), P=P0[keep], k=kidx[keep])


def subset(op, m):
    return {kk: (vv[m] if isinstance(vv, np.ndarray) else vv) for kk, vv in op.items()}


def fwd(op, x):
    return (x[op['idx']] * op['w']).sum(1)


def adj(op, r):
    return np.bincount(op['idx'].ravel(), (op['w'] * r[:, None]).ravel(), minlength=N ** 3).astype(np.float32)


def grads(v):
    return [np.roll(v, -1, a) - v for a in range(3)]


def divT(g, wgt):
    """D^T (W D x) given g = D x (list) and weights wgt (list or array)."""
    out = 0
    for a in range(3):
        q = g[a] * (wgt[a] if isinstance(wgt, list) else wgt)
        out = out + (np.roll(q, 1, a) - q)
    return out


def poly_basis(P, deg=2):
    p = P / HALF
    cols = [np.ones(len(p))]
    for i in range(3):
        cols.append(p[:, i])
    for i in range(3):
        for j in range(i, 3):
            cols.append(p[:, i] * p[:, j])
    return np.stack(cols, 1)


def prepare(sers, holdout=None, inplane=0.3):
    ops, val = [], []
    for s in sers:
        st = load_stack(s, inplane=inplane)
        op = build_op(st)
        op['y'] /= np.percentile(op['y'], 99)
        if holdout == 'k3':
            m = (op['k'] % 3) == 0
            val.append(subset(op, m)); op = subset(op, ~m)
        ops.append(op)
    return ops, val


def bias_correct(ops, rounds=2):
    den = sum(adj(o, np.ones_like(o['y'])) for o in ops)
    num = sum(adj(o, o['y']) for o in ops)
    x0 = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
    fields = {}
    for it in range(rounds):
        for o in ops:
            pred = fwd(o, x0)
            m = (pred > np.percentile(pred, 30)) & (o['y'] > 0.05)
            B = poly_basis(o['P'][m])
            coef, *_ = np.linalg.lstsq(B, np.log(o['y'][m] / pred[m]), rcond=None)
            fields[o['ser']] = coef if o['ser'] not in fields else fields[o['ser']] + coef
            o['y'] = (o['y'] / np.exp(poly_basis(o['P']) @ coef)).astype(np.float32)
        num = sum(adj(o, o['y']) for o in ops)
        x0 = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
    return x0, fields


def solve(ops, x0, lam, mode='huber', delta=0.02, outer=4, inner=25, wst=None, log=print):
    wst = wst or {o['ser']: 1.0 for o in ops}
    b = sum(wst[o['ser']] * adj(o, o['y']) for o in ops)
    x = x0.copy()
    wgt = np.ones(N ** 3, np.float32).reshape(N, N, N)
    for oi in range(outer if mode == 'huber' else 1):
        if mode == 'huber' and oi > 0:
            g = grads(x.reshape(N, N, N))
            mag = np.sqrt(sum(gi * gi for gi in g))
            wgt = (delta / np.maximum(mag, delta)).astype(np.float32)  # Huber IRLS weight, =1 in smooth areas
        def mv(v):
            v = v.astype(np.float32)
            V = v.reshape(N, N, N)
            return sum(wst[o['ser']] * adj(o, fwd(o, v)) for o in ops) + lam * divT(grads(V), wgt).ravel() + 1e-4 * v
        A = LinearOperator((N ** 3, N ** 3), matvec=mv, dtype=np.float32)
        x, info = cg(A, b, x0=x, maxiter=inner)
        x = x.astype(np.float32)
        r = {o['ser']: round(float(np.corrcoef(fwd(o, x), o['y'])[0, 1]), 4) for o in ops}
        log('outer', oi, 'fit corr', r)
    return x


def evaluate(val, x):
    out = {}
    for o in val:
        p = fwd(o, x); y = o['y']
        out[o['ser']] = (round(float(np.corrcoef(p, y)[0, 1]), 4), round(float(np.sqrt(((p - y) ** 2).mean())), 4))
    return out


def apply_bias(val, fields):
    for o in val:
        o['y'] = (o['y'] / np.exp(poly_basis(o['P']) @ fields[o['ser']])).astype(np.float32)


if __name__ == '__main__':
    t0 = time.time()
    out = sys.argv[1]; sers = sys.argv[2].split(',')
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    delta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    outer = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    hold = sys.argv[6] if len(sys.argv) > 6 else None
    mode = 'lap' if outer == 0 else 'huber'
    ops, val = prepare(sers, hold)
    print('rows', {o['ser']: len(o['y']) for o in ops}, round(time.time() - t0), 's', flush=True)
    x0, fields = bias_correct(ops)
    x = solve(ops, x0, lam, mode, delta, max(outer, 1), log=lambda *a: print(*a, round(time.time() - t0), 's', flush=True))
    if val:
        apply_bias(val, fields)
        print('HOLDOUT', out, 'lam', lam, 'delta', delta, 'outer', outer, evaluate(val, x), 'x0', evaluate(val, x0), flush=True)
    np.save(HERE + '/' + out + '.npy', np.clip(x.reshape(N, N, N), 0, None).astype(np.float32))
    print('done', round(time.time() - t0), 's')
