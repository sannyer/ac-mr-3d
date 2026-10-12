"""Shared helpers for super-resolution reconstruction of the AC joint.
Viewer/volume coordinates: x = patient left, y = superior, z = anterior, origin at AC joint centre W (mm)."""
import os, pydicom, glob, numpy as np
from scipy import ndimage as ndi

HERE = os.environ.get('AC3D_WORK', '/tmp/ac3d_work')  # must contain dcm/ with all 181 DICOMs; intermediates are written here
W = np.array([-123.41535402, -40.52544073, 63.64495086])
M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # LPS -> viewer

HALF = 45.0   # ROI half size (mm)
VOX = 0.5     # output voxel (mm)
N = int(round(2 * HALF / VOX))  # 180

REG = {  # moving -> reference 301, (tx,ty,tz mm, rx,ry,rz deg)
    ('301', '501'): np.array([-0.096, 0.905, -0.685, -0.345, 0.401, -0.469]),
    ('301', '701'): np.array([-0.057, 1.586, -0.988, -0.365, 0.476, -0.261]),
}
SIB = {'301': '301', '401': '301', '501': '501', '601': '501', '701': '701', '801': '701'}


def to_view(p):
    return (np.asarray(p) - W) @ M.T


_HDR = None
def _headers():
    global _HDR
    if _HDR is None:
        _HDR = {}
        for f in glob.glob(HERE + '/dcm/*'):
            d = pydicom.dcmread(f, stop_before_pixels=True)
            _HDR.setdefault(str(d.SeriesNumber), []).append(f)
    return _HDR


def load_stack(ser, inplane=0.5):
    """Return dict with pixel data downsampled to ~inplane mm, plus geometry in viewer coords."""
    ds = [pydicom.dcmread(f) for f in _headers()[ser]]
    ds.sort(key=lambda d: int(d.InstanceNumber))
    o = np.array(ds[0].ImageOrientationPatient, float)
    r, c = (o[:3]) @ M.T, (o[3:]) @ M.T
    n = np.cross(r, c)
    ps = float(ds[0].PixelSpacing[0])
    f = max(1, int(round(inplane / ps)))
    imgs, pos = [], []
    for d in ds:
        a = d.pixel_array.astype(np.float32)
        R = (a.shape[0] // f) * f
        a = a[:R, :R].reshape(R // f, f, R // f, f).mean((1, 3))
        imgs.append(a)
        P = to_view(np.array(d.ImagePositionPatient, float))
        pos.append(P + (r + c) * ps * (f - 1) / 2)
    pos = np.array(pos)
    z = pos @ n
    if np.median(np.diff(z)) < 0:  # make slice order follow +n
        imgs, pos, z = imgs[::-1], pos[::-1], z[::-1]
    return dict(ser=ser, img=np.stack(imgs), pos=pos, r=r, c=c, n=n, sp=ps * f,
                th=float(ds[0].SliceThickness), gap=float(np.median(np.diff(z))))


def stack_points(st, crop=True, margin=6.0):
    Z, R, C = st['img'].shape
    jj, ii = np.meshgrid(np.arange(R), np.arange(C), indexing='ij')
    P = st['pos'][:, None, None, :] + st['r'][None, None, None, :] * (ii[None, ..., None] * st['sp']) \
        + st['c'][None, None, None, :] * (jj[None, ..., None] * st['sp'])
    P = P.reshape(-1, 3)
    v = st['img'].reshape(-1)
    if crop:
        m = np.all(np.abs(P) < HALF + margin, axis=1)
        return P[m], v[m], np.nonzero(m)[0]
    return P, v, np.arange(len(v))


def world_to_vox(P):
    return (P + HALF) / VOX - 0.5


def grid_points():
    g = (np.arange(N) + 0.5) * VOX - HALF
    X, Y, Zg = np.meshgrid(g, g, g, indexing='ij')
    return np.stack([X, Y, Zg], -1).reshape(-1, 3)


def rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rigid(ser):
    """(R, t) mapping reference-frame point P -> P R^T + t in the stack's frame."""
    s = SIB[ser]
    if s == '301':
        return np.eye(3), np.zeros(3)
    p = REG[('301', s)]
    return rot(*np.deg2rad(p[3:])), p[:3]


def stack_coords(st, P):
    """Continuous (k, j, i) index of reference-frame points P in stack st (rigid applied)."""
    Rm, t = rigid(st['ser'])
    Q = P @ Rm.T + t
    d = Q - st['pos'][0]
    return d @ st['n'] / st['gap'], d @ st['c'] / st['sp'], d @ st['r'] / st['sp']
