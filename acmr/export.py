"""Viewer data (window.AC3D) and the derived-data files.

Viewer volumes are cropped to the bounding box of the reconstructed tissue, stored uint8 in [z][y][x] order (x fastest).
Labels: 1 soft-tissue zone around the joint, 2 clavicle, 3 acromion, 4 coracoid, 5 humerus. near = within 3 mm of bone."""
import base64, io, json
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from .geometry import N, VOX, HALF, NAMES, rigid

VALS = dict(clav=2, acro=3, cora=4, hum=5)
CROP_LO = np.array([23, 0, 13]); CROP_DIMS = np.array([157, 134, 113])  # same box as the published viewer


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def rle(L):
    ch = np.flatnonzero(np.diff(L)) + 1; st = np.r_[0, ch]; ln = np.diff(np.r_[st, len(L)])
    return np.stack([L[st].astype(np.uint32), ln.astype(np.uint32)], 1).ravel()


def soft_zone(bonelab, pd, meas):
    M = {b: bonelab == v for b, v in VALS.items()}
    D = {b: ndi.distance_transform_edt(~m) * VOX for b, m in M.items()}
    bone = bonelab > 0
    z = ((D['clav'] < 4) | (D['acro'] < 4) | (D['cora'] < 3) | ((D['clav'] + D['acro']) < meas['ac_gap_min'] + 12)
         | ((D['clav'] + D['cora']) < meas['cc_min'] + 8)) & ~bone
    z &= ndi.binary_fill_holes(ndi.gaussian_filter(pd, 2) > 0.08)
    lab = np.zeros((N, N, N), np.uint8); lab[z] = 1; lab[bone] = bonelab[bone]
    return lab


def native_atlas(st, size=84.0, px=240, zmax=40.0, cols=6, q=85):
    """JPEG atlas of native slices cropped around the AC centre, with each tile's corner/axes in the reference frame."""
    Rm, t = rigid(st['ser'])
    tiles, sl = [], []
    for k in range(st['img'].shape[0]):
        d = t - st['pos'][k]; z = float(-d @ st['n'])
        if abs(z) > zmax:
            continue
        u0, v0 = d @ st['r'] - size / 2, d @ st['c'] - size / 2
        g = (np.arange(px) + 0.5) * size / px
        ii = (u0 + g) / st['sp']; jj = (v0 + g) / st['sp']
        J, I = np.meshgrid(jj, ii, indexing='ij')
        tiles.append(ndi.map_coordinates(st['img'][k].astype(np.float32), [J, I], order=1, cval=0))
        o = st['pos'][k] + st['r'] * u0 + st['c'] * v0
        sl.append(dict(k=k, z=round(z, 2), o=[round(float(x), 3) for x in (o - t) @ Rm],
                       r=[round(float(x), 5) for x in st['r'] @ Rm], c=[round(float(x), 5) for x in st['c'] @ Rm]))
    T = np.stack(tiles); T = np.clip(T / np.percentile(T, 99.5) * 255, 0, 255).astype(np.uint8)
    rows = -(-len(T) // cols)
    A = np.zeros((rows * px, cols * px), np.uint8)
    for i, a in enumerate(T):
        A[(i // cols) * px:(i // cols + 1) * px, (i % cols) * px:(i % cols + 1) * px] = a
    buf = io.BytesIO(); Image.fromarray(A).save(buf, 'JPEG', quality=q)
    return dict(name=NAMES[st['ser']], size=size, px=px, cols=cols, rows=rows, th=st['th'],
                jpg=base64.b64encode(buf.getvalue()).decode(), sl=sl)


def viewer_data(pd, fat, lab, meshes, meas, natives):
    idx = np.argwhere(lab > 0)  # the published box, grown if the labels reach beyond it
    lo = np.minimum(CROP_LO, idx.min(0)); hi = np.maximum(CROP_LO + CROP_DIMS, idx.max(0) + 1)
    sl = tuple(slice(lo[i], hi[i]) for i in range(3))
    T = lambda a: np.ascontiguousarray(a[sl].transpose(2, 1, 0)).ravel()
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    D = ndi.distance_transform_edt(~(lab >= 2)) * VOX
    d = dict(vox=VOX, origin=[float(x) for x in (lo + 0.5) * VOX - HALF], dims=[int(x) for x in hi - lo],
             pd=b64(T(u8(pd))), fat=b64(T(u8(fat))), labrle=b64(rle(T(lab))), nearrle=b64(rle(T((D < 3).astype(np.uint8)))),
             meshes={}, meas=meas, natives=natives)
    for b, (V, F) in meshes.items():
        assert len(V) < 65536, f'{b}: too many vertices for Uint16 faces'
        d['meshes'][b] = dict(v=b64(V.astype(np.float32)), f=b64(F.astype(np.uint16)), nv=len(V), nf=len(F))
    return d


def write_viewer_js(path, d):
    with open(path, 'w') as f:
        f.write('window.AC3D=' + json.dumps(d, separators=(',', ':')) + ';\n')
