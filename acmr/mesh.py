"""Bone surfaces: optional 3D Gaussian on the SDF, cubic upsampling, marching cubes, Taubin smoothing."""
import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from .geometry import N, VOX, HALF, grid_axis


def taubin(V, F, it=20, lam=0.5, mu=-0.53):
    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = np.concatenate([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr(); A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    Wm = sp.diags(1 / deg) @ A
    V = V.copy()
    for _ in range(it):
        V = V + lam * (Wm @ V - V)
        V = V + mu * (Wm @ V - V)
    return V


def bone_mesh(phi, bone, gs=0.0, up=1.5, it=20, clav_cut=30.0):
    phi = phi.copy()
    if gs > 0:
        phi = ndi.gaussian_filter(phi, gs / VOX)
    g = grid_axis()
    if bone == 'clav':  # the medial clavicle beyond x = 30 mm is seen by one stack only: clean cut
        phi = np.maximum(phi, (g - clav_cut)[:, None, None])
    ins = phi < 0; l, k = ndi.label(ins)
    if k > 1:
        keep = l == (np.argmax(ndi.sum(ins, l, range(1, k + 1))) + 1)
        phi = np.where(ins & ~keep, np.abs(phi) + 0.1, phi)
    for a in range(3):
        sl = [slice(None)] * 3; sl[a] = [0, -1]; phi[tuple(sl)] = np.maximum(phi[tuple(sl)], 0.5)
    idx = np.argwhere(phi < 1.5); lo = np.maximum(idx.min(0) - 3, 0); hi = np.minimum(idx.max(0) + 4, N)
    sub = phi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    cs = [np.arange(0, n - 1 + 1e-6, 1 / up) for n in sub.shape]
    fine = ndi.map_coordinates(ndi.spline_filter(sub, 3), np.meshgrid(*cs, indexing='ij'), order=3, prefilter=False)
    V, F, _, _ = marching_cubes(fine, 0.0)
    V = (V / up + lo + 0.5) * VOX - HALF
    V = taubin(V, F, it=it)
    return V.astype(np.float32), F.astype(np.int32), phi


def write_ply(path, V, F):
    with open(path, 'wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n'
                f'element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n'.encode())
        f.write(V.astype('<f4').tobytes())
        rec = np.empty(len(F), dtype=[('n', 'u1'), ('i', '<i4', 3)]); rec['n'] = 3; rec['i'] = F
        f.write(rec.tobytes())
