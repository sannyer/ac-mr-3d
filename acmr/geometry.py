"""Coordinate frames, ROI grid and inter-series rigid registration.

Viewer/volume frame: x = patient left, y = superior, z = anterior (mm), origin at the AC-joint centre W (DICOM LPS).
The ROI is a cube of +-HALF mm sampled at VOX mm (N^3 voxels, index order x, y, z)."""
import numpy as np

W = np.array([-123.41535402, -40.52544073, 63.64495086])
M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # LPS -> viewer

HALF = 45.0
VOX = 0.5
N = int(round(2 * HALF / VOX))  # 180

# moving -> reference 301 (tx, ty, tz mm, rx, ry, rz deg); estimated by NCC + Powell on the PD stacks
REG = {
    '501': np.array([-0.096, 0.905, -0.685, -0.345, 0.401, -0.469]),
    '701': np.array([-0.057, 1.586, -0.988, -0.365, 0.476, -0.261]),
}
SIB = {'301': '301', '401': '301', '501': '501', '601': '501', '701': '701', '801': '701'}
SERIES = ['301', '401', '501', '601', '701', '801']
NAMES = {'301': 'PD zsírelnyomott · axiális', '401': 'T2 zsírelnyomott · axiális', '501': 'PD zsírelnyomott · koronális',
         '601': 'T2 · koronális', '701': 'PD zsírelnyomott · szagittális', '801': 'T1 · szagittális'}


def to_view(p):
    return (np.asarray(p) - W) @ M.T


def rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rigid(ser):
    """(R, t) mapping a reference-frame point P -> P R^T + t in the stack's own frame."""
    s = SIB[ser]
    if s == '301':
        return np.eye(3), np.zeros(3)
    p = REG[s]
    return rot(*np.deg2rad(p[3:])), p[:3]


def world_to_vox(P):
    return (P + HALF) / VOX - 0.5


def grid_axis():
    return (np.arange(N) + 0.5) * VOX - HALF


def grid_points():
    g = grid_axis()
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    return np.stack([X, Y, Z], -1).reshape(-1, 3)


def stack_coords(st, P):
    """Continuous (k, j, i) index of reference-frame points P in stack st."""
    Rm, t = rigid(st['ser'])
    Q = P @ Rm.T + t
    d = Q - st['pos'][0]
    return d @ st['n'] / st['gap'], d @ st['c'] / st['sp'], d @ st['r'] / st['sp']


def slice_world(st, k):
    """Reference-frame position of every pixel of slice k, shape (rows, cols, 3)."""
    Z, R, C = st['img'].shape
    jj, ii = np.mgrid[:R, :C]
    P = st['pos'][k] + np.outer(ii.ravel(), st['r']) * st['sp'] + np.outer(jj.ravel(), st['c']) * st['sp']
    Rm, t = rigid(st['ser'])
    return ((P - t) @ Rm).reshape(R, C, 3)
