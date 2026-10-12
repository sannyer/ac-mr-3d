"""Full humeral-head prior: the first contours only reach the old physeal scar, so a sphere is fitted to the upper cortex
arcs of all humerus contours and used (cut at the ROI floor) as the prior for a second contour search."""
import numpy as np
from .geometry import slice_world, grid_axis
from scipy import ndimage as ndi

# Sphere read at the v4.1 review from the upper cortex arcs of the full contour set (viewer frame, mm). A fit to the
# first-pass contours alone underestimates it (they cover only the cap above the physeal scar), so this curated value
# is the default prior; fit_sphere() is still logged as a check.
SPHERE = (np.array([-13.3, -38.0, -10.0]), 19.0)


def fit_sphere(stacks, seg):
    P = []
    for ser, kk in seg.items():
        for k, bm in kk.items():
            if 'hum' not in bm or not bm['hum'].any():
                continue
            m = bm['hum']; e = m & ~ndi.binary_erosion(m)
            Wp = slice_world(stacks[ser], k)[e]
            P.append(Wp[Wp[:, 1] > Wp[:, 1].min() + 4])
    P = np.concatenate(P)
    keep = np.ones(len(P), bool)
    for _ in range(6):
        A = np.c_[2 * P[keep], np.ones(keep.sum())]; bb = (P[keep] ** 2).sum(1)
        x = np.linalg.lstsq(A, bb, rcond=None)[0]; c = x[:3]; r = float(np.sqrt(x[3] + c @ c))
        d = np.abs(np.linalg.norm(P - c, axis=1) - r)
        keep = d < max(1.0, 2.5 * np.median(d[keep]))
    return c, r, float(np.median(d[keep]))


def ball_prior(lab, c, r, floor=-43.0):
    g = grid_axis(); X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    ball = ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 < r * r) & (Y > floor)
    out = lab.copy(); out[out == 5] = 0; out[ball & (out == 0)] = 5
    return out


def replace(lab, phi):
    out = lab.copy(); out[out == 5] = 0; out[(phi < 0) & (out == 0)] = 5
    return out
