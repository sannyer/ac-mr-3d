"""Geometric measurements on the bone meshes (non-diagnostic)."""
import numpy as np
from scipy.spatial import cKDTree


def measure(meshes):
    C, A, K = meshes['clav'][0], meshes['acro'][0], meshes['cora'][0]
    d, i = cKDTree(A).query(C); ci = int(np.argmin(d)); pc, pa = C[ci], A[i[ci]]
    d2, i2 = cKDTree(K).query(C); cj = int(np.argmin(d2))
    def top_near(V, p, r=6.0):
        sel = V[np.hypot(V[:, 0] - p[0], V[:, 2] - p[2]) < r]
        return float(sel[:, 1].max()), float(sel[:, 1].min())
    ct, cb = top_near(C, pc); at, ab = top_near(A, pa)
    r1 = lambda v: [round(float(x), 1) for x in v]
    return dict(ac_gap_min=round(float(d[ci]), 1), cc_min=round(float(d2[cj]), 1), clav_top=round(ct, 1), acro_top=round(at, 1),
                step_superior=round(ct - at, 1), step_inferior=round(cb - ab, 1), joint_mid=r1((pc + pa) / 2),
                p_clav=r1(pc), p_acro=r1(pa), p_cc_clav=r1(C[cj]), p_cc_cora=r1(K[i2[cj]]))
