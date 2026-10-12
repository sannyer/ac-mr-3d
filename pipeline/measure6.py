"""Geometric measurements on the fused bone model (non-diagnostic)."""
import pickle, json, numpy as np
from scipy.spatial import cKDTree
from srr_common import HERE

import os; TAG=os.environ.get('TAG','b')
M = pickle.load(open(HERE + '/meshes6_'+TAG+'.pkl', 'rb'))
C, A, K = M['clav'][0], M['acro'][0], M['cora'][0]
tc, ta, tk = cKDTree(C), cKDTree(A), cKDTree(K)
d, i = ta.query(C)
ci = int(np.argmin(d)); ac_gap = float(d[ci]); pc, pa = C[ci], A[i[ci]]
d2, i2 = tk.query(C)
cj = int(np.argmin(d2)); cc = float(d2[cj])
# vertical step: superior border of each bone within 6 mm (horizontal) of its joint-facing point
def top_near(V, p, r=6.0):
    h = np.hypot(V[:, 0] - p[0], V[:, 2] - p[2])
    sel = V[h < r]
    return float(sel[:, 1].max()), float(sel[:, 1].min())
ct, cb = top_near(C, pc); at, ab = top_near(A, pa)
mid = (pc + pa) / 2
out = dict(ac_gap_min=round(ac_gap, 1), cc_min=round(cc, 1), clav_top=round(ct, 1), acro_top=round(at, 1),
           step_superior=round(ct - at, 1), step_inferior=round(cb - ab, 1), joint_mid=mid.round(1).tolist(),
           p_clav=pc.round(1).tolist(), p_acro=pa.round(1).tolist(), p_cc_clav=C[cj].round(1).tolist(), p_cc_cora=K[i2[cj]].round(1).tolist())
print(json.dumps(out, indent=1))
json.dump(out, open(HERE + '/measure6.json', 'w'))
