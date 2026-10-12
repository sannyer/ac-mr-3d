import numpy as np
import scipy.sparse as sp
def taubin(V, F, it=50, lam=0.5, mu=-0.53):
    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    W = sp.diags(1 / deg) @ A
    V = V.copy()
    for _ in range(it):
        V = V + lam * (W @ V - V)
        V = V + mu * (W @ V - V)
    return V


