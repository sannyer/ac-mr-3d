"""Array backend: PyTorch on CUDA / Apple MPS when available, otherwise NumPy (CPU).

Only the heavy linear algebra (slab forward model, its adjoint, regularisers, conjugate gradients) runs through here.
Select with the ACMR_DEVICE environment variable: auto (default), cuda, mps, cpu-torch, numpy."""
import os
import numpy as np

try:
    import torch
except Exception:  # torch is optional
    torch = None


class NumpyBackend:
    name = 'numpy'
    def asarray(self, a, dtype=np.float32):
        return np.ascontiguousarray(a, dtype=dtype)
    def index(self, a):
        return np.ascontiguousarray(a, dtype=np.int32)
    def numpy(self, a):
        return np.asarray(a)
    def zeros(self, n):
        return np.zeros(n, np.float32)
    def fwd(self, x, idx, w):
        return (x[idx] * w).sum(1)
    def adj(self, r, idx, w, n):
        return np.bincount(idx.ravel(), (w * r[:, None]).ravel(), minlength=n).astype(np.float32)
    def roll(self, v, s, a):
        return np.roll(v, s, a)
    def dot(self, a, b):
        return float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)))
    def sqrt(self, a):
        return np.sqrt(a)
    def maximum(self, a, b):
        return np.maximum(a, b)


class TorchBackend:
    def __init__(self, device):
        self.dev = torch.device(device)
        self.name = 'torch-' + device
    def asarray(self, a, dtype=np.float32):
        return torch.as_tensor(np.ascontiguousarray(a, dtype=dtype), device=self.dev)
    def index(self, a):
        return torch.as_tensor(np.ascontiguousarray(a, dtype=np.int64), device=self.dev)
    def numpy(self, a):
        return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    def zeros(self, n):
        return torch.zeros(n, dtype=torch.float32, device=self.dev)
    def fwd(self, x, idx, w):
        return (x[idx] * w).sum(1)
    def adj(self, r, idx, w, n):
        out = torch.zeros(n, dtype=torch.float32, device=self.dev)
        out.index_add_(0, idx.reshape(-1), (w * r[:, None]).reshape(-1))
        return out
    def roll(self, v, s, a):
        return torch.roll(v, s, a)
    def dot(self, a, b):
        return float((a.reshape(-1).double() * b.reshape(-1).double()).sum())
    def sqrt(self, a):
        return torch.sqrt(a)
    def maximum(self, a, b):
        return torch.clamp(a, min=b) if not torch.is_tensor(b) else torch.maximum(a, b)


def get(choice=None):
    choice = (choice or os.environ.get('ACMR_DEVICE', 'auto')).lower()
    if choice == 'numpy' or torch is None:
        if choice not in ('auto', 'numpy'):
            raise RuntimeError('PyTorch is not installed; use ACMR_DEVICE=numpy')
        return NumpyBackend()
    if choice == 'auto':
        if torch.cuda.is_available():
            return TorchBackend('cuda')
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return TorchBackend('mps')
        return NumpyBackend()
    if choice == 'cpu-torch':
        return TorchBackend('cpu')
    return TorchBackend(choice)


def cg(B, A, b, x0, iters, log=None):
    """Conjugate gradients for the SPD system A x = b on backend B."""
    x = x0.clone() if hasattr(x0, 'clone') else x0.copy()
    r = b - A(x)
    p = r.clone() if hasattr(r, 'clone') else r.copy()
    rs = B.dot(r, r); r0 = rs
    for i in range(iters):
        Ap = A(p)
        pAp = B.dot(p, Ap)
        if pAp <= 0:
            break
        a = rs / pAp
        x = x + a * p
        r = r - a * Ap
        rn = B.dot(r, r)
        if rn < 1e-12 * r0:
            break
        p = r + (rn / rs) * p
        rs = rn
    return x
