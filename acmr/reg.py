"""Regularisers on the N^3 grid (backend arrays shaped (N, N, N))."""
from .geometry import N


def grads(B, V):
    return [B.roll(V, -1, a) - V for a in range(3)]


def divT(B, g, wgt):
    out = 0
    for a in range(3):
        q = g[a] * wgt
        out = out + (B.roll(q, 1, a) - q)
    return out


def lap(B, V):
    out = -6 * V
    for a in range(3):
        out = out + B.roll(V, 1, a) + B.roll(V, -1, a)
    return out


def huber_weight(B, V, delta):
    g = grads(B, V)
    mag = B.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
    return delta / B.maximum(mag, delta)
