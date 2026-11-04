"""Historical Cholec80 scan backend. Kept distinct from sequential inference.
Gain recursion uses detached scale and a variance floor; not advertised as
identical to the sequential backend in all edge cases.
"""
import numpy as np
import torch

def _scan_coeffs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix scan of x_t = a_t * x_{t-1} + b_t with x_{-1} = 0.

    a, b: (C, T). Returns x: (C, T). Fully differentiable w.r.t. b (and a).
    """
    T = a.shape[1]
    step = 1
    while step < T:
        a_prev = torch.cat([torch.ones_like(a[:, :step]), a[:, :-step]], dim=1)
        b_prev = torch.cat([torch.zeros_like(b[:, :step]), b[:, :-step]], dim=1)
        b = b + a * b_prev
        a = a * a_prev
        step *= 2
    return b


def _p_recursion(lam: float, s: float, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Scalar Kalman gain recursion in float64 numpy (same for all channels).

    Matches rts_smooth_diff seeding: x_f(0)=z(0) (k_0=1) and p_f(0)=r."""
    q = lam * s * s
    r = max(s * s, 1e-6)
    ks = np.empty(T, dtype=np.float64)
    ps = np.empty(T, dtype=np.float64)
    ks[0] = 1.0
    ps[0] = r
    p = r
    for t in range(1, T):
        p_pred = p + q
        k = p_pred / (p_pred + r)
        ks[t] = k
        ps[t] = (1.0 - k) * p_pred
        p = ps[t]
    return ks, ps


def fast_rts_smooth(logits: torch.Tensor, lam: float) -> torch.Tensor:
    """logits (C, T) -> smoothed (C, T). Exact same recursion as rts_smooth_diff
    with log_lam = log(lam) constant and s = logits.std().detach()."""
    C, T = logits.shape
    dev = logits.device
    s = float(logits.std().detach())
    ks, ps = _p_recursion(lam, s, T)
    k = torch.from_numpy(ks).to(dev).float().unsqueeze(0)          # (1, T)
    p_f = torch.from_numpy(ps).to(dev).float().unsqueeze(0)        # (1, T)
    q = torch.full((1, T), lam * s * s, device=dev, dtype=torch.float32)
    g = p_f / (p_f + q)                                            # (1, T), RTS gain

    # forward filter: x_f(t) = (1-k_t) x_f(t-1) + k_t z_t, x_f(-1) = 0
    a = 1.0 - k.expand(C, T)
    b = (k * logits).contiguous()
    x_f = _scan_coeffs(a, b)

    # backward RTS: x_s(t) = (1-g_t) x_f(t) + g_t x_s(t+1), x_s(T-1) = x_f(T-1)
    # reverse and reuse the same scan; seed y_0 = x_f(T-1).
    g_rev = torch.flip(g.expand(C, T), dims=[1])
    xf_rev = torch.flip(x_f, dims=[1])
    c = g_rev
    d = (1.0 - g_rev) * xf_rev
    # reversed recursion: y_j = c_j y_{j-1} + d_j with y_0 = d_0 (= x_f(T-1),
    # since g at the last frame is multiplied by x_s(T) which doesn't exist ->
    # original recursion at t=T-1 has no g term; enforce by zeroing c[:,0]).
    c = torch.cat([torch.zeros_like(c[:, :1]), c[:, 1:]], dim=1)
    d = d.contiguous().clone()
    d[:, 0] = xf_rev[:, 0]        # seed: x_s(T-1) = x_f(T-1) exactly
    x_s_rev = _scan_coeffs(c, d)
    x_s = torch.flip(x_s_rev, dims=[1])
    return x_s
