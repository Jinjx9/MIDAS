"""Exact historical sequential inference; no added trainable parameters."""

import torch

def rts_smooth_batch(logits: torch.Tensor, lam: float) -> torch.Tensor:
    """Bidirectional RTS smoother, batch dim kept. logits (B,C,T) -> (B,C,T)."""
    B = logits.shape[0]
    outs = []
    for b in range(B):
        x = logits[b]
        s = x.std().clamp_min(1e-6)
        q = lam * s * s
        r = s * s
        C, T = x.shape
        xf = torch.empty_like(x)
        pf = torch.empty(C, T, dtype=x.dtype, device=x.device)
        xp = x[:, 0].clone()
        pp = torch.full((C,), float(r), dtype=x.dtype, device=x.device)
        xf[:, 0] = xp
        pf[:, 0] = pp
        for t in range(1, T):
            pp = pp + q
            k = pp / (pp + r)
            xf[:, t] = xp + k * (x[:, t] - xp)
            pf[:, t] = (1 - k) * pp
            xp, pp = xf[:, t], pf[:, t]
        xs = xf.clone()
        for t in range(T - 2, -1, -1):
            g = pf[:, t] / (pf[:, t] + q)
            xs[:, t] = xf[:, t] + g * (xs[:, t + 1] - xf[:, t])
        outs.append(xs)
    return torch.stack(outs, dim=0)

def forward_only(logits,lam):
    outs=[]
    for x in logits:
        s=x.std().clamp_min(1e-6);q=lam*s*s;r=s*s
        C,T=x.shape;xf=torch.empty_like(x);xp=x[:,0].clone()
        pp=torch.full((C,),float(r),dtype=x.dtype,device=x.device);xf[:,0]=xp
        for t in range(1,T):
            pp=pp+q;k=pp/(pp+r)
            xf[:,t]=xp+k*(x[:,t]-xp);pp=(1-k)*pp;xp=xf[:,t]
        outs.append(xf)
    return torch.stack(outs)
