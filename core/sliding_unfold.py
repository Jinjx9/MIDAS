"""Historical authorized 50Salads window construction; input-device adaptation only."""
import torch
import torch.nn.functional as F
from midas.asformer import AttentionHelper

def UNFOLD_IMPL(self, q, k, v, mask):
    m, c1, L = q.size(); _, c2, _ = k.size(); _, c3, _ = v.size()
    assert m == 1
    nb = L // self.bl
    if L % self.bl != 0:
        pad = self.bl - L % self.bl
        q = torch.cat([q, torch.zeros(m, c1, pad, device=q.device)], dim=-1)
        k = torch.cat([k, torch.zeros(m, c2, pad, device=k.device)], dim=-1)
        v = torch.cat([v, torch.zeros(m, c3, pad, device=v.device)], dim=-1)
        nb += 1
    pm = torch.cat([mask[:, 0:1, :].reshape(m, 1, L),
                    torch.zeros(m, 1, self.bl * nb - L, device=q.device)], dim=-1)
    q = q.reshape(m, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(nb, c1, self.bl)
    k = torch.cat([torch.zeros(m, c2, self.bl // 2, device=q.device), k,
                   torch.zeros(m, c2, self.bl // 2, device=q.device)], dim=-1)
    v = torch.cat([torch.zeros(m, c3, self.bl // 2, device=q.device), v,
                   torch.zeros(m, c3, self.bl // 2, device=q.device)], dim=-1)
    pm = torch.cat([torch.zeros(m, 1, self.bl // 2, device=q.device), pm,
                    torch.zeros(m, 1, self.bl // 2, device=q.device)], dim=-1)
    W = self.bl + 2 * (self.bl // 2)
    k = k.unfold(-1, W, self.bl).squeeze(0).permute(1, 0, 2).reshape(nb, c2, W)
    v = v.unfold(-1, W, self.bl).squeeze(0).permute(1, 0, 2).reshape(nb, c3, W)
    pm = pm.unfold(-1, W, self.bl).squeeze(0).permute(1, 0, 2).reshape(nb, 1, W)
    fm = self.window_mask.to(q.device).repeat(nb, 1, 1) * pm
    out, _ = AttentionHelper().scalar_dot_att(q, k, v, fm)
    out = self.conv_out(F.relu(out))
    out = out.reshape(m, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m, -1, nb * self.bl)
    out = out[:, :, 0:L]
    return out * mask[:, 0:1, :]
