"""Portable wrapper matching the historical sequential AutoLaparo runtime."""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from types import MethodType

from .asformer import MyTransformer, AttLayer
from core.smoothing import rts_smooth_batch
from core.fast_smoothing import fast_rts_smooth
from core.sliding_unfold import UNFOLD_IMPL


def read_checkpoint(path):
    # Historical validation dictionaries contain np.float64 scalars. Allow those
    # types explicitly; never fall back to arbitrary pickle execution.
    from numpy.core.multiarray import scalar
    with torch.serialization.safe_globals([scalar, np.dtype, type(np.dtype('float64'))]):
        return torch.load(path, map_location='cpu', weights_only=True)


class MIDAS(nn.Module):
    def __init__(self, input_dim=2048, num_classes=7, num_layers=10,
                 num_f_maps=64, num_decoders=3, lam=1e-4, mid=True, backend='sequential', attention_impl='native'):
        super().__init__()
        if lam <= 0:
            raise ValueError('lam must be positive')
        if backend not in ('sequential', 'historical_fast'):
            raise ValueError('Unknown smoothing backend')
        if attention_impl not in ('native','unfold'):
            raise ValueError('Unknown attention window implementation')
        self.backbone = MyTransformer(num_decoders, num_layers, 2, 2,
                                      num_f_maps, input_dim, num_classes, .3)
        if attention_impl == 'unfold':
            for layer in self.backbone.modules():
                if isinstance(layer,AttLayer):
                    layer._sliding_window_self_att = MethodType(UNFOLD_IMPL,layer)
        self.lam, self.mid = lam, mid
        self.backend = backend
        self.config = dict(input_dim=input_dim, num_classes=num_classes,
                           num_layers=num_layers, num_f_maps=num_f_maps,
                           num_decoders=num_decoders, lam=lam, mid=mid, backend=backend, attention_impl=attention_impl)

    def forward(self, features):
        # Batch size one preserves full-video historical smoothing/masking semantics.
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[2] < 2:
            raise ValueError('Expected one full video (1, channels, time>=2)')
        if not torch.isfinite(features).all():
            raise ValueError('Features contain NaN or infinity')
        mask = features.new_ones(1, 1, features.shape[-1])
        out, feature = self.backbone.encoder(features, mask)
        outputs = [out]
        for i, decoder in enumerate(self.backbone.decoders):
            out, feature = decoder(F.softmax(out, dim=1) * mask,
                                   feature * mask, mask)
            if self.mid and i < len(self.backbone.decoders)-1:
                out = (rts_smooth_batch(out, self.lam) if self.backend == 'sequential'
                       else torch.stack([fast_rts_smooth(x, self.lam) for x in out]))
            outputs.append(out)
        return torch.stack(outputs)

    @staticmethod
    def loss(stages, target):
        loss = 0
        for p in stages:
            loss = loss + F.cross_entropy(p.transpose(2, 1).reshape(-1, p.shape[1]),
                                          target.reshape(-1), ignore_index=-100)
            delta = F.mse_loss(F.log_softmax(p[:, :, 1:], dim=1),
                               F.log_softmax(p.detach()[:, :, :-1], dim=1),
                               reduction='none')
            loss = loss + .15 * delta.clamp(0, 16).mean()
        return loss

    def load_checkpoint(self, path):
        """Only use trusted checkpoints. Strict backbone loading; no silent omissions."""
        state = read_checkpoint(path)
        weights = state.get('backbone', state.get('model', state))
        if all(k.startswith('backbone.') for k in weights):
            weights = {k[len('backbone.'):]: v for k, v in weights.items()}
        self.backbone.load_state_dict(weights, strict=True)
        return state
