"""CPU checks; no dataset, checkpoint, training or model selection."""
import unittest
import torch
from core.smoothing import rts_smooth_batch, forward_only

class HistoricalFunctions(unittest.TestCase):
    def test_shape_dtype_and_finite(self):
        torch.manual_seed(3)
        x=torch.randn(1,7,30,dtype=torch.float64)
        for fn in (rts_smooth_batch,forward_only):
            y=fn(x,1e-4)
            self.assertEqual(y.shape,x.shape)
            self.assertEqual(y.dtype,x.dtype)
            self.assertTrue(torch.isfinite(y).all())

    def test_single_time_step(self):
        x=torch.ones(1,7,1)
        for fn in (rts_smooth_batch,forward_only):
            self.assertTrue(torch.equal(fn(x,1e-4),x))

    def test_forward_causal_backward_noncausal(self):
        x=torch.zeros(1,3,32,dtype=torch.float64)
        changed=x.clone();changed[...,16:]=2
        self.assertTrue(torch.equal(forward_only(x,1e-4)[...,:16],forward_only(changed,1e-4)[...,:16]))
        self.assertGreater((rts_smooth_batch(x,1e-4)[...,:16]-rts_smooth_batch(changed,1e-4)[...,:16]).abs().sum().item(),0)

if __name__=='__main__':unittest.main()
