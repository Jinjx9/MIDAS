import unittest
import torch
from midas.model import MIDAS


class BackendTests(unittest.TestCase):
    def test_unfold_preserves_all_outputs(self):
        torch.set_num_threads(1)
        native = MIDAS(input_dim=8,num_classes=3,num_layers=2,num_f_maps=8).eval()
        unfold = MIDAS(input_dim=8,num_classes=3,num_layers=2,num_f_maps=8,attention_impl='unfold').eval()
        unfold.load_state_dict(native.state_dict(),strict=True)
        x = torch.randn(1,8,37)
        with torch.no_grad():
            self.assertTrue(torch.equal(native(x),unfold(x)))

    def test_fast_backend_forward_and_gradients(self):
        torch.set_num_threads(1)
        m = MIDAS(input_dim=8, num_classes=3, num_layers=2, num_f_maps=8, backend='historical_fast')
        z = m(torch.randn(1,8,16))
        m.loss(z, torch.zeros(1,16,dtype=torch.long)).backward()
        self.assertTrue(torch.isfinite(m.backbone.encoder.conv_out.weight.grad).all())
        self.assertEqual(m.config['backend'], 'historical_fast')

    def test_unknown_backend_fails(self):
        with self.assertRaises(ValueError):
            MIDAS(backend='unknown')
