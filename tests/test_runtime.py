import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from midas.model import MIDAS, read_checkpoint


class RuntimeTests(unittest.TestCase):
    def test_historical_numpy_metadata_is_read_without_unsafe_pickle(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'legacy.pt'
            torch.save({'val': {'Edit': np.float64(80.)}}, path)
            self.assertEqual(read_checkpoint(path)['val']['Edit'], 80.)

    def test_forward_and_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = MIDAS(input_dim=8, num_classes=3, num_layers=2, num_f_maps=8)
        x = torch.randn(1, 8, 16)
        result = model(x)
        self.assertEqual(tuple(result.shape), (4, 1, 3, 16))
        loss = model.loss(result, torch.zeros(1, 16, dtype=torch.long))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.backbone.encoder.conv_1x1.weight.grad.abs().sum(), 0)
        self.assertGreater(model.backbone.decoders[-1].conv_out.weight.grad.abs().sum(), 0)

    def test_baseline_has_same_parameter_keys(self):
        a = MIDAS(input_dim=8, num_classes=3, num_layers=2, num_f_maps=8)
        b = MIDAS(input_dim=8, num_classes=3, num_layers=2, num_f_maps=8, mid=False)
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))

    def test_invalid_features_fail(self):
        model = MIDAS(input_dim=8, num_classes=3, num_layers=2, num_f_maps=8)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 8, 16))


if __name__ == '__main__':
    unittest.main()
