import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from midas.model import MIDAS
from midas.__main__ import apply_inference_precision


class CLITests(unittest.TestCase):
    def test_public_evaluator_exposes_only_supported_protocols(self):
        result=subprocess.run([sys.executable,'-m','midas','evaluate','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('c80_legacy',result.stdout)
        self.assertIn('half_open',result.stdout)

    def test_recipe_precision_disables_tf32_explicitly(self):
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            apply_inference_precision({'allow_tf32':False})
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    def test_historical_recipe_checks_weights_and_runs(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = MIDAS(input_dim=8,num_classes=3,num_layers=2,num_f_maps=8,mid=False).eval()
            torch.save({'model':model.state_dict(),'epoch':1,'binding':'fixture'},root/'best.pt')
            recipe = {'checkpoint_sha256':hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),
                      'model':model.config}
            (root/'recipe.json').write_text(json.dumps(recipe))
            np.save(root/'v.npy',np.zeros((8,16),dtype=np.float32))
            (root/'test.bundle').write_text('v.txt\n')
            cmd = [sys.executable,'-m','midas','predict','--features',str(root),'--split',str(root/'test.bundle'),
                   '--checkpoint',str(root/'best.pt'),'--recipe',str(root/'recipe.json'),
                   '--output',str(root/'pred.json')]
            good = subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(good.returncode,0,good.stderr)
            recipe['checkpoint_sha256'] = '0'*64
            (root/'recipe.json').write_text(json.dumps(recipe))
            bad = subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn('SHA256',bad.stderr)

    def test_train_predict_evaluate_without_parent_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'features').mkdir()
            (root/'groundTruth').mkdir()
            for name in ('train', 'val', 'test'):
                np.save(root/'features'/f'{name}.npy', np.zeros((8, 16), dtype=np.float32))
                (root/'groundTruth'/f'{name}.txt').write_text('a\n'*16)
                (root/f'{name}.bundle').write_text(name+'.txt\n')
            (root/'mapping.txt').write_text('0 a\n1 b\n2 c\n')
            def run(*args):
                result = subprocess.run([sys.executable, '-m', 'midas', *map(str, args)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            run('train', '--data-root', root, '--train-split', root/'train.bundle',
                '--val-split', root/'val.bundle', '--output', root/'run', '--epochs', 1,
                '--input-dim', 8, '--classes', 3, '--layers', 2, '--width', 8, '--device', 'cpu')
            self.assertTrue((root/'run/best_val.pt').is_file())
            # Hide labels: prediction must depend only on features, list and checkpoint.
            (root/'groundTruth').rename(root/'hidden-labels')
            run('predict', '--features', root/'features', '--split', root/'test.bundle',
                '--checkpoint', root/'run/best_val.pt', '--output', root/'pred.json', '--device', 'cpu')
            run('evaluate', '--predictions', root/'pred.json', '--labels', root/'hidden-labels',
                '--mapping', root/'mapping.txt', '--output', root/'metrics.json')
            self.assertIn('F1@50', json.loads((root/'metrics.json').read_text()))


if __name__ == '__main__':
    unittest.main()
