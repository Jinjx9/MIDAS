from pathlib import Path
import tempfile
import unittest
from midas.prepare import prepare_salads_labels
from midas.prepare import prepare_c80_labels
import numpy as np


class PrepareTests(unittest.TestCase):
    def test_c80_drops_only_declared_extra_terminal_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'raw'; (root/'groundTruth').mkdir(parents=True); (root/'features').mkdir()
            (root/'mapping.txt').write_text('0 a\n1 b\n')
            (root/'groundTruth/v.txt').write_text('a\na\nb\n')
            np.save(root/'features/v.npy',np.zeros((2048,2),dtype=np.float32))
            with self.assertRaises(ValueError):
                prepare_c80_labels(root,root/'features',Path(temp)/'strict')
            out = Path(temp)/'prepared'
            prepare_c80_labels(root,root/'features',out,drop_one_terminal=True)
            self.assertEqual((out/'groundTruth/v.txt').read_text(),'a\na\n')
            self.assertEqual((root/'groundTruth/v.txt').read_text(),'a\na\nb\n')

    def test_merge_background_preserves_source_and_timeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'raw'; root.mkdir()
            (root/'groundTruth').mkdir()
            (root/'mapping.txt').write_text('\n'.join(f'{i} c{i}' for i in range(19))+'\n')
            text = 'c0\nc17\nc18\nc16\n'
            (root/'groundTruth/v.txt').write_text(text)
            out = Path(temp)/'prepared'
            prepare_salads_labels(root,out)
            self.assertEqual((out/'groundTruth/v.txt').read_text(),'c0\nbackground\nbackground\nc16\n')
            self.assertEqual((root/'groundTruth/v.txt').read_text(),text)
            self.assertEqual(len((out/'mapping.txt').read_text().splitlines()),18)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                prepare_salads_labels(Path(temp),Path(temp))
