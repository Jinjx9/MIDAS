import unittest
from midas.official_evaluation import evaluate_official_asformer


class OfficialMetricTests(unittest.TestCase):
    def test_exact_foreground_sequence(self):
        r = evaluate_official_asformer([[0,0,1,1]], [[0,0,1,1]], 2)
        self.assertEqual(r['F1@50'],100.)
        self.assertEqual(r['Edit'],100.)

    def test_empty_foreground_is_handled(self):
        r = evaluate_official_asformer([[0,0]], [[0,0]], 2, background=0)
        self.assertEqual(r['F1@50'],0.)
        self.assertEqual(r['Edit'],100.)

    def test_alignment_error(self):
        with self.assertRaises(ValueError):
            evaluate_official_asformer([[0]],[[0,0]],2)
