import unittest
from midas.evaluation import evaluate_segmentation


class EvaluationTests(unittest.TestCase):
    def test_one_prediction_cannot_match_two_truth_segments(self):
        result = evaluate_segmentation([[0]*5], [[0,0,1,0,0]], num_classes=2)
        self.assertAlmostEqual(result['F1@10'], 50.)
        self.assertAlmostEqual(result['Acc'], 80.)

    def test_background_excluded_only_from_segment_metrics(self):
        r = evaluate_segmentation([[0,1,0]], [[0,1,0]], num_classes=2, background=0)
        self.assertEqual(r['F1@50'], 100.)
        self.assertEqual(r['Edit'], 100.)
        self.assertEqual(r['Acc'], 100.)

    def test_misaligned_inputs_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_segmentation([[0]], [[0,0]], num_classes=2)

    def test_jaccard_macro_per_video_is_explicit(self):
        r = evaluate_segmentation([[0,0],[1,1]], [[0,1],[1,1]], num_classes=2)
        self.assertEqual(r['Jaccard_video_macro'], 62.5)

    def test_all_background_has_finite_scores(self):
        r = evaluate_segmentation([[0,0]], [[0,0]], num_classes=2, background=0)
        self.assertEqual(r['F1@50'], 0.)
        self.assertEqual(r['Edit'], 100.)
