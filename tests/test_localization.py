import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from midas import localization as loc


class LocalizationTests(unittest.TestCase):
    def test_interval_readout_uses_strict_threshold_mean_score_and_score_topk(self):
        scores = np.array([[.9,.1],[.8,.2],[.5,.5],[.95,.05],[.96,.04]])
        detections = loc.intervals_from_scores(scores, ['a'], .5, 2, 1)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['segment'], [3.,5.])
        self.assertAlmostEqual(detections[0]['score'], .955)

    def test_global_ranking_not_mean_video_ap(self):
        gt = {'v1': [{'label':'a','segment':[0,2]}],
              'v2': [{'label':'a','segment':[0,2]}]}
        pred = {'v1':[{'label':'a','segment':[0,2],'score':.8}],
                'v2':[{'label':'a','segment':[4,6],'score':.9},
                      {'label':'a','segment':[0,2],'score':.7}]}
        r = loc.evaluate_detections(pred, gt, ['a'], [.5])
        self.assertAlmostEqual(r['average_mAP'], 100*2/3)
        self.assertEqual(r['evaluated_classes'], ['a'])

    def test_duplicates_cannot_match_twice_and_missing_predictions_are_false_negatives(self):
        gt = {'v':[{'label':'a','segment':[0,2]}, {'label':'a','segment':[4,6]}]}
        p = {'v':[{'label':'a','segment':[0,2],'score':.9},
                  {'label':'a','segment':[0,2],'score':.8}]}
        self.assertEqual(loc.evaluate_detections(p,gt,['a','absent'],[.5])['average_mAP'],50.)
        self.assertEqual(loc.evaluate_detections({'v':[]},gt,['a'],[.5])['average_mAP'],0.)
        with self.assertRaises(ValueError):
            loc.evaluate_detections({},gt,['a'],[.5])

    def test_bad_scores_unknown_classes_and_cases_fail(self):
        gt = {'v':[{'label':'a','segment':[0,2]}]}
        for entry in [{'label':'a','segment':[0,2],'score':float('nan')},
                      {'label':'oops','segment':[0,2],'score':.5},
                      {'label':'a','segment':[2,0],'score':.5}]:
            with self.assertRaises(ValueError):
                loc.evaluate_detections({'v':[entry]},gt,['a'],[.5])
        with self.assertRaises(ValueError):
            loc.evaluate_detections({'other':[]},gt,['a'],[.5])
        with self.assertRaises(ValueError):
            loc.intervals_from_scores(np.full((3,2),np.nan), ['a'])

    def test_data_contract_normalizes_time_channels_and_rejects_silent_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.save(root/'case.npy',np.arange(12,dtype=np.float32).reshape(6,2))
            record={'duration':6.,'fps':1.,'subset':'train',
                    'annotations':[{'label':'a','segment':[1.2,3.8]}]}
            x,y = loc.load_case(root,'case',record,['a'],2)
            self.assertEqual(tuple(x.shape),(1,2,6))
            self.assertEqual(y.tolist(),[[1,0,0,1,1,1]])
            np.testing.assert_allclose(x.mean(-1).numpy(),0,atol=1e-6)
            record['duration']=4
            with self.assertRaises(ValueError):
                loc.load_case(root,'case',record,['a'],2)
            with self.assertRaises(ValueError):
                loc.load_case(root,'../case',record,['a'],2)

    def test_annotations_require_all_splits_known_labels_and_no_conflicting_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json'
            db={s:{'duration':6.,'fps':1.,'subset':s,
                   'annotations':[{'label':'a','segment':[1,3]}]} for s in ['train','val','test']}
            p.write_text(json.dumps({'database':db}))
            self.assertEqual(len(loc.read_annotations(p,['a'])),3)
            db['train']['annotations'].append({'label':'b','segment':[2,4]})
            p.write_text(json.dumps({'database':db}))
            with self.assertRaises(ValueError):loc.read_annotations(p,['a','b'])
            self.assertEqual(len(loc.read_annotations(p,['a','b'],overlap_policy='last_annotation_wins')),3)
            np.save(Path(tmp)/'train.npy',np.ones((6,2),dtype='float32'))
            _,y=loc.load_case(Path(tmp),'train',db['train'],['a','b'],2)
            self.assertEqual(y.tolist(),[[2,0,1,1,2,2]])
            del db['test']
            p.write_text(json.dumps({'database':db}))
            with self.assertRaises(ValueError):loc.read_annotations(p,['a','b'])
