import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class TALCLITests(unittest.TestCase):
    def test_end_to_end_label_free_prediction_and_configured_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);features=root/'features';features.mkdir()
            rng=np.random.default_rng(7)
            db={s:{'duration':16.,'fps':1.,'subset':s,
                   'annotations':[{'label':'a','segment':[2.,12.]}]} for s in ['train','val','test']}
            for name in db:np.save(features/(name+'.npy'),rng.normal(size=(16,8)).astype('float32'))
            ann=root/'annotations.json';ann.write_text(json.dumps({'database':db}))
            config={'schema_version':1,'status':'reconstructed_new_run','classes':['a'],
                    'model':{'input_dim':8,'num_classes':2,'num_layers':2,'num_f_maps':8,
                             'num_decoders':3,'lam':.0001,'mid':True,'backend':'historical_fast','attention_impl':'unfold'},
                    'training':{'seed':7,'epochs':1,'lr':.0001,'weight_decay':.00001,'grad_clip':5},
                    'data':{'normalization':'video_zscore','feature_layout':'time_channels','fps':1},
                    'readout':{'threshold':.5,'min_length':1,'top_k':100},
                    'evaluation':{'thresholds':[.3,.4,.5,.6,.7]}}
            cfg=root/'config.json';cfg.write_text(json.dumps(config))
            def run(*args,good=True):
                r=subprocess.run([sys.executable,'-m','midas.tal',*map(str,args)],capture_output=True,text=True)
                if good:self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                else:self.assertNotEqual(r.returncode,0)
                return r
            run('prepare','--annotations',ann,'--config',cfg,'--output',root/'prepared')
            config['data']['expected_split_counts']={'train':100,'val':25,'test':30}
            cfg.write_text(json.dumps(config))
            run('prepare','--annotations',ann,'--config',cfg,'--output',root/'wrong_split',good=False)
            del config['data']['expected_split_counts']
            cfg.write_text(json.dumps(config))
            run('preflight','--annotations',ann,'--features',features,'--config',cfg,'--output',root/'audit.json')
            self.assertEqual(json.loads((root/'audit.json').read_text())['split_counts'],{'train':1,'val':1,'test':1})
            run('train','--annotations',ann,'--features',features,'--config',cfg,'--output',root/'run')
            self.assertTrue((root/'run/best_val.pt').exists())
            self.assertTrue((root/'run/run_manifest.json').exists())
            ann.rename(root/'hidden_annotations.json')
            run('predict','--features',features,'--split',root/'prepared/test.bundle',
                '--checkpoint',root/'run/best_val.pt','--output',root/'pred.json')
            pred=json.loads((root/'pred.json').read_text())
            self.assertEqual(set(pred['results']),{'test'})
            self.assertEqual(len(pred['checkpoint_sha256']),64)
            run('evaluate','--predictions',root/'pred.json','--annotations',root/'hidden_annotations.json',
                '--subset','test','--output',root/'metric.json')
            self.assertIn('average_mAP',json.loads((root/'metric.json').read_text()))
            run('predict','--features',features,'--split',root/'prepared/test.bundle',
                '--checkpoint',root/'run/best_val.pt','--output',root/'pred.json',good=False)
            record=json.loads((root/'run/metrics.jsonl').read_text())
            self.assertEqual(record['val']['num_videos'],1)
            self.assertNotIn('test',record)
            ann=root/'hidden_annotations.json'
            db['train']['duration']=14.;ann.write_text(json.dumps({'database':db}))
            run('preflight','--annotations',ann,'--features',features,'--config',cfg,'--output',root/'bad.json',good=False)
