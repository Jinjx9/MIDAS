"""1-fps localization inputs, interval readout and dataset-level detection AP.

New portable implementation; not a recovered historical SurgVU evaluator.
Intervals use half-open seconds. AP uses globally ranked detections per class,
one-to-one same-video matching and the all-points precision envelope.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch


def safe_name(name):
    if not isinstance(name,str) or not name or name in ('.','..') or any(c in name for c in '/\\:'):
        raise ValueError('Unsafe video identifier')
    return name


def segment(value):
    if len(value)!=2 or not all(math.isfinite(float(x)) for x in value):
        raise ValueError('Invalid temporal segment')
    a,b=map(float,value)
    if a<0 or b<=a:
        raise ValueError('Segments must have nonnegative start and positive length')
    return a,b


def read_annotations(path, classes, overlap_policy='reject'):
    if overlap_policy not in ('reject','last_annotation_wins'):
        raise ValueError('Unknown overlap policy')
    if not classes or len(set(classes))!=len(classes):
        raise ValueError('Foreground classes must be nonempty and unique')
    db=json.loads(Path(path).read_text(encoding='utf-8'))['database']
    if not isinstance(db,dict) or not db:
        raise ValueError('Empty annotation database')
    for name,r in db.items():
        safe_name(name)
        if r.get('subset') not in ('train','val','test'):
            raise ValueError('Subset must be train, val or test')
        if not math.isfinite(float(r['duration'])) or r['duration']<2 or r.get('fps',1)!=1:
            raise ValueError('Expected finite duration >=2 and 1-fps annotations')
        occupied=[]
        for a in r.get('annotations',[]):
            s,e=segment(a['segment'])
            if a['label'] not in classes or e>r['duration']+1e-6:
                raise ValueError('Unknown annotation class or segment outside duration')
            for ps,pe,label in occupied:
                if overlap_policy=='reject' and a['label']!=label and min(e,pe)>max(s,ps):
                    raise ValueError('Overlapping classes cannot form a single-label target')
            occupied.append((s,e,a['label']))
    if {r['subset'] for r in db.values()}!={'train','val','test'}:
        raise ValueError('All three disjoint video subsets are required')
    return db


def load_features(root,name,input_dim,normalization='video_zscore'):
    safe_name(name)
    x=np.load(Path(root)/(name+'.npy'),allow_pickle=False)
    if x.ndim!=2 or x.shape[1]!=input_dim or x.shape[0]<2:
        raise ValueError(f'{name}: expected (time>=2, {input_dim}) features')
    x=x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f'{name}: nonfinite features')
    if normalization=='video_zscore':
        x=(x-x.mean(0,keepdims=True))/(x.std(0,keepdims=True)+1e-6)
    elif normalization!='none':
        raise ValueError('Unknown feature normalization')
    return torch.from_numpy(np.ascontiguousarray(x.T))[None]


def load_case(root,name,record,classes,input_dim,normalization='video_zscore'):
    x=load_features(root,name,input_dim,normalization)
    t=x.shape[-1]
    # No silent min(T_feature,T_annotation): mismatches require explicit re-preparation.
    if t!=int(record['duration']):
        raise ValueError(f'{name}: feature/annotation length mismatch {t} != {int(record["duration"])}')
    y=torch.full((1,t),len(classes),dtype=torch.long)
    for a in record.get('annotations',[]):
        s,e=segment(a['segment'])
        y[:,int(s):int(e)]=classes.index(a['label'])
    return x,y


def intervals_from_scores(scores,classes,threshold=.5,min_length=3,top_k=100):
    scores=np.asarray(scores)
    if (scores.ndim!=2 or scores.shape[1]!=len(classes)+1 or not np.isfinite(scores).all()
            or (scores<0).any() or (scores>1).any()):
        raise ValueError('Expected finite probabilities (time, foreground classes + background)')
    if not 0<=threshold<=1 or min_length<1 or top_k<1:
        raise ValueError('Invalid interval readout settings')
    result=[]
    for ci,label in enumerate(classes):
        active=scores[:,ci]>threshold
        edges=np.diff(np.r_[False,active,False].astype(int))
        detections=[{'label':label,'segment':[float(s),float(e)],
                     'score':float(scores[s:e,ci].mean())}
                    for s,e in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if e-s>=min_length]
        result.extend(sorted(detections,key=lambda v:-v['score'])[:top_k])
    return result


def evaluate_detections(predictions,ground_truth,classes,thresholds=(.3,.4,.5,.6,.7)):
    if not ground_truth or set(predictions)!=set(ground_truth):
        raise ValueError('Predictions must contain exactly every evaluation video, including empty detections')
    if not classes or len(set(classes))!=len(classes) or not thresholds or any(not 0<float(t)<=1 for t in thresholds):
        raise ValueError('Invalid class list or tIoU thresholds')
    for entries in list(ground_truth.values())+list(predictions.values()):
        for d in entries:
            segment(d['segment'])
            if d['label'] not in classes:
                raise ValueError('Unknown detection class')
    for entries in predictions.values():
        if any(not math.isfinite(float(d['score'])) or not 0<=float(d['score'])<=1 for d in entries):
            raise ValueError('Invalid detection confidence')
    supported=[c for c in classes if any(d['label']==c for v in ground_truth.values() for d in v)]
    if not supported:
        raise ValueError('No foreground ground truth in evaluation subset')
    per_class={}
    for label in supported:
        gt={v:[segment(d['segment']) for d in entries if d['label']==label] for v,entries in ground_truth.items()}
        npos=sum(map(len,gt.values()))
        ranked=sorted([(v,d) for v in sorted(predictions) for d in predictions[v] if d['label']==label],
                      key=lambda item:-float(item[1]['score']))
        aps=[]
        for threshold in thresholds:
            used={v:set() for v in gt};tp=[]
            for v,d in ranked:
                s,e=segment(d['segment']);best=-1;best_iou=-1.
                for j,(gs,ge) in enumerate(gt[v]):
                    if j in used[v]:continue
                    inter=max(0.,min(e,ge)-max(s,gs));iou=inter/((e-s)+(ge-gs)-inter)
                    if iou>=threshold and iou>best_iou:best,best_iou=j,iou
                tp.append(float(best>=0))
                if best>=0:used[v].add(best)
            if not tp:aps.append(0.);continue
            hits=np.cumsum(tp);precision=hits/np.arange(1,len(tp)+1);recall=hits/npos
            r=np.r_[0.,recall,1.];p=np.r_[0.,precision,0.]
            p=np.maximum.accumulate(p[::-1])[::-1]
            changes=np.where(r[1:]!=r[:-1])[0]
            aps.append(float(np.sum((r[changes+1]-r[changes])*p[changes+1]))*100)
        per_class[label]=aps
    means=np.mean(list(per_class.values()),axis=0)
    return {'protocol':'dataset_class_global_interpolated_ap_v1','thresholds':list(thresholds),
            'mAP':means.tolist(),'average_mAP':float(means.mean()),'per_class_AP':per_class,
            'evaluated_classes':supported,'excluded_no_gt_classes':[c for c in classes if c not in supported],
            'num_videos':len(ground_truth)}
