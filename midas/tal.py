"""SurgVU localization using the main MIDAS model, with fixed run metadata.

Reconstructed training pipeline: benchmark performance awaits a new run.
No historical result is inferred from this implementation or its smoke tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from .model import MIDAS, read_checkpoint
from .localization import (read_annotations,load_case,load_features,safe_name,
                           intervals_from_scores,evaluate_detections)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def config_from(path):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    if c.get('schema_version')!=1 or c['model']['num_classes']!=len(c['classes'])+1:
        raise ValueError('Config must specify foreground classes plus one background')
    if c['data']['fps']!=1 or c['data']['feature_layout']!='time_channels':
        raise ValueError('This pipeline requires 1-fps time-by-channel inputs')
    if c['training']['epochs']<1 or c['training']['lr']<=0:
        raise ValueError('Invalid training budget')
    if len(c['evaluation']['thresholds'])!=len(set(c['evaluation']['thresholds'])):
        raise ValueError('Duplicate evaluation thresholds')
    return c


def split_names(db,subset):
    return sorted(name for name,r in db.items() if r['subset']==subset)


def annotation_db(path,c):
    db=read_annotations(path,c['classes'],c['data'].get('overlap_policy','reject'))
    counts={s:len(split_names(db,s)) for s in ('train','val','test')}
    if 'expected_split_counts' in c['data'] and counts!=c['data']['expected_split_counts']:
        raise ValueError('Annotation split counts differ from the fixed configuration')
    return db


def prepare(args):
    c=config_from(args.config);db=annotation_db(args.annotations,c)
    args.output.mkdir(parents=True,exist_ok=False)
    for subset in ('train','val','test'):
        (args.output/(subset+'.bundle')).write_text('\n'.join(split_names(db,subset))+'\n',encoding='utf-8')
    write_json(args.output/'data_manifest.json',{'annotations_sha256':sha256(args.annotations),
               'classes':c['classes'],'background_id':len(c['classes']),
               'split_counts':{s:len(split_names(db,s)) for s in ('train','val','test')}})


def inspect_inputs(annotations,features,c,subsets=('train','val','test')):
    db=annotation_db(annotations,c);records=[]
    for name in sorted(db):
        if db[name]['subset'] not in subsets:continue
        x,_=load_case(features,name,db[name],c['classes'],c['model']['input_dim'],c['data']['normalization'])
        path=features/(name+'.npy')
        records.append({'video':name,'subset':db[name]['subset'],'steps':x.shape[-1],
                        'bytes':path.stat().st_size,'sha256':sha256(path)})
    return db,{'annotations_sha256':sha256(annotations),'features':records,
              'split_counts':{s:len(split_names(db,s)) for s in subsets},
              'total_steps':sum(r['steps'] for r in records)}


@torch.no_grad()
def predict_names(model,names,features,c,device):
    model.eval();results={}
    for name in names:
        x=load_features(features,name,c['model']['input_dim'],c['data']['normalization']).to(device)
        scores=model(x)[-1,0].softmax(0).T.cpu().numpy()
        results[name]=intervals_from_scores(scores,c['classes'],**c['readout'])
    return results


def evaluate(results,db,names,c):
    ground_truth={name:db[name].get('annotations',[]) for name in names}
    return evaluate_detections(results,ground_truth,c['classes'],c['evaluation']['thresholds'])


def set_runtime(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def train(args):
    c=config_from(args.config);settings=c['training'];set_runtime(settings['seed'])
    # Validate train/val features only; no test prediction or test metric during selection.
    db,inputs=inspect_inputs(args.annotations,args.features,c,('train','val'))
    train_ids=split_names(db,'train');val_ids=split_names(db,'val')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'config.json',c)
    source=Path(__file__).parent
    manifest={'pipeline_status':'reconstructed_new_run','config_sha256':sha256(args.config),
              'inputs':inputs,'python':platform.python_version(),'torch':str(torch.__version__),
              'numpy':np.__version__,'device':str(args.device),
              'source_sha256':{p.name:sha256(p) for p in [source/'tal.py',source/'localization.py',
                  source/'model.py',source/'asformer.py',source.parent/'core/fast_smoothing.py',
                  source.parent/'core/smoothing.py',source.parent/'core/sliding_unfold.py']},
              'allow_tf32':False,'selection':'strictly highest validation average_mAP; earlier ties',
              'test_access':'no test inference during training','initialization':'scratch'}
    write_json(args.output/'run_manifest.json',manifest)
    model=MIDAS(**c['model']).to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    best=-1.;best_epoch=0
    for epoch in range(1,settings['epochs']+1):
        start=time.monotonic();model.train();order=list(train_ids);random.shuffle(order);total=0.
        for index,name in enumerate(order,1):
            x,y=load_case(args.features,name,db[name],c['classes'],c['model']['input_dim'],c['data']['normalization'])
            x=x.to(args.device);y=y.to(args.device);opt.zero_grad(set_to_none=True)
            loss=model.loss(model(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if settings['grad_clip']>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip'],error_if_nonfinite=True)
            opt.step();total+=float(loss.detach())
            if index==1 or index%10==0 or index==len(order):
                print(json.dumps({'epoch':epoch,'video':index,'videos':len(order),
                                  'elapsed_seconds':time.monotonic()-start,'loss':float(loss.detach())}),flush=True)
        val=evaluate(predict_names(model,val_ids,args.features,c,args.device),db,val_ids,c)
        record={'epoch':epoch,'seconds':time.monotonic()-start,'train_loss':total/len(order),'val':val}
        state={'backbone':model.backbone.state_dict(),'config':model.config,'tal_config':c,
               'epoch':epoch,'val':val,'run_manifest_sha256':sha256(args.output/'run_manifest.json')}
        torch.save(state,args.output/'last.pt')
        if val['average_mAP']>best:
            best=val['average_mAP'];best_epoch=epoch;torch.save(state,args.output/'best_val.pt')
        with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    write_json(args.output/'selection.json',{'best_epoch':best_epoch,'validation_average_mAP':best,
               'checkpoint_sha256':sha256(args.output/'best_val.pt'),'test_evaluated':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');check=sub.add_parser('preflight');tr=sub.add_parser('train')
    for parser in (prep,check,tr):
        parser.add_argument('--annotations',type=Path,required=True)
        parser.add_argument('--config',type=Path,required=True)
        parser.add_argument('--output',type=Path,required=True)
    for parser in (check,tr):parser.add_argument('--features',type=Path,required=True)
    pred=sub.add_parser('predict')
    for name in ('features','split','checkpoint','output'):pred.add_argument('--'+name,type=Path,required=True)
    ev=sub.add_parser('evaluate')
    for name in ('predictions','annotations','output'):ev.add_argument('--'+name,type=Path,required=True)
    ev.add_argument('--subset',choices=['val','test'],required=True)
    for parser in (tr,pred):
        parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    if args.output.exists():p.error('Output already exists; choose a fresh path')
    if hasattr(args,'threads'):
        if args.threads<1:p.error('threads must be positive')
        torch.set_num_threads(args.threads)
    if args.command=='prepare':prepare(args)
    elif args.command=='preflight':
        c=config_from(args.config);_,audit=inspect_inputs(args.annotations,args.features,c)
        write_json(args.output,audit)
        print(json.dumps({k:v for k,v in audit.items() if k!='features'}),flush=True)
    elif args.command=='train':train(args)
    elif args.command=='predict':
        state=read_checkpoint(args.checkpoint)
        c=state['tal_config']
        if state['config']!=c['model']:raise ValueError('Model and TAL checkpoint configurations differ')
        set_runtime(c['training']['seed'])
        model=MIDAS(**c['model']).to(args.device);model.load_checkpoint(args.checkpoint)
        names=[safe_name(v.strip()) for v in args.split.read_text(encoding='utf-8').splitlines() if v.strip()]
        if not names or len(names)!=len(set(names)):raise ValueError('Empty or duplicate split')
        results=predict_names(model,names,args.features,c,args.device)
        write_json(args.output,{'results':results,'tal_config':c,'checkpoint_sha256':sha256(args.checkpoint),
                               'selected_epoch':state['epoch'],'split_sha256':sha256(args.split)})
    else:
        predictions=json.loads(args.predictions.read_text(encoding='utf-8'));c=predictions['tal_config']
        db=annotation_db(args.annotations,c);names=split_names(db,args.subset)
        result=evaluate(predictions['results'],db,names,c)
        result.update({'subset':args.subset,'checkpoint_sha256':predictions['checkpoint_sha256'],
                       'annotations_sha256':sha256(args.annotations),'selected_epoch':predictions['selected_epoch']})
        write_json(args.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
