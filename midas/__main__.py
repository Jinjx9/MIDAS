"""Full-video sequential MIDAS training, label-free prediction and evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .model import MIDAS, read_checkpoint
from .metrics import metrics
from .evaluation import evaluate_segmentation
from .official_evaluation import evaluate_official_asformer


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def apply_inference_precision(settings):
    if 'allow_tf32' in settings:
        torch.backends.cuda.matmul.allow_tf32 = bool(settings['allow_tf32'])
        torch.backends.cudnn.allow_tf32 = bool(settings['allow_tf32'])


def videos(path):
    names = [Path(line.strip()).stem for line in Path(path).read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError('Split must contain unique video identifiers')
    return names


def mapping(path):
    pairs = [line.split(maxsplit=1) for line in Path(path).read_text().splitlines() if line.strip()]
    result = {name: int(index) for index, name in pairs}
    if len(result) != len(pairs) or sorted(result.values()) != list(range(len(result))):
        raise ValueError('Mapping must have unique names and contiguous class IDs starting at zero')
    return result


def labels(root, name, classes):
    return [classes[line.strip()] for line in (Path(root)/f'{name}.txt').read_text().splitlines() if line.strip()]


def features(root, name, device):
    x = np.load(Path(root)/f'{name}.npy', allow_pickle=False)
    if x.ndim != 2:
        raise ValueError('Feature files must be (channels, time), not (time, channels)')
    return torch.from_numpy(x.astype(np.float32))[None].to(device)


@torch.no_grad()
def predict(model, names, root, device):
    model.eval()
    return {name: model(features(root, name, device))[-1, 0].argmax(0).cpu().tolist() for name in names}


def score(pred, root, classes, background=None, metric_protocol='half_open'):
    if not pred:
        raise ValueError('No predictions')
    gt = [labels(root, name, classes) for name in pred]
    for (name, p), g in zip(pred.items(), gt):
        if not g or len(p) != len(g) or any(v not in classes.values() for v in p):
            raise ValueError(f'Invalid labels or time alignment: {name}')
    # The other historical diagnostic fields are dataset-specific, not headline metrics.
    evaluator = {'official_asformer':evaluate_official_asformer,
                 'half_open':evaluate_segmentation}[metric_protocol]
    return evaluator(list(pred.values()), gt, len(classes), background)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    root = args.data_root
    classes = mapping(root/'mapping.txt')
    if len(classes) != args.classes:
        raise ValueError('Number of classes differs from mapping')
    train_names, val_names = videos(args.train_split), videos(args.val_split)
    if set(train_names) & set(val_names):
        raise ValueError('Training and validation splits overlap')
    out = args.output
    # Never append a new run to an existing experiment or silently overwrite it.
    out.mkdir(parents=True, exist_ok=False)
    config = dict(input_dim=args.input_dim, num_classes=args.classes, num_layers=args.layers,
                  num_f_maps=args.width, lam=args.lam, mid=not args.no_mid, backend=args.backend,
                  attention_impl=args.attention_impl)
    model = MIDAS(**config).to(args.device)
    if args.parent:
        model.load_checkpoint(args.parent)
    save_json(out/'config.json', {'model': model.config, 'arguments': {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}, 'torch': torch.__version__, 'numpy': np.__version__})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=3)
    best, best_epoch = -1., 0
    for epoch in range(1, args.epochs+1):
        started = time.time()
        model.train()
        total = 0.
        # torch sampler matches historical one-video DataLoader random ordering.
        if args.shuffle == 'python':
            order = list(train_names)
            random.shuffle(order)
            loader = [(name,) for name in order]
        else:
            loader = torch.utils.data.DataLoader(train_names, batch_size=1, shuffle=True)
        for (name,) in loader:
            x = features(args.features or root/'features', name, args.device)
            y = torch.tensor(labels(root/'groundTruth', name, classes), device=args.device)[None]
            if x.shape[-1] != y.shape[-1]:
                raise ValueError(f'Feature/label length mismatch: {name}')
            optimizer.zero_grad()
            loss = model.loss(model(x), y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach())
        scheduler.step(total/len(train_names))
        val = score(predict(model, val_names, args.features or root/'features', args.device),
                    root/'groundTruth',classes,args.background,args.metric_protocol)
        record = dict(epoch=epoch, seconds=time.time()-started, train_loss=total/len(train_names), val=val)
        with (out/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        state = dict(backbone=model.backbone.state_dict(), config=model.config, epoch=epoch, val=val)
        torch.save(state, out/'last.pt')
        if val['F1@50'] > best:
            best, best_epoch = val['F1@50'], epoch
            torch.save(state, out/'best_val.pt')
        print(json.dumps(record), flush=True)
        if args.patience > 0 and epoch-best_epoch >= args.patience:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train')
    for name in ('data-root', 'train-split', 'val-split', 'output'):
        tr.add_argument('--'+name, type=Path, required=True)
    tr.add_argument('--parent', type=Path, help='Trusted parent ASFormer backbone checkpoint; omitted means scratch')
    for name, default in [('epochs',60), ('patience',15), ('seed',20260303), ('input-dim',2048), ('classes',7), ('layers',10), ('width',64)]:
        tr.add_argument('--'+name, type=int, default=default)
    tr.add_argument('--lam', type=float, default=1e-4)
    tr.add_argument('--lr', type=float, default=1e-4)
    tr.add_argument('--no-mid', action='store_true')
    tr.add_argument('--backend', choices=['sequential', 'historical_fast'], default='sequential')
    tr.add_argument('--attention-impl',choices=['native','unfold'],default='native')
    tr.add_argument('--features',type=Path,help='Original feature directory, separate from prepared labels')
    tr.add_argument('--grad-clip',type=float,default=10.)
    tr.add_argument('--shuffle',choices=['torch','python'],default='torch')
    tr.add_argument('--deterministic',action='store_true')
    tr.add_argument('--background',type=int)
    tr.add_argument('--metric-protocol',choices=['half_open','official_asformer'],default='half_open')
    pr = sub.add_parser('predict')
    for name in ('features', 'split', 'checkpoint', 'output'):
        pr.add_argument('--'+name, type=Path, required=True)
    pr.add_argument('--lam', type=float, help='Required for historical checkpoints lacking config')
    pr.add_argument('--input-dim', type=int, default=2048)
    pr.add_argument('--classes', type=int, default=7)
    pr.add_argument('--backend', choices=['sequential', 'historical_fast'])
    pr.add_argument('--recipe',type=Path,help='Verified model configuration and checkpoint hash')
    pr.add_argument('--attention-impl',choices=['native','unfold'])
    pr.add_argument('--no-mid',action='store_true')
    ev = sub.add_parser('evaluate')
    for name in ('predictions', 'labels', 'mapping', 'output'):
        ev.add_argument('--'+name, type=Path, required=True)
    ev.add_argument('--background', type=int)
    ev.add_argument('--metric-protocol',choices=['half_open','official_asformer'],default='half_open')
    for parser in (tr, pr):
        parser.add_argument('--device', default='cpu')
        parser.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    if hasattr(args, 'threads'):
        torch.set_num_threads(args.threads)
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        state = read_checkpoint(args.checkpoint)
        config = state.get('config')
        if args.recipe:
            recipe = json.loads(args.recipe.read_text())
            apply_inference_precision(recipe.get('runtime',{}))
            actual = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            if actual != recipe['checkpoint_sha256']:
                ap.error('Checkpoint SHA256 differs from the recipe')
            if config is not None and config != recipe['model']:
                ap.error('Checkpoint configuration differs from the recipe')
            config = recipe['model']
        if config is None:
            if args.lam is None:
                ap.error('Historical checkpoint requires explicit --lam (smoothing metadata is not stored in weights)')
            if args.backend is None:
                ap.error('Historical checkpoint requires explicit --backend')
            config = dict(input_dim=args.input_dim,num_classes=args.classes,lam=args.lam,backend=args.backend,
                          mid=not args.no_mid,attention_impl=args.attention_impl or 'native')
        elif args.lam is not None and args.lam != config['lam']:
            ap.error('--lam differs from the checkpoint configuration')
        if args.backend is not None and args.backend != config.get('backend', 'sequential'):
            ap.error('--backend differs from checkpoint configuration')
        if args.attention_impl is not None and args.attention_impl != config.get('attention_impl','native'):
            ap.error('--attention-impl differs from checkpoint configuration')
        if args.no_mid and config.get('mid',True):
            ap.error('--no-mid differs from checkpoint configuration')
        model = MIDAS(**config).to(args.device)
        model.load_checkpoint(args.checkpoint)
        save_json(args.output, predict(model, videos(args.split), args.features, args.device))
    else:
        predictions = json.loads(args.predictions.read_text())
        save_json(args.output, score(predictions,args.labels,mapping(args.mapping),args.background,args.metric_protocol))


if __name__ == '__main__':
    main()
