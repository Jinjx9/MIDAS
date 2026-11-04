"""Prepare labels from authorized raw 50Salads data; do not redistribute datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def prepare_c80_labels(raw_root,features,output,drop_one_terminal=False):
    raw_root,features,output = Path(raw_root),Path(features),Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    transformed,records = {},[]
    known = {line.split(maxsplit=1)[1] for line in (raw_root/'mapping.txt').read_text().splitlines() if line.strip()}
    for source in sorted((raw_root/'groundTruth').glob('*.txt')):
        feat = features/(source.stem+'.npy')
        x = np.load(feat,mmap_mode='r',allow_pickle=False)
        gt = source.read_text().splitlines()
        if x.ndim != 2 or x.shape[0] != 2048 or not gt or any(label not in known for label in gt):
            raise ValueError('Invalid feature/label input: '+source.stem)
        original = len(gt)
        if len(gt) != x.shape[1]:
            if not drop_one_terminal or len(gt) != x.shape[1]+1:
                raise ValueError('Timeline mismatch (no implicit truncation): '+source.stem)
            gt = gt[:-1]
        transformed[source.name] = gt
        records.append(dict(video=source.stem,original_label_length=original,evaluated_length=len(gt)))
    if not transformed:
        raise ValueError('No labels found')
    (output/'groundTruth').mkdir(parents=True)
    (output/'mapping.txt').write_bytes((raw_root/'mapping.txt').read_bytes())
    for name,gt in transformed.items():
        (output/'groundTruth'/name).write_text('\n'.join(gt)+'\n')
    (output/'preparation.json').write_text(json.dumps(dict(drop_one_terminal=drop_one_terminal,records=records),indent=2))
    return records


def prepare_salads_labels(raw_root, output):
    raw_root, output = Path(raw_root), Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    pairs = [line.split(maxsplit=1) for line in (raw_root/'mapping.txt').read_text().splitlines() if line.strip()]
    names = {int(i):name for i,name in pairs}
    if sorted(names) != list(range(19)) or len(pairs) != 19 or len(set(names.values())) != 19:
        raise ValueError('Expected the original 19-class mapping')
    inverse = {name:i for i,name in names.items()}
    inputs = list((raw_root/'groundTruth').glob('*.txt'))
    if not inputs:
        raise ValueError('No ground-truth files found')
    # Validate all labels before creating an output directory.
    transformed = {}
    for path in inputs:
        tokens = path.read_text().splitlines()
        if not tokens or any(line not in inverse for line in tokens):
            raise ValueError('Empty or unknown label in '+path.name)
        transformed[path.name] = ['background' if inverse[line] >= 17 else line for line in tokens]
    (output/'groundTruth').mkdir(parents=True)
    (output/'mapping.txt').write_text(''.join(f'{i} {names[i]}\n' for i in range(17))+'17 background\n')
    records = []
    for path in inputs:
        (output/'groundTruth'/path.name).write_text('\n'.join(transformed[path.name])+'\n')
        records.append(dict(video=path.stem,frames=len(transformed[path.name]),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (output/'preparation.json').write_text(json.dumps(dict(policy='raw IDs17/18 -> background17; no temporal resampling',
        videos=len(records),records=records,features='Use original full-resolution features separately; no copies or transposition here'),indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',choices=['50salads','cholec80','autolaparo'],default='50salads')
    parser.add_argument('--features',type=Path)
    parser.add_argument('--drop-one-terminal',action='store_true',help='Explicit historical C80 terminal-sample policy')
    args = parser.parse_args()
    if args.dataset == '50salads':
        records = prepare_salads_labels(args.raw_root,args.output)
    else:
        if args.features is None or (args.dataset == 'autolaparo' and args.drop_one_terminal):
            parser.error('Supply --features; AutoLaparo does not permit terminal truncation')
        records = prepare_c80_labels(args.raw_root,args.features,args.output,args.drop_one_terminal)
    print(json.dumps({'prepared_videos':len(records)}))


if __name__ == '__main__':
    main()
