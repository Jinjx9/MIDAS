"""Explicit pooled one-to-one segment evaluation; all values are percentages."""
import numpy as np


def segments(sequence, background=None):
    result = []
    start = 0
    for end in range(1, len(sequence)+1):
        if end == len(sequence) or sequence[end] != sequence[start]:
            if sequence[start] != background:
                result.append((sequence[start], start, end))
            start = end
    return result


def segment_edit(pred, gt):
    a, b = [v[0] for v in pred], [v[0] for v in gt]
    if not a and not b:
        return 100.
    previous = list(range(len(b)+1))
    for i, label in enumerate(a, 1):
        current = [i]
        for j, target in enumerate(b, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(label != target)))
        previous = current
    return 100.*(1-previous[-1]/max(len(a), len(b)))


def evaluate_segmentation(predictions, ground_truth, num_classes, background=None):
    if not predictions or len(predictions) != len(ground_truth):
        raise ValueError('Predictions and ground truth must contain the same nonzero number of videos')
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    counts = np.zeros((3, 3), dtype=np.int64)
    edits, jaccards = [], []
    for p, g in zip(predictions, ground_truth):
        p, g = np.asarray(p), np.asarray(g)
        if p.ndim != 1 or g.ndim != 1 or len(p) != len(g) or not len(g):
            raise ValueError('Video timelines must be nonempty and exactly aligned')
        if (not np.issubdtype(p.dtype, np.integer) or not np.issubdtype(g.dtype, np.integer)
                or p.min() < 0 or g.min() < 0 or p.max() >= num_classes or g.max() >= num_classes):
            raise ValueError('Class IDs must be integers in the declared mapping')
        local = np.zeros_like(confusion)
        np.add.at(local, (g, p), 1)
        confusion += local
        union = local.sum(0)+local.sum(1)-local.diagonal()
        active = union > 0
        jaccards.append(float((local.diagonal()[active]/union[active]).mean())*100.)
        ps, gs = segments(p, background), segments(g, background)
        edits.append(segment_edit(ps, gs))
        for row, threshold in enumerate((.1, .25, .5)):
            used, tp = set(), 0
            for label, start, end in ps:
                candidates = [(max(0, min(end, b)-max(start, a))/(max(end, b)-min(start, a)), i)
                              for i, (c, a, b) in enumerate(gs) if c == label and i not in used]
                if candidates and max(candidates)[0] >= threshold:
                    used.add(max(candidates)[1])
                    tp += 1
            counts[row] += (tp, len(ps)-tp, len(gs)-tp)
    union = confusion.sum(0)+confusion.sum(1)-confusion.diagonal()
    active = union > 0
    result = {'Acc': 100.*float(confusion.trace()/confusion.sum()), 'Edit': float(np.mean(edits)),
              'Jaccard_video_macro': float(np.mean(jaccards)),
              'Jaccard_global_class_macro': 100.*float((confusion.diagonal()[active]/union[active]).mean())}
    for threshold, (tp, fp, fn) in zip((10,25,50), counts):
        result[f'F1@{threshold}'] = 100.*float(2*tp/max(1,2*tp+fp+fn))
    return result
