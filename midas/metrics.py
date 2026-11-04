"""Historical evaluator; pooled segment counts, frame accuracy, mean video edit.
Dataset-specific diagnostic fields retained for audit, not universal metrics.
"""
import numpy as np

def seg(seq):
    out = []
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            out.append((seq[start], start, i))
            start = i
    return out


def edit(pred, gt):
    p = [x[0] for x in seg(pred)]
    y = [x[0] for x in seg(gt)]
    d = np.arange(len(y) + 1)
    for a in p:
        nd = np.empty(len(y) + 1, dtype=int)
        nd[0] = d[0] + 1
        for (j, b) in enumerate(y, 1):
            nd[j] = min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (a != b))
        d = nd
    return 100 * (1 - d[-1] / max(len(p), len(y)))


def f1_counts(pred, gt, thr):
    (p, y) = (seg(pred), seg(gt))
    used = set()
    tp = 0
    for (label, s, e) in p:
        cand = [(max(0, min(e, b) - max(s, a)) / (max(e, b) - min(s, a)), i) for (i, (c, a, b)) in enumerate(y) if c == label and i not in used]
        if cand and max(cand)[0] >= thr:
            used.add(max(cand)[1])
            tp += 1
    return (tp, len(p) - tp, len(y) - tp)


def metrics(preds, gts):
    total = sum(map(len, gts))
    result = {'Acc': 100 * sum((sum((a == b for (a, b) in zip(p, g))) for (p, g) in zip(preds, gts))) / total, 'Edit': np.mean([edit(p, g) for (p, g) in zip(preds, gts)])}
    for t in (0.1, 0.25, 0.5):
        counts = np.asarray([f1_counts(p, g, t) for (p, g) in zip(preds, gts)]).sum(axis=0)
        (tp, fp, fn) = counts
        result[f'F1@{int(100 * t)}'] = 100 * (2 * tp / max(1, 2 * tp + fp + fn))
    hits = count = short_hits = short_count = 0
    ratios = []
    for (p, g) in zip(preds, gts):
        hits += sum((a == b == 4 for (a, b) in zip(p, g)))
        count += sum((b == 4 for b in g))
        for (label, s, e) in seg(g):
            if e - s <= 180:
                short_count += e - s
                short_hits += sum((a == label for a in p[s:e]))
        ratios.append(len(seg(p)) / max(1, len(seg(g))))
    result.update({'SpecimenRemoval_recall': 100 * hits / max(1, count), 'short_phase_recall': 100 * short_hits / max(1, short_count), 'segment_count_ratio': float(np.mean(ratios))})
    return result
