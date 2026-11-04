"""50Salads historical official-ASFormer metric protocol, kept explicitly distinct."""
import numpy as np
from core.asformer_metrics import get_labels_start_end_time, levenstein, f_score
from .evaluation import evaluate_segmentation


def evaluate_official_asformer(predictions, ground_truth, num_classes, background=None):
    result = evaluate_segmentation(predictions,ground_truth,num_classes,background)
    counts = np.zeros((3,3))
    edits = []
    bg = [] if background is None else [str(background)]
    for pred, gt in zip(predictions,ground_truth):
        pp, gg = [str(int(v)) for v in pred], [str(int(v)) for v in gt]
        ps, gs = get_labels_start_end_time(pp,bg)[0], get_labels_start_end_time(gg,bg)[0]
        edits.append(levenstein(ps,gs,True) if ps or gs else 100.)
        for i, th in enumerate((.1,.25,.5)):
            counts[i] += f_score(pp,gg,th,bg) if ps and gs else [0,len(ps),len(gs)]
    result['Edit'] = float(np.mean(edits))
    for th,(tp,fp,fn) in zip((10,25,50),counts):
        result[f'F1@{th}'] = float(200*tp/max(1,2*tp+fp+fn))
    return result
