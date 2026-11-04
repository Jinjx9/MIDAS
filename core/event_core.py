def align_exact(gt,pred,gt_hz,pred_hz):
    if gt_hz<=0 or pred_hz!=gt_hz or len(gt)!=len(pred) or not len(gt):
        raise ValueError('Rate/length mismatch or empty sequence; explicit timeline reconstruction required')
    return gt,pred

def analyse(gt,pred,hz):
    align_exact(gt,pred,hz,hz)
    events=[];other=[];repaired=list(pred)
    for gi,(label,gs,ge) in enumerate(segments(gt)):
        i=gs
        while i<ge:
            if pred[i]==label:i+=1;continue
            start=i
            while i<ge and pred[i]!=label:i+=1
            end=i;wrong=segments(pred[start:end])
            row=dict(gt_segment_index=gi,gt_class=int(label),gt_start=gs,gt_end=ge,
                     start=start,end=end,duration_samples=end-start,duration_seconds=(end-start)/hz,
                     wrong_classes=[int(c) for c,_,_ in wrong],wrong_segment_count=len(wrong))
            if start>gs and end<ge and pred[start-1]==label and pred[end]==label:
                row['extra_switches']=len(wrong)+1
                events.append(row);repaired[start:end]=[label]*(end-start)
            else:
                row['reason']='entire_gt_segment_error' if start==gs and end==ge else 'touch_gt_left' if start==gs else 'touch_gt_right' if end==ge else 'not_correctly_enclosed'
                row['single_wrong_class']=len(wrong)==1
                other.append(row)
    assert len(segments(pred))-len(segments(repaired))==sum(e['extra_switches'] for e in events)
    assert sum(e['duration_samples'] for e in events)+sum(e['duration_samples'] for e in other)==sum(a!=b for a,b in zip(gt,pred))
    return events,other,repaired

def segments(seq):
    if not len(seq):return []
    output=[];start=0
    for i in range(1,len(seq)+1):
        if i==len(seq) or seq[i]!=seq[start]:
            output.append((seq[start],start,i));start=i
    return output
