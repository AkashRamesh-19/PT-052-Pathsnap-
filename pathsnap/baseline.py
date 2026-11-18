"""Measure geometry-only proxy accuracy on permitted development trips."""
import json
import numpy as np
from .data import ROOT, Network, read_trip, read_reference
from .features import proxy_labels


def main():
    rows=[]
    for folder in sorted((ROOT/'student_test').iterdir()):
        if not folder.is_dir(): continue
        trip=read_trip(folder); net=Network(trip)
        ids,distances,_=net.candidates(trip.track,k=6)
        reference=read_reference(folder,len(trip.arcs))
        target=proxy_labels(trip,net,reference)
        target_distance,_=net.distances(net.project(trip.track[:,:2]),target)
        wrong=ids[:,0]!=target
        row=dict(trip=trip.id,observations=len(ids),nearest_top1_proxy_accuracy=float(np.mean(~wrong)),top2_proxy_coverage=float(np.mean((ids[:,:2]==target[:,None]).any(1))),top6_proxy_coverage=float(np.mean((ids==target[:,None]).any(1))),wrong_but_within_1cm_distance_of_proxy=float(np.mean(wrong & (np.abs(target_distance-distances[:,0])<=.01))))
        rows.append(row);print(json.dumps(row),flush=True)
    aggregate={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ('trip','observations')}
    result=dict(partition='student_test',label_method='ordered-reference alignment proxy; not independently verified observation labels',aggregation='unweighted mean across trips',model_trained=False,evaluator_accessed=False,aggregate=aggregate,trips=rows)
    (ROOT/'results/geometric_baseline.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Mean across trips:',json.dumps(aggregate),flush=True)


if __name__=='__main__': main()
