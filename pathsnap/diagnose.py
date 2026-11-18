"""Post-selection diagnostics; never changes predictions or tuning parameters."""
import argparse
import json
import numpy as np
from .train import record, positive, save, RESULTS
from .data import ROOT,read_trip,Network
from .features import FEATURE_NAMES
from .api import assets
from .decoder import Decoder

def main():
    parser=argparse.ArgumentParser();parser.add_argument('trips',nargs='*',default=['00000077','00000065','00000056']);args=parser.parse_args()
    model,config=assets();columns=[FEATURE_NAMES.index(name) for name in config['feature_names']];rows=[]
    for name in args.trips:
        if len(name)!=8 or not name.isdigit():parser.error('Expected an eight-digit trip ID')
        r=record(ROOT/'student_test'/name);trip=read_trip(r['folder']);net=Network(trip);k=config['k'];ids=r['ids'][:,:k]
        p=positive(model,r['x'][:,:k][:,:,columns]);raw=ids[np.arange(len(ids)),p.argmax(1)]
        decoder=Decoder(net,**config['decoder']);predicted,resets=decoder.decode(trip.track,ids,r['fractions'][:,:k],p/p.sum(1,keepdims=True))
        wrong=raw!=r['labels'];distance,_=net.distances(net.project(trip.track[:,:2]),r['labels'])
        rows.append(dict(trip=name,observations=len(ids),classifier_accuracy=float(np.mean(~wrong)),hmm_accuracy=float(np.mean(predicted==r['labels'])),
                         alignment_confident_coverage=float(r['confident'].mean()),candidate_miss_rate=float(np.mean(~(ids==r['labels'][:,None]).any(1))),
                         fraction_of_classifier_errors_on_uncertain_labels=float(np.mean(~r['confident'][wrong])) if wrong.any() else 0.,
                         median_distance_to_aligned_reference_m=float(np.median(distance)),p95_distance_to_aligned_reference_m=float(np.quantile(distance,.95)),
                         hmm_corrected_observations=int(np.sum(wrong&(predicted==r['labels']))),hmm_worsened_observations=int(np.sum(~wrong&(predicted!=r['labels']))),
                         alignment_resets=len(r['alignment_resets']),decoder_resets=len(resets)))
    save(RESULTS/'weak_trip_diagnostics.json',dict(label_warning='Inferred ordered-route proxies, not verified observation ground truth',used_for_tuning=False,trips=rows))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
