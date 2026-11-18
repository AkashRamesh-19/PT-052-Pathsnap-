"""Evidence-only audit: fresh inference, no training, tuning or evaluator access."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings
import joblib
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from .api import assets
from .data import ROOT,read_trip,read_reference,Network
from .features import prepare,FEATURE_NAMES
from .alignment import align_reference
from .decoder import Decoder
from .reconstruct import reconstruct_sections
from .train import RESULTS,SEED,metrics,average,lattice_continuity,parameters,initial_point,save,record

OUT=ROOT/'results/final_verification'

def replay_bayesian(config):
    history=json.loads((RESULTS/'bayesian_trials.json').read_text());rng=np.random.default_rng(SEED);zs=[];scores=[]
    for i,trial in enumerate(history):
        if i==0:z=np.array(initial_point(config['model']))
        elif i<3:z=rng.random(len(zs[0]))
        else:
            gp=GaussianProcessRegressor(kernel=Matern(length_scale=np.ones(len(zs[0]))*.4,length_scale_bounds=(.05,4),nu=2.5),alpha=1e-5,normalize_y=True,random_state=SEED)
            with warnings.catch_warnings():warnings.simplefilter('ignore');gp.fit(np.array(zs),scores)
            pool=rng.random((512,len(zs[0])));mu,sd=gp.predict(pool,return_std=True);improvement=mu-max(scores)-.0001
            z=pool[np.argmax(improvement*norm.cdf(improvement/np.maximum(sd,1e-12))+sd*norm.pdf(improvement/np.maximum(sd,1e-12)))]
        np.testing.assert_allclose(z,trial['unit_parameters'],atol=1e-12)
        zs.append(z);scores.append(trial['objective'])
    winner=max(history,key=lambda t:t['objective']);expected=parameters(config['model'],winner['unit_parameters']).get_params()
    for key in ('learning_rate','max_leaf_nodes','l2_regularization','max_iter','early_stopping'):
        assert expected[key]==config['parameters'][key],key
    assert abs(config['cv_top1_accuracy']-winner['objective'])<1e-12
    return dict(status='PASS',trials=len(history),gp_expected_improvement_proposals=len(history)-3,selected_trial=winner['trial'],best_cv_top1=winner['objective'],parameters_replay_matches=True)

def plots():
    # Missing current plots are generated exclusively from recorded current results.
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data_cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cv=json.loads((RESULTS/'model_comparison.json').read_text());student=json.loads((RESULTS/'student_model_comparison.json').read_text())
    byname={r['model']:r for r in student};names=[r['model'] for r in cv]
    for filename,title,series in [
        ('cv_macro_f1.png','Training-trip grouped validation: observation Macro F1',{'CV Macro F1':[r['cv_macro_f1'] for r in cv]}),
        ('student_accuracy_f1.png','Student-test trip means: aligned-proxy metrics',{'Top-1':[byname[n]['top1_accuracy'] for n in names],'Macro F1':[byname[n]['macro_f1'] for n in names]}),
        ('student_log_loss.png','Student-test mean observation Log Loss',{'Log Loss':[byname[n]['log_loss'] for n in names]})]:
        fig,ax=plt.subplots(figsize=(9,5));positions=np.arange(len(names));width=.75/len(series)
        for j,(label,values) in enumerate(series.items()):ax.bar(positions+j*width,values,width,label=label)
        ax.set_xticks(positions+width*(len(series)-1)/2,names,rotation=18,ha='right');ax.set_title(title);ax.legend();fig.tight_layout();fig.savefig(RESULTS/filename,dpi=130);plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    model,config=assets();reload_model=joblib.load(ROOT/'artifacts/model.joblib')
    assert model.n_features_in_==len(config['feature_names'])>=12
    assert model.early_stopping is False
    used=set()
    for stage in model._predictors:
        for tree in stage:used.update(tree.nodes['feature_idx'][tree.nodes['is_leaf']==0].tolist())
    assert len(used)>=12
    model_fingerprint=hashlib.sha256((ROOT/'artifacts/model.joblib').read_bytes()).hexdigest()
    trainfolders=sorted(p for p in (ROOT/'train').iterdir() if p.is_dir());studentfolders=sorted(p for p in (ROOT/'student_test').iterdir() if p.is_dir())
    trainids={p.name for p in trainfolders};studentids={p.name for p in studentfolders}
    assert len(trainids)==80 and len(studentids)==10 and not trainids&studentids
    counts=[]
    for folder in trainfolders:
        t=read_trip(folder);reference=read_reference(folder,len(t.arcs))
        counts.append(dict(trip=t.id,observations=len(t.track),nodes=len(t.nodes),arcs=len(t.arcs),reference_entries=len(reference)))
    folds=[]
    for fit,valid in GroupKFold(3).split(np.arange(80),groups=[p.name for p in trainfolders]):
        a={trainfolders[i].name for i in fit};b={trainfolders[i].name for i in valid};assert not a&b
        folds.append(dict(fit=sorted(a),validation=sorted(b)))
    decoder_records=json.loads((RESULTS/'decoder_validation.json').read_text())
    assert len(decoder_records['trials'])==16 and len(decoder_records['trips'])==9
    assert {r['trip'] for r in decoder_records['trips']}<=trainids
    assert config['decoder']==max(decoder_records['trials'],key=lambda r:(r['mean_top1_accuracy'],r['mean_continuity']))['parameters']
    comparison=json.loads((RESULTS/'model_comparison.json').read_text());assert len({r['model'] for r in comparison})>=5
    for r in comparison:assert len(json.loads((RESULTS/f'{r["model"]}_cv_trips.json').read_text()))==80
    bayes=replay_bayesian(config)
    save(OUT/'training_policy.json',dict(features=config['feature_names'],features_used_in_splits=sorted(used),trained_models=[r['model'] for r in comparison],bayesian=bayes,decoder_trials=16,decoder_training_trips=decoder_records['trips'],folds=folds,train_counts=counts,evaluator_folder_present=(ROOT/'evaluator_test').exists(),evaluator_accessed=False))
    stored=json.loads((RESULTS/'student_test.json').read_text());old={r['trip']:r for r in stored['trips']};rows=[]
    for folder in studentfolders:
        start=time.perf_counter();trip=read_trip(folder);net=Network(trip)
        ids,frac,x=prepare(trip,net,config['k'],feature_names=config['feature_names']);x=x.astype(np.float32);frac=frac.astype(np.float32)
        raw_probability=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,list(model.classes_).index(1)].reshape(ids.shape)
        raw_probability=np.clip(raw_probability,1e-9,1-1e-9)
        np.testing.assert_array_equal(raw_probability,reload_model.predict_proba(x.reshape(-1,x.shape[-1]))[:,list(reload_model.classes_).index(1)].reshape(ids.shape).clip(1e-9,1-1e-9))
        raw=ids[np.arange(len(ids)),raw_probability.argmax(1)];q=raw_probability/raw_probability.sum(1,keepdims=True)
        decoder=Decoder(net,**config['decoder']);lattice=decoder.lattice(trip.track,ids,frac);predicted,resets=decoder.decode_lattice(ids,q,lattice)
        sections=reconstruct_sections(trip,net,predicted,resets,max_speed=decoder.max_speed,gap_reset=decoder.gap_reset)
        for section in sections:
            if len(section)>1:assert np.array_equal(trip.arcs[section[:-1],1],trip.arcs[section[1:],0])
        # Evaluation starts here, AFTER the complete reference-free prediction.
        alignment=align_reference(trip,net,read_reference(folder,len(trip.arcs)))
        r=dict(ids=ids,labels=alignment.labels,confident=alignment.confident)
        cached=record(folder);columns=[FEATURE_NAMES.index(n) for n in config['feature_names']]
        np.testing.assert_array_equal(x,cached['x'][:,:config['k']][:,:,columns]);np.testing.assert_array_equal(ids,cached['ids'][:,:config['k']])
        m=metrics(r,raw_probability,config['k']);hit=(ids==alignment.labels[:,None]).astype(int).ravel();binary=(raw_probability>=.5).astype(int).ravel()
        precision,recall,f1,_=precision_recall_fscore_support(hit,binary,average='macro',zero_division=0)
        m.update(trip=trip.id,observations=len(ids),classifier_correct=int(np.sum(raw==alignment.labels)),hmm_correct=int(np.sum(predicted==alignment.labels)),
                 hmm_accuracy=float(np.mean(predicted==alignment.labels)),classifier_continuity=lattice_continuity(ids,raw,lattice,decoder),hmm_continuity=lattice_continuity(ids,predicted,lattice,decoder),
                 sequence_exact_match=float(np.array_equal(predicted,alignment.labels)),candidate_accuracy=float(np.mean(hit==binary)),candidate_macro_precision=float(precision),candidate_macro_recall=float(recall),candidate_macro_f1=float(f1),
                 route_sections=len(sections),resets=len(resets),seconds=time.perf_counter()-start)
        for key in ('top1_accuracy','hmm_accuracy','top2_accuracy','top5_accuracy','candidate_miss_rate','log_loss','macro_f1','hmm_continuity','classifier_continuity','sequence_exact_match'):
            assert abs(m[key]-old[trip.id][key])<1e-10,(trip.id,key,m[key],old[trip.id][key])
        np.savez_compressed(OUT/'predictions'/f'{trip.id}.npz',candidate_ids=ids,positive_probabilities=raw_probability,classifier_arc_ids=raw,hmm_arc_ids=predicted,aligned_proxy_arc_ids=alignment.labels,alignment_confident=alignment.confident)
        rows.append(m);print(f'Fresh audit {trip.id}: {len(ids)} observations; metrics match saved results',flush=True)
    total=sum(r['observations'] for r in rows);pooled=dict(classifier=sum(r['classifier_correct'] for r in rows)/total,hmm=sum(r['hmm_correct'] for r in rows)/total)
    summary=dict(status='PASS',label_method='inferred ordered-reference alignment; not independently verified ground truth',mean_per_trip=average(rows),pooled=pooled,total_observations=total,model_sha256=model_fingerprint,metrics_match_existing=True,reference_used_only_after_prediction=True,trips=rows)
    save(OUT/'fresh_metrics.json',summary)
    plots()
    run=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'unit_tests.txt').write_text(run.stdout+run.stderr,encoding='utf-8')
    assert run.returncode==0,'See unit_tests.txt'
    save(OUT/'execution.json',dict(unit_tests_returncode=run.returncode,commands=['python -m pathsnap.audit','python -m unittest discover -s tests -v'],training_executed=False,model_unchanged=hashlib.sha256((ROOT/'artifacts/model.joblib').read_bytes()).hexdigest()==model_fingerprint))
    print(f'Audit PASS: {total} observations, mean HMM={summary["mean_per_trip"]["hmm_accuracy"]:.6f}, pooled HMM={pooled["hmm"]:.6f}',flush=True)

if __name__=='__main__':main()
